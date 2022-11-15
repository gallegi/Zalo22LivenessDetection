import os
import json
from re import T

from .model import Model
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
import comet_ml

# from .logger import Logger
from .checkpointer import Checkpointer
from .logger import Logger
from .general import seed_everything, AverageMeter

def str_to_class(class_name):
    return eval(class_name)

class Trainer():
    PBAR_FORMAT = '{l_bar}{bar:50}{r_bar}{bar:-50b}' 

    def __init__(self, cfg, model:nn.Module,
                train_loader: DataLoader, val_loader: DataLoader,
                optimizer:torch.optim.Optimizer,
                lr_scheduler:torch.optim.lr_scheduler._LRScheduler,
                fold=0):
        self.cfg = cfg
        self.model = model
        self.resume = self.cfg.resume

        self.num_epochs = self.cfg.epochs
        seed_everything(self.cfg.seed) # make reproducible
        self.device = self.cfg.device
        self.model.to(self.device)

        self.fold = fold
        self.exp_name = self.cfg.version_note + '_' + self.cfg.backbone.replace('/', '_')
        self.output_folder = os.path.join(self.cfg.model_dir, self.exp_name, f'fold{self.fold}')
        
        # commet logger
        if self.resume:
            self.logger = comet_ml.ExistingExperiment(
                api_key=self.cfg.comet_api_key,
                project_name=self.cfg.comet_project_name,
                experiment_key=self.cfg.resume_key
            )
        else:
            self.logger = comet_ml.Experiment(
                api_key=self.cfg.comet_api_key,
                project_name=self.cfg.comet_project_name,
            )
            print('New comet experiment with key:', self.logger.get_key())
            self.logger.log_others({'experiment_key': self.logger.get_key()})
        
        self.logger.set_name(self.exp_name + f'_fold{self.fold}')

        # log config
        self.logger.log_parameters(self.cfg.__dict__)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # create an optimizer
        self.optimizer = optimizer
        
        # create a scheduler
        self.lr_scheduler = lr_scheduler

        self.accumulation_steps = self.cfg.accumulation_steps
        self.clip_grad_norm = self.cfg.clip_grad_norm
        self.fp16 = self.cfg.fp16

        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self.batch_index = 0
        self.current_train_step = 0
        self.current_valid_step = 0
        
        self.checkpoint_monitor = self.cfg.checkpoint_monitor
        self.checkpointer = Checkpointer(self.output_folder, save_best_only=self.cfg.save_best_only, logger=self.logger)

        # resume training
        if self.resume:
            self.start_ep = self.checkpointer.resume(self.model, self.optimizer, self.lr_scheduler)
        else:
            self.start_ep = 1


    def fit(self):
        print('Start training ...')
        for current_epoch in range(self.start_ep, self.num_epochs+1):
            with self.logger.train():
                train_monitor = self._train_one_epoch(self.train_loader, current_epoch)
            with self.logger.validate():
                val_monitor = self._val_one_epoch(self.val_loader, current_epoch)

            print(f'===== Epoch {current_epoch} ======')
            print('----- Train metrics -----')
            print(train_monitor)
            print('----- Validation metrics -----')
            print(val_monitor)

            for k, v in train_monitor.items():
                if 'train_'+k == self.checkpoint_monitor:
                    tracked_metric = v
            for k, v in val_monitor.items():
                if 'validate_'+k == self.checkpoint_monitor:
                    tracked_metric = v
            self.checkpointer.update(self.model, self.optimizer, self.lr_scheduler, tracked_metric)
            
            if self.lr_scheduler:
                self.lr_scheduler.step()

    def _train_one_step(self, data):
        if self.accumulation_steps == 1 and self.batch_index == 0:
            self.optimizer.zero_grad()
        if self.fp16:
            with torch.cuda.amp.autocast():
                output = self.model.training_step(data)
        else:
            output = self.model.training_step(data)

        loss = output['loss']
        loss = loss / self.accumulation_steps
        
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        if (self.batch_index + 1) % self.accumulation_steps == 0:
            if self.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.batch_index > 0:
                self.optimizer.zero_grad()

        return output

    def _val_one_step(self, data):
        if self.fp16:
            with torch.cuda.amp.autocast():
                output = self.model.validation_step(data)
        else:
            output = self.model.validation_step(data)

        return output

    def _train_one_epoch(self, data_loader=None, current_epoch=0):
        self.model.train()
        loss_meters = AverageMeter()
        training_step_outputs = []

        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()

        tk0 = tqdm(data_loader, total=len(data_loader), bar_format=self.PBAR_FORMAT)
        for b_idx, data in enumerate(tk0):
            self.batch_index = b_idx
            output = self._train_one_step(data)
            loss_value = output['loss'].item()
            loss_meters.update(loss_value * self.accumulation_steps, data_loader.batch_size)

            self.current_train_step += 1
            tk0.set_postfix(epoch=current_epoch, loss=loss_meters.avg, lr=self.optimizer.param_groups[0]['lr'])
            
            # acummulate output for epoch-end processing
            training_step_outputs.append(output)

        tk0.close()

        # on train epoch end
        monitor = self.model.training_epoch_end(training_step_outputs)
        monitor['loss'] = loss_meters.avg
        self.logger.log_metrics(monitor, epoch=current_epoch)
        return monitor

    def _val_one_epoch(self, data_loader, current_epoch=0):
        self.model.eval()
        loss_meters = AverageMeter()
        validation_step_outputs = []
        
        tk0 = tqdm(data_loader, total=len(data_loader), bar_format=self.PBAR_FORMAT)
        for b_idx, data in enumerate(tk0):
            with torch.no_grad():
                output = self._val_one_step(data)
            loss_value = output['loss'].item()
            loss_meters.update(loss_value, data_loader.batch_size)
            tk0.set_postfix(epoch=current_epoch, loss=loss_meters.avg,)
            self.current_valid_step += 1

            # acummulate output for epoch-end processing
            validation_step_outputs.append(output)

        tk0.close()
        
        monitor = self.model.training_epoch_end(validation_step_outputs)
        monitor['loss'] = loss_meters.avg
        self.logger.log_metrics(monitor, epoch=current_epoch)
        return monitor
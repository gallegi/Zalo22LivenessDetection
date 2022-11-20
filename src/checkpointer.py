import os
import numpy as np
import torch
import glob

class Checkpointer:
    def __init__(self, output_folder, smaller_is_better=True,
                 save_best_only=False, logger=None):
        self.output_folder = output_folder
        self.save_best_only = save_best_only
        self.smaller_is_better = smaller_is_better
        self.best_epoch = 1
        self.current_epoch = 0
        self.logger = logger

        os.makedirs(self.output_folder, exist_ok=True)

        if self.smaller_is_better:
            self.best_value = np.inf
        else:
            self.best_value = 0

    def update(self, model, optimizer, scheduler, current_tracked_metric):
        # update logs
        self.current_epoch += 1

        # save checkpoints
        state_dict = dict()
        state_dict['epoch'] = self.current_epoch
        state_dict['best_epoch'] = self.best_epoch
        state_dict['model'] = model.state_dict()
        state_dict['optimizer'] = optimizer.state_dict()
        state_dict['scheduler'] = scheduler.state_dict()

        if not self.save_best_only:
            state_dict_path = os.path.join(self.output_folder, f'epoch{self.current_epoch:03d}.pth')
            torch.save(state_dict, state_dict_path)

        # update if having better value
        if self.smaller_is_better:
            if current_tracked_metric < self.best_value:
                print(f'Metric decreases from {self.best_value} to {current_tracked_metric}. Overwrite new best checkpoint')
                self.best_epoch = self.current_epoch
                self.best_value = current_tracked_metric
                state_dict_path = os.path.join(self.output_folder, f'best.pth')
                torch.save(state_dict, state_dict_path)
                self.logger.log_others({'best_epoch': self.best_epoch})
        else:
            if current_tracked_metric > self.best_value:
                print(f'Metric increases from {self.best_value} to {current_tracked_metric}. Overwrite new best checkpoint')
                self.best_epoch = self.current_epoch
                self.best_value = current_tracked_metric
                state_dict_path = os.path.join(self.output_folder, f'best.pth')
                torch.save(state_dict, state_dict_path)
                self.logger.log_others({'best_epoch': self.best_epoch})

    def resume(self, model, optimizer, scheduler):
        all_ck_paths = glob.glob(os.path.join(self.output_folder, '*.pth'))
        if len(all_ck_paths) == 0:
            print('No checkpoints found. Start training from scratch')
            return 1
        if not self.save_best_only:
            ck_path = sorted(all_ck_paths)[-1]
        else:
            ck_path = os.path.join(self.output_folder, 'best.pth')

        print('Resume from checkpoint:', ck_path)

        state_dict = torch.load(ck_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])
        self.current_epoch = state_dict['epoch'] + 1
        self.best_epoch = state_dict['best_epoch']

        return state_dict['epoch'] + 1

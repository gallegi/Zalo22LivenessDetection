
# %%
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import Dataset as Dataset
import cv2
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import gc
import random

# %%
VIDEO_DIR = 'data/train/videos'
CSV_PATH = 'data/identified_metadata.csv'
face_crop_csv = pd.read_csv('data/frame14_fd.csv')
OUT_DIR = 'models/cspdarknet_lstm'

os.makedirs(OUT_DIR, exist_ok=True)

# %%
face_crop_csv = face_crop_csv.dropna()
len(face_crop_csv)

# %%
# face_crop_csv

# %%
csv = pd.read_csv(CSV_PATH)
csv = csv[(csv['set'] == 'train')]
len(csv)

# %%
# config
DEVICE = torch.device('cuda:0')
EPOCHS = 50
FOLD_LST = [0]
DIM = (384, 384)
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 2 * TRAIN_BATCH_SIZE
LR = 1e-4
SAMPLE = None
PRETRAINED_NAME = 'darknet53'
seed = 26082000
SMOOTH_THRESH = 0.0

train_transform = A.ReplayCompose(
    [
        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=30, p=0.5), # OK
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5), # OK
        A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=0.5), # OK
        A.Flip(p = 0.5), # Ok
        A.Cutout(num_holes=8, max_h_size=12, max_w_size=12, fill_value=0, p = 0.5), # OK
        A.Resize(DIM[0], DIM[1], always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
        ToTensorV2(),
    ],
)
valid_transform = A.ReplayCompose(
    [
        A.Resize(DIM[0], DIM[1], always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
)

# %%
class LivenessDataset(Dataset):
    def __init__(self, df, video_dir, take_frame = 5, transform=None, input_set ='train'):
        self.df = df.reset_index(drop = True)
        self.take_frame = take_frame
        self.video_dir = video_dir
        self.transform = transform
        self.input_set = input_set

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        zoom_prob = random.uniform(0, 1)
        target= float(self.df.iloc[idx]['liveness_score'])
        if self.input_set == 'train':
            self.transform = A.ReplayCompose(
                [
                    A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.05, rotate_limit=30, p=0.5), # OK
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5), # OK
                    A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=0.5), # OK
                    A.Flip(p = 0.5), # Ok
                    A.Blur(blur_limit = 10, p=0.5),
                    A.ImageCompression(quality_lower=1, quality_upper=10, p=0.5),
#                     A.Cutout(num_holes=8, max_h_size=12, max_w_size=12, fill_value=0, p = 0.5), # OK
                    A.Resize(DIM[0], DIM[1], always_apply=True),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
                    ToTensorV2(),
                ],)
#             target = abs(target - SMOOTH_THRESH)
        fname = self.df.iloc[idx]['fname']
        video_path = os.path.join(self.video_dir, fname)
        video = cv2.VideoCapture(video_path)
        image_lst = []
        frame_number = 0
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = length // self.take_frame
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                if frame is not None and frame_number % frame_step == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.transform is not None:
                        if zoom_prob > 0.5 and self.input_set == 'train':
                            temp_csv = face_crop_csv[face_crop_csv['fname']==fname]
                            if len(temp_csv) > 0:
                                x_min, y_min = temp_csv['xmin'].values[0], temp_csv['ymin'].values
                                x_max, y_max = temp_csv['xmax'].values[0], temp_csv['ymax'].values
                                scale_w, scale_h = temp_csv['width'].values[0], temp_csv['height'].values

                                ori_h, ori_w, _ = frame.shape

                                x_min = x_min * ori_w / scale_w
                                x_max = x_max * ori_w / scale_w
                                y_min = y_min * ori_h / scale_h
                                y_max = y_max * ori_h / scale_h
                                w,h = x_max - x_min, y_max - y_min

                                x_min = x_min - 0.2 * w
                                y_min = y_min - 0.2 * h
                                x_max = x_max + 0.2 * w
                                y_max = y_max + 0.2 * h

                                if x_min < 0:
                                    x_min = 0

                                if x_max >= ori_w:
                                    x_max = ori_w - 1

                                if y_min < 0:
                                    y_min = 0

                                if y_max >= ori_h:
                                    y_max = ori_h - 1
                                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                                frame = frame[y_min:y_max, x_min:x_max]
                            
                        if frame_number == 0:
                            replay_data = self.transform(image=frame)
#                             print(replay_data['replay'])
                            frame = replay_data["image"]
                        else:
                            frame = A.ReplayCompose.replay(replay_data['replay'], image=frame)['image']
                    image_lst.append(frame)
                frame_number += 1
            else:
                break
        video.release()
        image_lst = image_lst[:5]
        return torch.stack(image_lst, axis =0), torch.tensor([target])
#         return image_lst

# %%
# fold = 0
# train_csv = csv[csv['fold'] != fold]
# valid_csv = csv[csv['fold'] == fold]
# train_dataset = LivenessDataset(train_csv, VIDEO_DIR, take_frame = 5, transform= train_transform, input_set = 'train')
# valid_dataset = LivenessDataset(valid_csv, VIDEO_DIR, take_frame = 5, transform= valid_transform, input_set = 'valid')

# train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

# %%
# index = 50
# f = train_dataset[index]
# plt.imshow(f[0][:,:,::-1])
# plt.show()

# plt.imshow(f[1][:,:,::-1])
# plt.show()

# plt.imshow(f[2][:,:,::-1])
# plt.show()

# plt.imshow(f[3][:,:,::-1])
# plt.show()

# plt.imshow(f[4][:,:,::-1])
# plt.show()

# %%
# index = 250
# f = valid_dataset[index]
# plt.imshow(f[0][:,:,::-1])
# plt.show()

# plt.imshow(f[1][:,:,::-1])
# plt.show()

# plt.imshow(f[2][:,:,::-1])
# plt.show()

# plt.imshow(f[3][:,:,::-1])
# plt.show()

# plt.imshow(f[4][:,:,::-1])
# plt.show()

# %%
# f = LivenessDataset(csv, VIDEO_DIR, take_frame = 1, transform= train_transform)
# s = LivenessDataset(csv, VIDEO_DIR, take_frame = 1, transform= None)

# %%
# # 0,3,6,8
# index = 6
# f_img = f[index][0]
# s_img = s[index][0]
# plt.imshow(f_img[:,:,::-1])
# plt.show()

# plt.imshow(s_img[:,:,::-1])
# plt.show()

# %%
class LivenessModel(torch.nn.Module):
    def __init__(self, pretrained_name):
        super(LivenessModel, self).__init__()
        self.backbone = timm.create_model(pretrained_name, pretrained=True)
        if pretrained_name == 'restnet50':
            self.in_feats = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Identity()
        if pretrained_name == 'darknet53':
            self.in_feats = self.backbone.head.fc.in_features
            self.backbone.head.fc = torch.nn.Identity()
        self.lstm = torch.nn.LSTM(self.in_feats, self.in_feats, 2,
                                  bidirectional = True, dropout = 0.3, batch_first = True)
        self.linear = torch.nn.Linear(self.in_feats * 2, 1)
    def forward(self, x):
        b, f, c, h, w = x.shape
        x = torch.reshape(x, (b * f, c, h, w))
        x = self.backbone(x)
        x = torch.reshape(x, (b, f, self.in_feats))
        output, (h, c) = self.lstm(x)
        x = output[:,-1,:]
        x = self.linear(x)
        return x

# %%
import torch.nn as nn
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25, reduction='mean'):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# %%
def calculate_eer(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def calculate_auc(target, predict):
    return roc_auc_score(target, predict)

# %%
def train_fn(model, dataloader, crite, opti, device):
    model.train()
    total_loss = 0
    for i, batch in tqdm(enumerate(dataloader)):
        opti.zero_grad()
        inp, target = batch
        inp = inp.to(device)
        target = target.to(device)
        output = model(inp)
        loss = crite(output, target)
        total_loss += loss
        loss.backward()
        opti.step()
        del inp, target, loss, output
        torch.cuda.empty_cache()
        gc.collect()
        
    return total_loss / len(dataloader)

def valid_fn(model, dataloader, crite, device):
    model.eval()
    total_loss = 0
    target_lst, pred_lst = [], []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            inp, target = batch
            inp = inp.to(device)
            target = target.to(device)
            output = model(inp)
            loss = crite(output, target)
            total_loss += loss
            
            np_output = output.detach().cpu().numpy()
            np_target = target.to(torch.device('cpu'))
            
            pred_lst.append(np_output)
            target_lst.append(np_target)
            del inp, target, loss, output, np_output, np_target
            torch.cuda.empty_cache()
            gc.collect()
    
    gt = np.concatenate(target_lst, axis = 0)
    pred = np.concatenate(pred_lst, axis = 0)
    auc_score = calculate_auc(gt, pred)
    eer = calculate_eer(gt, pred)
    
    return total_loss / len(dataloader), auc_score, eer

# %%
for fold in FOLD_LST:
    train_csv = csv[csv['fold'] != fold]
    valid_csv = csv[csv['fold'] == fold]
    if SAMPLE:
        train_csv = train_csv[:SAMPLE]
        valid_csv = valid_csv[:SAMPLE]
    train_dataset = LivenessDataset(train_csv, VIDEO_DIR, 5, train_transform, 'train')
    valid_dataset = LivenessDataset(valid_csv, VIDEO_DIR, 5, valid_transform, 'valid')
    
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
    
    model = LivenessModel(PRETRAINED_NAME)
    model = model.to(DEVICE)
    opti = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(opti, 'min')
    crite = torch.nn.BCEWithLogitsLoss(reduction='mean')
#     crite = FocalLoss(loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='mean'), gamma = 2.0, alpha = 1.5)()
    
    os.makedirs(f'{OUT_DIR}/fold{fold}', exist_ok = True)
    epoch_loss, valid_epoch_loss = 9999.0, 9999.0
    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_dataloader, crite, opti, DEVICE)
        valid_loss, auc_score, eer = valid_fn(model, valid_dataloader, crite, DEVICE)
        scheduler.step(valid_loss)
        print(f'Epoch {epoch}: train_loss: {train_loss} - valid_loss: {valid_loss} - auc_score: {auc_score} - EER: {eer}')
        
        if eer < epoch_loss:
            epoch_loss = eer
            valid_epoch_loss = valid_loss
            saver = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'auc_score': auc_score,
                'eer': eer,
                'opti': opti,
                'scheduler': scheduler
            }
        
        elif eer == epoch_loss and valid_loss < valid_epoch_loss:
            epoch_loss = eer
            valid_epoch_loss = valid_loss
            saver = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'auc_score': auc_score,
                'eer': eer,
                'opti': opti,
                'scheduler': scheduler
            }
            torch.save(saver, f'{OUT_DIR}/fold{fold}/best.pt')
        with open(f'{OUT_DIR}/fold{fold}/log.txt', 'a+') as f:
            f.write(f'Epoch {epoch}: train_loss: {train_loss} - valid_loss: {valid_loss} - auc_score: {auc_score} - EER: {eer}\n')
            
        if epoch_loss == 0.0:
            break


import os
import argparse
import importlib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, CometLogger

from src.model import LivenessLit
from src.dataset import LivenessDataset

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config_v1',
                    help='config file to run an experiment')
parser.add_argument('--fold', type=int, default=0,
                    help='fold to evaluate')
parser.add_argument('--weight', type=str, default='models/v1_baseline_tf_efficientnet_b0/fold0/epoch=3-val_loss=0.155-val_acc=0.950.ckpt',
                    help='trained weight file')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
CFG = config_module.CFG

CFG.output_dir_name = CFG.version_note + '_' + CFG.backbone.replace('/', '_') 
CFG.output_dir = os.path.join(CFG.model_dir, CFG.output_dir_name)

# Load model and data
model = LivenessLit.load_from_checkpoint(args.weight, cfg=CFG)

df = pd.read_csv(CFG.metadata_file)


train_df = df[df.fold != args.fold]
val_df = df[df.fold == args.fold]
# val_df = val_df[val_df.fname.isin(np.random.choice(val_df.fname.unique(), 10))]


val_ds = LivenessDataset(CFG, val_df, CFG.train_video_dir, CFG.val_transforms)


batch_size = CFG.batch_size
valid_loader = torch.utils.data.DataLoader(val_ds,batch_size=batch_size,num_workers=CFG.num_workers,
                                            shuffle=False,pin_memory=True,drop_last=False)


# logger = CometLogger(api_key=CFG.comet_api_key, project_name=CFG.comet_project_name, experiment_name=CFG.output_dir_name + f'_fold{valid_fold}')

# Predict
trainer = pl.Trainer(default_root_dir=CFG.output_dir,  
                    # logger=logger,
                    accelerator=CFG.accelerator, devices=CFG.devices)


val_preds = trainer.predict(model, dataloaders=valid_loader)


val_preds = torch.cat(val_preds)


val_preds = val_preds.cpu().numpy()


val_df.loc[:, 'prob'] = val_preds

val_df_grouped = val_df.groupby('fname').mean().reset_index()

y = val_df_grouped['liveness_score']
y_pred = val_df_grouped['prob']

os.makedirs(CFG.valid_pred_folder, exist_ok=True)
val_df_grouped.to_csv(os.path.join(CFG.valid_pred_folder, CFG.version_note + '_valid.csv'), index=False)

# Compute EER
fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

_filter = threshold <= 1
plt.plot(threshold[_filter], fnr[_filter], label='FRR')
plt.plot(threshold[_filter], fpr[_filter], label='FAR')
plt.legend()
plt.show()


eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print('Threshold at the intersection of FRR and FAR:', eer_threshold)
print(f'Equal Error Rate (EER) on valid fold {args.fold}:', eer)





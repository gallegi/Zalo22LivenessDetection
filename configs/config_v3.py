import cv2
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

class CFG:
    version_note = 'v3_more_data'

    root_folder = './'
    run_folds = [0]
    accelerator = 'gpu'
    devices = 1
    comet_api_key = 'zR96oNVqYeTUXArmgZBc7J9Jp' # change to your key
    comet_project_name = 'Zalo22Liveness'
    frame_sampling_rate = 15
    im_size = 224

    num_workers=0
    backbone="tf_efficientnet_b0_ns"
    gradient_checkpointing=False
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    # num_cycles=0.5
    # num_warmup_steps=0
    epochs=10
    init_lr=1e-4
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=64
    weight_decay=0.01
    # gradient_accumulation_steps=1
    # max_grad_norm=1000
    seed=67    
    sample = 100
    # patience = 10

CFG.metadata_file = f'{CFG.root_folder}/data/label_sr{CFG.frame_sampling_rate}_frame_5folds.csv'
CFG.train_video_dir = f'{CFG.root_folder}/data/train/videos'
CFG.test_video_dir = f'{CFG.root_folder}/data/public/videos'
CFG.model_dir = f'{CFG.root_folder}/models'
CFG.valid_pred_folder = f'{CFG.root_folder}/valid_predictions'
CFG.submission_folder = f'{CFG.root_folder}/submissions'

# data augmentation and transformations
CFG.train_transforms = A.Compose(
        [
            A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),

            A.Resize(height=CFG.im_size, width=CFG.im_size, always_apply=True),
            A.OneOf(
                [A.CoarseDropout(max_height=16, max_width=16, max_holes=8, p=1), # several small holes
                A.CoarseDropout(max_height=64, max_width=64, max_holes=1, p=1),], # 1 big hole
                p=0.3
            ),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
        ],
        p=1.0,
        
    )


CFG.val_transforms = A.Compose(
        [
            A.Resize(height=CFG.im_size, width=CFG.im_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
        ],
        p=1.0,
       
    )
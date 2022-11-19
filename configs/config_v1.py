import cv2
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

class CFG:
    version_note = 'v1.1_zoomin'

    root_folder = './'
    run_folds = [0] #[0,1,2,3,4]
    device = 'cuda:0'
    comet_api_key = 'zR96oNVqYeTUXArmgZBc7J9Jp' # change to your key
    comet_project_name = 'Zalo22Liveness2'
    frames_per_vid = 10
    im_size = 224

    num_workers=2
    backbone="tf_efficientnet_b0_ns"
    gradient_checkpointing=False
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    
    resume = False
    resume_key = None
    epochs=20
    init_lr=5e-4
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=256
    weight_decay=0.01
    warmup_factor = 10
    fp16 = True
    save_best_only=False
    checkpoint_monitor = 'validate_loss'

    clip_grad_norm = 10
    accumulation_steps = 1

    seed=67    
    sample = None
    patience = 10

CFG.metadata_file = f'{CFG.root_folder}/data/label_{CFG.frames_per_vid}frames_10folds.csv'
CFG.train_video_dir = f'{CFG.root_folder}/data/train/videos'
CFG.train_image_dir = f'{CFG.root_folder}/data/train_frames'
CFG.test_video_dir = f'{CFG.root_folder}/data/public/videos'
CFG.model_dir = f'{CFG.root_folder}/models'
CFG.valid_pred_folder = f'{CFG.root_folder}/valid_predictions'
CFG.submission_folder = f'{CFG.root_folder}/submissions'

# data augmentation and transformations
CFG.train_transforms = A.Compose(
        [   
            A.Downscale(scale_min=0.25, scale_max=0.5, p=0.5),
            A.Affine(scale=(1.5, 2.0), keep_ratio=True, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=CFG.im_size, width=CFG.im_size, always_apply=True),
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
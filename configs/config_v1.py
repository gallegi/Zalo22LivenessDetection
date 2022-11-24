import torchvision
import cv2
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

class CFG:
    version_note = 'v4_metric_learning'

    root_folder = './'
    run_folds = [0] #[0,1,2,3,4]
    device = 'cuda:0'
    comet_api_key = 'zR96oNVqYeTUXArmgZBc7J9Jp' # change to your key
    comet_project_name = 'Zalo22Liveness2'
    im_size = 384
    embedding_size = 512

    frames_per_vid = 3 # number of frames per video to run prediction

    num_workers=2
    backbone="regnet_y_16gf"
    pretrained_weights = torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
    gradient_checkpointing=False
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    
    resume = False
    resume_key = None
    epochs=40
    init_lr=1e-4
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=32
    weight_decay=0.01
    warmup_factor = 10
    fp16 = True
    save_best_only=True
    checkpoint_monitor = 'validate_acc'

    ema = True
    ema_decay = 0.99

    clip_grad_norm = 10
    accumulation_steps = 1

    seed=67    
    sample = None
    patience = 10

CFG.metadata_file = f'{CFG.root_folder}/data/identified_metadata.csv'
CFG.train_video_dir = f'{CFG.root_folder}/data/train/videos'
CFG.test_video_dir = f'{CFG.root_folder}/data/public/videos'
CFG.model_dir = f'{CFG.root_folder}/models'
CFG.valid_pred_folder = f'{CFG.root_folder}/valid_predictions'
CFG.submission_folder = f'{CFG.root_folder}/submissions'

# data augmentation and transformations
CFG.train_transforms = A.Compose(
        [   
            A.OneOf([
                A.Blur(blur_limit = 10, p=1.0),
                A.ImageCompression(quality_lower=30, quality_upper=60, p=1.0),
            ],
            p=0.5),
            A.Downscale(scale_min=0.25, scale_max=0.5, p=0.5),
            A.OneOf([
                A.Compose([
                    A.Resize(width=720, height=1280, always_apply=True),
                    A.RandomCrop(width=504, height=896, p=1.0),
                ],
                p=1.0),
                A.Affine(scale=(1.5, 2.0), keep_ratio=True, p=1.0),
            ],
            p=0.5
            ),
            A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),

            A.OneOf([
                A.Blur(blur_limit = 10, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)
            ],
            p=0.5),

            A.Resize(height=CFG.im_size, width=CFG.im_size, always_apply=True),
            A.OneOf(
                [A.CoarseDropout(max_height=int(CFG.im_size / 15), max_width=int(CFG.im_size / 15), max_holes=8, p=1), # several small holes
                A.CoarseDropout(max_height=int(CFG.im_size / 4), max_width=int(CFG.im_size / 4), max_holes=1, p=1),], # 1 big hole
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
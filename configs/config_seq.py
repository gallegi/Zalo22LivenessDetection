import cv2
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

class CFG:
    version_note = 'lstm'

    root_folder = './'
    run_folds = [0] #[0,1,2,3,4]
    device = 'cuda:0'
    im_size = 384

    backbone="convnext_large"
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
    batch_size=16
    weight_decay=0.01
    warmup_factor = 10
    fp16 = True
    save_best_only=True
    checkpoint_monitor = 'validate_loss'

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

CFG.val_transforms = A.Compose(
        [
            A.Resize(height=CFG.im_size, width=CFG.im_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
        ],
        p=1.0,
       
    )
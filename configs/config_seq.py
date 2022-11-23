import cv2
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

class CFG:
    version_note = 'lstm'
    im_size = 384

    backbone="darknet53"

# data augmentation and transformations
CFG.val_transforms = A.Compose(
        [
            A.Resize(height=CFG.im_size, width=CFG.im_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
        ],
        p=1.0,
       
    )
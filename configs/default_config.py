class CFG:
    version_note = 'v1_baseline'

    root_folder = './'
    run_folds = [0,1,2,3,4]
    accelerator = 'mps'
    devices = 1
    comet_api_key = 'zR96oNVqYeTUXArmgZBc7J9Jp'
    comet_project_name = 'Zalo22Liveness'
    frames_per_vid = 3
    im_size = 224

    num_workers=0
    backbone="tf_efficientnet_b0"
    gradient_checkpointing=False
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=10
    init_lr=1e-4
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=16
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=67    
    sample = 100
    patience = 10

CFG.metadata_file = f'{CFG.root_folder}/data/train/label_per_frame_5folds.csv'
CFG.video_dir = f'{CFG.root_folder}/data/train/videos'
CFG.model_dir = f'{CFG.root_folder}/models/'
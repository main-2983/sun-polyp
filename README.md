# Config model
Config model in `mcode/model.py`  
How to config:  
+ `pretrained`: path to pretrain checkpoint
+ Please change `pretrained=None` to `pretrained=pretrained` in backbone
+ Config decode head `type` to head of your choice

# Training
## Config in training
See `config` section in `main.py`
What to config?
+ Wandb:
  + `use_wandb`: True, False if debug
  + `wandb_key`: Please use your wandb authorize key
  + `wandb_name`: Name of your experiments, please make it as specific as possible
+ Dataset:
    + `train_images`: path to image in Train Dataset
    + `train_masks`: path to mask in Train Dataset
    + `test_folder`: path to Test Dataset
    + `test_images` and `test_masks`: leave it 
    + `num_workers`: torch workers
    + `save_path`: path to save checkpoints and logs
    + `bs`: train batch size
    + `grad_accumulate_rate`: num iters to backward, if `bs=16`, this should be `1`
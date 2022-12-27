# Installation 
## Install pytorch
- Version: 1.10.1 (recommended)  
`conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge`
## Install MMSegmentation
- Install openmim: `pip install openmim`
- Install mmcv: `mim install mmcv-full==1.6.0`
- Install mmseg:
```python
cd sun-polyp
pip install -v -e .
```
## Install dependencies
- Install wandb (for logging): `pip install wandb`
- Install pytorch-lightning: `pip install pytorch-lightning`
- Install segmentation model pytorch: `pip install segmentation-models-pytorch`

# Config
Config everything in `mcode/config.py`  
What to config?
+ Model:
    + Follow `mmseg` config
    + `pretrained`: path to ImageNet pretrained MiT backbone
    + Please change `pretrained` in `backbone` to `pretrained=pretrained`
    + Config model head to head of your choice
+ Wandb:
  + `use_wandb`: True, False if debug
  + `wandb_key`: Please use your wandb authorize key
  + `wandb_name`: Name of your experiments, please make it as specific as possible
  + `wandb_group`: We need 5 runs/experiments, grouping makes it easier to see on wandb UI
+ Dataset:
    + `train_images`: path to image in Train Dataset
    + `train_masks`: path to mask in Train Dataset
    + `test_folder`: path to Test Dataset
    + `test_images` and `test_masks`: leave it 
    + `num_workers`: torch workers
    + `save_path`: path to save checkpoints and logs
    + `bs`: this should be 16 if possible
    + `grad_accumulate_rate`: num iters to backward, if `bs=16`, this should be `1`

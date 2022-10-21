# Config
## English
Config everything in `mcode/config.py`  
How to config?
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

## Tiếng Việt
Config được đặt trong file `mcode/config.py`
Cần chỉnh sửa config của
+ Mô hình
    + Dựa trên config của `mmseg`
    + `pretrained`: đường dẫn tới backbone MiT train với ImageNet
    + Trong trường `backbone`, đặt `pretrained=pretrained`
    + Đặt config model head theo nhu cầu
+ Wandb:
    + `use_wandb`: Đặt True để log, đặt False khi debug mô hình
    + `wandb_key`: dùng key của tài khoản wandb
    + `wandb_name`: Tên thí nghiệm, nên đặt cụ thể
    + `wandb_group`: cần 5 thí nghiệm để tiện theo dõi
+ Dataset:
    + `train_images`: đường dẫn tới ảnh để train
    + `train_masks`: đường dẫn tới mask để train
    + `test_folder`: đường dẫn tới folder train
    + `test_images` và `test_masks`: không cần đặt
    + `num_workers`: số worker xử lí dữ liệu
    + `save_path`: đường dẫn lưu log và checkpoint
    + `bs`: hãy đặt 16
    + `grad_accumulate_rate`: số iter giữa mỗi lần cập nhật tham số, đặt `1` nếu `bs=16`

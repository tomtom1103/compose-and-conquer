# initialize local fuser weights with UniControlnet
python utils/prepare_weights.py init_local_fromuni ckpt/uni.ckpt configs/local_fuser_v1.yaml ckpt/init_local_fromuni.ckpt

# initialize global fuser weights with SD
python utils/prepare_weights.py init_global ckpt/v1-5-pruned.ckpt configs/global_fuser_v1.yaml ckpt/init_global.ckpt

# fuse trained local and global fuser weights for finetuning
python utils/prepare_weights.py integrate v4_8_epoch=28-step=231000.ckpt global_v4_1_epoch=24-step=196000.ckpt configs/cnc_v1.yaml init_cnc.ckpt
python src/train/train.py \
--config-path ./configs/global_fuser_v1.yaml \
--batch-size 4 \
--training-epochs 30 \
--resume-path ./ckpt/init_global.ckpt \
--logdir ./log_global/ \
--log-freq 15000 \
--num-workers 16
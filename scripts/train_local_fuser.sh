python src/train/train.py \
--config-path ./configs/local_fuser_v1.yaml \
--batch-size 4 \
--training-epochs 30 \
--resume-path ./ckpt/init_local_fromuni.ckpt \
--logdir ./log_local/ \
--log-freq 7000 \
--num-workers 16
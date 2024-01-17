python src/train/train.py \
--config-path ./configs/cnc_v1.yaml \
--batch-size 4 \
--training-epochs 9 \
--resume-path ./cnc_v1.ckpt \
--logdir ./log_cnc/ \
--log-freq 7000 \
--num-workers 16
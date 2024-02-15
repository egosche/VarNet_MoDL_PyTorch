GPU_NUM=0
TRAIN_CONFIG_YAML="configs/radial_varnet,k=5.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python train.py \
    --config=$TRAIN_CONFIG_YAML \
    --write_image=10
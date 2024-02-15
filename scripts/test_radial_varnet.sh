GPU_NUM=0
TEST_CONFIG_YAML="configs/radial_varnet,k=5.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python test.py \
    --config=$TEST_CONFIG_YAML \
    --batch_size=1 \
    --write_image=1  # \
    # --workspace="./workspace/base_varnet_layer-03_iter-05_epoch-040_loss-MSE_optim-Adam_lr-0.010000" \
    # --tensorboard_dir="./runs/base_varnet_layer-03_iter-05_epoch-040_loss-MSE_optim-Adam_lr-0.010000"
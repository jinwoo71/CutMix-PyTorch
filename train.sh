python train.py \
    --net_type resnet \
    --dataset imagenet \
    --batch_size 128 \
    --lr 0.1 \
    --depth 18 \
    --epochs 250 \
    --expname ResNet18 \
    -j 40 \
    --beta 1.0 \
    --cutmix_prob 1.0 \
    --method content_style_mixup_loss_labeling \
    --no-verbose

#!/bin/bash
datasetroot="dataset/cifar10/"
logroot="logs/cifar10/"
xhidden=64
xsize=128
yhidden=256
depth=8
level=3
epochs=50
batchsize=2
checkpointgap=10000
loggap=1
savegap=10000
infergap=10000
lr=0.0002
grad_clip=0
grad_norm=10
regularizer=0
adv_loss=True
name='cifar_unt_pyramid'
learn_top=False
only=True
tanh=False
clamp=True
model_path='checkpoints/pyramid_model.pth.tar'
# model_path=''
ref='resnet,vgg,dense'
margin=20.0

python run_attack.py --dataset_root=${datasetroot} --log_root=${logroot} --x_hidden_channels=${xhidden} --x_hidden_size=${xsize} --y_hidden_channels=${yhidden} --flow_depth=${depth} --num_levels=${level} --num_epochs=${epochs} --batch_size=${batchsize} --checkpoints_gap=${checkpointgap} --nll_gap=${loggap} --inference_gap=${infergap} --lr=${lr} --max_grad_clip=${grad_clip} --max_grad_norm=${grad_norm} --save_gap=${savegap}  --regularizer=${regularizer}  --learn_top=${learn_top} --model_path=${model_path} --tanh=${tanh} --only=${only} --name=${name} --targetn=pyramidnet

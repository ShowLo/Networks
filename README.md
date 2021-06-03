# Networks
Lightweight Networks, ACNet, ESNet (our work in progress)

# Usage

```
CUDA_VISIBLE_DEVICES=5 python train.py --batch-size=64 --model=efficientnet-b0 --print-freq=100 --dataset=CIFAR10 --lr=0.3 --lr-decay=cos --warmup-epochs=5 --weight-decay=6e-5 --num-epochs=400 --num-workers=4 --mode=es --branch-nums=4 -polar-mask --polar-t=1 --polar-lambda=5e-5 -learn-mask --init=gauss -same-mask
```

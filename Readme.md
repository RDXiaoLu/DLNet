# DLNet: Direction-Aware Feature Integration for Robust Lane Detection in Complex Environments

## Description

Code for the paper "DLNet: Direction-Aware Feature Integration for Robust Lane Detection in Complex Environments". Zhaoxuan Lu, Lyuchao Liao, Ruimin Li, Fumin Zou, Sijing Cai and Guangjie Han. The paper is submitted to IEEE Transactions on Intelligent Transportation Systems.

## DLNet

![DLNet](https://github.com/RDXiaoLu/DLNet/blob/789f9d315790c2faa76b94bd137a636e9172f890/figs/DLNet.png)

## Results

![Result](https://github.com/RDXiaoLu/DLNet/blob/2b817f3dbdcf0b33d017db9dd87b431dabed0137/figs/result.png)

## Train

Run the following command to train a model on Curvelanes dataset:

```bash
python tools/train.py configs/dlnet/curvelanes/dlnet_curvelanes_dla34.py
```

## References
* [Hirotomusiker/CLRerNet](https://github.com/hirotomusiker/CLRerNet.git)
* [Turoad/CLRNet](https://github.com/Turoad/CLRNet/)
* [SoulmateB/CurveLanes](https://github.com/SoulmateB/CurveLanes.git)
* [CULane Dataset](https://xingangpan.github.io/projects/CULane.html)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)

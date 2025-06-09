# DLNet: Direction-Aware Feature Integration for Robust Lane Detection in Complex Environments

## Description

Code for the paper "DLNet: Direction-Aware Feature Integration for Robust Lane Detection in Complex Environments". Zhaoxuan Lu, Lyuchao Liao, Ruimin Li, Fumin Zou, Sijing Cai and Guangjie Han. The paper is submitted to IEEE Transactions on Intelligent Transportation Systems.
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dlnet-direction-aware-feature-integration-for/lane-detection-on-culane)](https://paperswithcode.com/sota/lane-detection-on-culane?p=dlnet-direction-aware-feature-integration-for)

## DLNet

![DLNet](https://github.com/RDXiaoLu/DLNet/blob/789f9d315790c2faa76b94bd137a636e9172f890/figs/DLNet.png)

## Results

![Result](https://github.com/RDXiaoLu/DLNet/blob/2b817f3dbdcf0b33d017db9dd87b431dabed0137/figs/result.png)

## Train

Run the following command to train a model on Curvelanes dataset:

```bash
python tools/train.py configs/dlnet/curvelanes/dlnet_curvelanes_dla34.py
```

Run the following command to train a model on CULane dataset:

```bash
python tools/train.py configs/dlnet/culane/dlnet_culane_dla34.py
```

## References
* [Hirotomusiker/CLRerNet](https://github.com/hirotomusiker/CLRerNet.git)
* [Turoad/CLRNet](https://github.com/Turoad/CLRNet/)
* [CurveLanes Dataset](https://github.com/SoulmateB/CurveLanes.git)
* [CULane Dataset](https://xingangpan.github.io/projects/CULane.html)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)

## Acknowledgments  
* We would like to express our sincere gratitude to the developers of [**CLRerNet**](https://github.com/hirotomusiker/CLRerNet.git) for their invaluable contributions to the field. Our core improvements can be easily integrated into the CLRerNet framework, enhancing its capabilities for lane detection tasks.

*  We thank the [**CULane**](https://github.com/SoulmateB/CurveLanes.git) and [**CurveLanes**](https://github.com/SoulmateB/CurveLanes.git) teams for providing their datasets, which were essential for our research on lane detection.

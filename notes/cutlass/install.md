#### 安装Cutlass

## Prerequisites

CUTLASS requires:

- NVIDIA CUDA Toolkit (11.4 or later required, [12.0](https://developer.nvidia.com/cuda-toolkit) recommended)
- CMake 3.18+
- host compiler supporting C++17 or greater (minimum g++ 7.5.0)
- Python 3.6+

首先配置环境，这里使用docker，nvidia官方提供了一些[Docker](https://hub.docker.com/r/nvidia/cuda)。

```bash
docker pull nvidia/cuda:12.1.0-base-ubuntu18.04



docker run --runtime nvidia -p 10114:22 -v /raid/mr:/home/mr --name mr-test -ti nvidia/cuda:12.1.0-base-ubuntu18.04
```



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

## 如何编译cutlass应用
[ref](https://github.com/NVIDIA/cutlass/issues/72#issuecomment-674926581)

CUTLASS is a header library. Essentially just need to include all the header files and compile with one nvcc line

nvcc  -Ipath_to_/cutlass/include -Ipath_to_/cutlass/examples/common -Ipath_to/build/include -Ipath_to/cutlass/tools/util/include -isystem=path_to/cuda/include  -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_75,code=[sm_75,compute_75] -std=c++11 -x cu -c path_to_cu -o path_to_binary

You can try below command to see an example

make 08_turing_tensorop_gemm VERBOSE=1

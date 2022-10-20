[参考](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

参考信息来自CUDA官网文档，这里只是对自己觉得有价值的地方进行摘录。后续考虑整理。

##### CUDA计算应用

![CUDA computing Applications](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/gpu-computing-applications.png)

#### A Scalable Programming Model

> The advent of multicore CPUs and manycore GPUs means that mainstream processor chips are now parallel systems. The challenge is to develop application software that transparently scales its parallelism to leverage the increasing number of processor cores, much as 3D graphics applications transparently scale their parallelism to manycore GPUs with widely varying numbers of cores.

CUDA parallel programming model就是为此而生的。其中的核心概念：a hierarchy of thread groups，shared memories，barrier synchronization。

这些抽象支持了细粒度的数据并行和线程并行。（fine-grained data parallelism and thread parallelism)，和粗粒度的数据并行和task parallelism。也许是一个理解上误区，对我而言线程是一个概念，这里的thread可能更偏向计算的含义。data and compute ，fine-grained and coarse-grained。

一个名词， SMs (Streaming Multiprocessors)，这里牵扯到GPU的硬件实现，A GPU is built around an array of Stream Multiprocessor。A multi threaded program is partitioned into blocks of threads that execute independently from each other。

##### 文档结构

CUDA官方文档分为若干部分，按顺序依次为introduction、cuda编程模型概述、编程接口、硬件实现、高性能指导。

另外还有一些附录内容，介绍CUDA支持的设备、C++语言扩展、不同cuda thread 的groups间的同步原语、CUDA同步编程、虚拟内存管理、Stream Ordered内存分配释放、Graph Memory Node、数学函数、C++语言支持、Texture Fetching、Compute Capabilities、Driver API、CUDA Environment Variables、Unified Memory Programming。

##### hardware Implementation

Nvidia GPU是由弹性的 multi threaded Streaming multiprocessor（SMs)构成。一个multirprocessor可以同时执行上百个threads。为了管理这些threads，GPU有一个SIMT(Single-Instruction Multiple-Thread)的架构。指令是流水线的，可以在单thread上进行instruction-level的并行，也可以扩展为thread-level parallelism（多个multi threading )。不同于CPU core，他们 issued in oreder and 没有分支预测或者 speculative execution。这里想到了当初高级计算机体系结构课程。。。。emmm发现课程还是得好好学。

The NVIDIA GPU architecture uses a little-endian representation.（采用小端表示）

#### SIMT Architecture

```
#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      std::cout << "    CUDA Error occurred: " << err << std::endl;            \
      std::cout << "    at line " << __LINE__ << std::endl;                    \
      std::exit(1);                                                            \
    }                                                                          \
  }

#define CUDNN_CALL(exp)                                                        \
  {                                                                            \
    cudnnStatus_t status = (exp);                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      std::cout << "Error on line " << __LINE__ << ": "                        \
                << cudnnGetErrorString(status) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }
```

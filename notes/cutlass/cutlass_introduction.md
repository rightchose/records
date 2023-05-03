### Cutlass是什么？

[GitHub - NVIDIA/cutlass: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass)

参考其github仓库主页介绍，CUTLASS是基于CUDA C++模板抽象的高性能矩阵和矩阵乘法（GEMM）的实现以及CUDA不同level以及scales的相关计算。其涉及到了多层解耦、数据移动的一些策略，这点类似于cuBLAS以及cuDNN。

CUTLASS通过C++模板类将“moving parts”抽象为reusable、modular software component。

用于不同level并行概念的Primitives可以被指定以及tune，通过tiling size、data types 以及其他算法策略。由此带来的灵活性简化了它们在自定义kernel以及应用中构建 blocks的使用。

为了支持形形色色的应用，CUTLASS对混合精度计算提供了大量的支持，

提供了专门的用于FP16、BF、TF32、FP32、FP32 emulation via tensor core instruction、FP64、integer data type（4b 和8b），binary data types（1 b）的data-movement、multiply-accumulate抽象。（这里的理解是CUTLASS支持了很多不同类型的数据，包括binary type，提供了data-movement、multiply-accumulate的接口。

CUTLASS 演示了warp-synchronous 矩阵乘法运算，验证了在NVIDIA的图灵等架构上实现的Tensor cores的可编程性、high-throughput。

目前CUTLASS版本已经到了3.0，引入了新的core library，**CuTE**。

CuTE可以描述和操作tensor的threads和data。CuTE是C++ CUDA模板的抽象集和，用于定义和操作多维布局分层的threads和data。CuTE提供`Layout`和`Tensor`对象，简洁地组装（package）type、shape、memory、space、and layout of data，同时为用户执行复杂的indexing（index操作由CuTE提供？只暴露简单接口给用户？）。这使得程序编写者更关注于算法额度逻辑描述， while CuTe does the mechanical bookkeeping for them。通过这些tools（CuTE？具体不清楚可能是先前的the mechanical bookkeeping），我们可以迅速地设计、实现、修改稠密的线性代数运算。

CuTE的核心抽象是多维布局分层，支持data arrays组成tensor（后面还需要具体了解下这种布局表征，representation of layouts）。这种表征足够强大可以去表示我们所需要实现的所有高效稠密的线性代数。同时该布局还可以通过功能组合（functional composition）进行组合和操作，在此基础上我们构建了大量常见操作，如tiling和partitining。

隐式的GEMM可以认为卷积运算的一种形式。CUTLASS 可以通过反复使用高度优化后的GEMM components去做卷积。（隐式的GEMM

这里参考cuda官方文档的Programming Model章节。

##### Kernel

kernel被调用时会被CUDA Threads并行地执行N次。我们使用`__global__`声明去定义`kernel`,使用`<<<...>>>`去设置CUDA Threads的数目。下面的例子使用了`threadIdx`内置变量。这里的`<<<1,N>>>`表明1个block，N个thread。

```c
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

##### Thread Hierarchy

CUDA内置的变量threadIdx是一个3-componet vector，因此对于一个线程，我们可以使用one-dimensional、two-dimensional、three-dimensional的thread index来描述。于是线程也被组织成one-dimensional、two-dimensional、three-dimensional 的blocks。因此，将计算线程支持一维，二维，三维形式，自然地支持了向量，矩阵，volume的计算。

在确定了线程描述的方式后，我们可以依据threadIdx去得到对应的计算线程id。

>The index of a thread and its thread ID relate to each other in a straightforward way: For a one-dimensional block, they are the same; for a two-dimensional block of size *(Dx, Dy)*,the thread ID of a thread of index *(x, y)* is *(x + y Dx)*; for a three-dimensional block of size *(Dx, Dy, Dz)*, the thread ID of a thread of index *(x, y, z)* is *(x + y Dx + z Dx Dy)*.

```c
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

在上面的例子中，为了计算矩阵相加，这里我们定义了`dim3`类型的`threadsPerBlock`变量。于是，在编写`kernel`函数时，我们可以使用两级索引找到对应的计算线程。但这里依旧`numBlocks`设定为1。

但每个block中能够定义的线程是有限制的。这里需要说下GPU中的线程总数为`blocks` * `number of thread per block`。对于每个block的线程我们可以使用上面的描述。当block内的线程达到一定程度后，我们得考虑使用增大blocks。

> a kernel can be executed by multiple equally-shaped thread blocks。

当我们将代码从单一block扩展到多blocks。这些blocks必须保持相同的shape。同样的，blocks也使用类似线程描述的方式。不过此时的内置变量要换成blockDim。如下图所示。这里我们整理下，对于一组thread，我们将其组织成block。对于一组block，我们将其组织为grid。

![图片](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png)

既然blocks的数目可以增加后，我们上面的代码也可以进行改变。

```C
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

改写后的代码中，每个block有16*16=256个threads。grid里有足够的blocks。

在GPU硬件上，每个block是相互独立执行的，因此它们可以是任意顺序，可以并行也可以串行。编写好的blocks会被SMs执行。

blocks内的thread可以通过共享内存共享数据，并通过同步它们的执行来协调内存访问来进行协作。更仔细点，你可以使用`__synthreads()`作为`barries`，去控制blocks中的threads同步。除此之外，`Cooperative Groups API`提供丰富的thread-synchronization原语。

这里文档多次出现Shared Memory，暂时还没阅读到。不过有一句话，

> For efficient cooperation, the shared memory is expected to be a low-latency memory near each processor core (much like an L1 cache) and __syncthreads() is expected to be lightweight.

共享内存需要是低延迟的，硬件上靠近process core。这里指的是SM的core。同时同步的开销也是轻量的。

##### Memory Hierarchy

首先看下下面的图

![Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/memory-hierarchy.png)

CUDA中thread有local memory， block有block的memory，grid间又有全局的global memory。

也就是说CUDA中的thread在执行过程中会方法不同的内存空间（thread-level, block-level, global）。

同时CUDA中也有两个制度的内存空间，可以供所属有的内存访问：the constant and texture memory spaces。

##### Heterogeneous Programming

谈及CUDA，我们时常提到异构编程，其实也就是CPU+GPU。这里要提及两个术语host，device。可以理解为CPU和GPU。具体如下图，**Note:** Serial code executes on the host while parallel code executes on the device.

![*Heterogeneous Programming*](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/heterogeneous-programming.png)

这里提及Unified Memory Programming。暂时不涉及。

##### Asynchronous SIMT Programming Model

这里涉及异步编程模型。文档这里以NVIDIA Ampere GPU 架构举例，CUDA Programming model通过异步编程模型加速内存访问。异步编程模型定义了与CUDA线程相关的异步操作行为。

> The asynchronous programming model defines the behavior of [Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier) for synchronization between CUDA threads.
>
>  The asynchronous programming model defines the behavior of [Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier) for synchronization between CUDA threads. 

随后介绍了CUDA编程中的异步操作，这块内容就不分析了。

##### Compute Capability

这里主要介绍了计算能力版本，也就是SM version，分为主次版本号。注意这个版本号不同于CUDA Version。
















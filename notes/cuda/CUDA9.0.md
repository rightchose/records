```
With the introduction of NVIDIA Compute Capability 9.0, the CUDA programming model introduces an optional level of hierarchy called Thread Block Clusters that are made up of thread blocks.
Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU.
```

有点类似在blocks层次上引入一个block至于threads的层，提供了blocks间的一些管理。

```
Similar to thread blocks, clusters are also organized into a one-dimension, two-dimension, or three-dimension as illustrated by Figure 5. The number of thread blocks in a cluster can be user-defined, and a maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA. 
```

一些限制，和使用。

```
Thread blocks that belong to a cluster have access to the Distributed Shared Memory. Thread blocks in a cluster have the ability to read, write, and perform atomics to any address in the distributed shared memory
```

distribution shared memory，新的概念

```
A synchronization object could be a cuda::barrier or a cuda::pipeline. These objects are explained in detail in Asynchronous Barrier and Asynchronous Data Copies using cuda::pipeline. These synchronization objects can be used at different thread scopes. A scope defines the set of threads that may use the synchronization object to synchronize with the asynchronous operation.
```

概念 thread scope，对于异步操作需要同步的一组thread。

```
Binary code is architecture-specific. A cubin object is generated using the compiler option -code that specifies the targeted architecture. In other words, a cubin object generated for compute capability X.y will only execute on devices of compute capability X.z where z≥y.

Some PTX instructions are only supported on devices of higher compute capabilities. For example, Warp Shuffle Functions are only supported on devices of compute capability 5.0 and above. The -arch compiler option specifies the compute capability that is assumed when compiling C++ to PTX code. 

```

 Binary Compatibility,  PTX Compatibility


```

PTX code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability. Note that a binary compiled from an earlier PTX version may not make use of some hardware features.
```

指定了arch编译的ptx，无法利用新硬件特性，部分ptx指令时arch相关，制定了arch，那么生成的ptx就不会包含新的硬件ptx指令。

```
When a CUDA kernel accesses a data region in the global memory repeatedly, such data accesses can be considered to be persisting. On the other hand, if the data is only accessed once, such data accesses can be considered to be streaming.
```

reside和streaming概念


```
Thread block clusters introduced in compute capability 9.0 provide the ability for threads in a thread block cluster to access shared memory of all the participating thread blocks in a cluster. This partitioned shared memory is called Distributed Shared Memory, and the corresponding address space is called Distributed shared memory address space. 
```

Distributed Shared Memory的概念

```
Threads that belong to a thread block cluster, can read, write or perform atomics in the distributed address space, regardless whether the address belongs to the local thread block or a remote thread block. 
```

Distributed Shared Memory可以被一个cluster内的thread访问读写以及一些atoimics操作。


```
Whether a kernel uses distributed shared memory or not, the shared memory size specifications, static or dynamic is still per thread block. The size of distributed shared memory is just the number of thread blocks per cluster multiplied by the size of shared memory per thread block.
```

DSmem的大小等于cluster内block的shared mem之和，理解先前Hopper架构前，似乎不同sm间的smem是不互通的，但同时如果支持过多的block Dsmem硬件成本过大，所以支持到8？

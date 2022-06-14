参考CUDA官方文档的Programming Interface章节。

##### Compilation with NVCC

kernel可以以CUDA instruction set architecture编写，也就是PTX。然后以高阶的编程语言编写是更加效率的。nvcc是一个编译驱动，可以简化编译C++ 或 PTX代码。

nvcc编译的源文件包含混合文件（device 和 host）。nvcc的基本工作流包含分离device code 和 host code。随后，

- 将device code转换到PTX code 或者 二进制文件(cubin object)。

- 将host code `<<<>>>`替换成CUDA runtime function calls，从PTX code 或 cubin object去加载并启动每个编译好的 kernel。

  > The modified host code is output either as C++ code that is left to be compiled using another tool or as object code directly by letting nvcc invoke the host compiler during the last compilation stage.

最终修改的host code要么是C++ code（被其他工具编译）或目标文件（在最后的编译阶段被nvcc直接调用host compiler，例如llvm？）。

对于应用而言：

- 链接编译后的host code
- 或忽略host code，使用CUDA driver API，去加载执行PTX code 或 cubin object。

##### just-in-Time Compilation

应用在运行加载的任意PTX 或 NVVM IR code后续都会被编译成binary code（使用device driver）。这就是just-in-time compilation。这样会增加应用的加载时间，但是允许应用受益于新的deivce driver （类似动态链接）。这样是应用运行在非编译器平台外的设备上的唯一方式。

> When the device driver just-in-time compiles some PTX or NVVM IR code for some application, it automatically caches a copy of the generated binary code in order to avoid repeating the compilation in subsequent invocations of the application. The cache - referred to as compute cache - is automatically invalidated when the device driver is upgraded, so that applications can benefit from the improvements in the new just-in-time compiler built into the device driver.

##### Binary Compatibility

二进制代码时架构相关的。二进制代码使用编译选项`-code指定`目标架构平台生成。例如`-code=sm_35`生成compute capability 3.5的二进制代码。

##### PTX Compatibility

一些PTX指令只支持在特定的device上。例如`Warp_Shuffle_Functions`只支持在`compute capability 3.0`以上。`-arch`编译选项指定了compute capability。因此如果代码中有warp shuffle，必须在编译选项中指定`-arch=compute_30`。

> PTX code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability. Note that a binary compiled from an earlier PTX version may not make use of some hardware features. For example, a binary targeting devices of compute capability 7.0 (Volta) compiled from PTX generated for compute capability 6.0 (Pascal) will not make use of Tensor Core instructions, since these were not available on Pascal. As a result, the final binary may perform worse than would be possible if the binary were generated using the latest version of PTX.

##### Application Compatibility

application compatibility依赖binary compatibility和PTX compatibility。 

这一块不是很懂。

##### CUDA Runtime

> The runtime is implemented in the cudart library, which is linked to the application, either statically via cudart.lib or libcudart.a, or dynamically via cudart.dll or libcudart.so

运行时的实现依靠`cudart`库

- Share Memory

- Page-Locked Host Memory(data transfers between host and device memory)
- Asynchronous Concurrent Execution，介绍一些概念和API（用于异步并发编程）
- Multi-Device System：多个device attached to the same host，多卡环境
- Error Checking：如何检查运行时错误
- Call Stack：运行时函数用于管理CUDA C++ 调用栈。
- Texture and Surface Memory
- Graphics Interoperability

##### initialization

> it initializes the first time a runtime function is called

运行时为每个device创建一个CUDA context。这个context对于该device是primary context。会被应用的所有host thread共享。

这里主要介绍CUDA context。

##### Device Memory

> Device Memory can be allocated either as linear memory or as CUDA arrays.

Device Memory可以通过线性内存或者CUDA arrary去分配。

CUDA Array是透明的内存布局（用于texture fetching)。

> Linear memory is allocated in a single unified address space, which means that separately allocated entities can reference one another via pointers, for example, in a binary tree or linked list. The size of the address space depends on the host system (CPU) and the compute capability of the used GPU:



Linear memory使用 single unified addressspace分配。

这一块更像是介绍API，就只列个大纲了。

##### Device Memory L2 Access Management

当CUDA kernel反复访问global memory的一个区域的数据，我们认为这种访问时persisting。相反的，如果data只被访问一次，那么就是streaming。

自CUDA 11.0，compute capability 8.0开始，GPU能够将persiting访问的数据存放再L2 cache中，提高了带宽，降低了延时（针对global memory）。

	###### L2 cache Set-Aside for Persisting Access

L2 cache中的set-aside用于persisting access。获取L2 cache的set-aside 信息可以看看下面代码

```c
cudaGetDeviceProperties(&prop, device_id);                
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/ 
```

当设置GPU为Multi-Instance GPU(MIG)模式时，L2 cache set-aside函数会被禁用。

当使用Multi-Process Service(MPS)时，L2 cache set-aside 的size可以被cudaDeviceSetLimit修改。设定的大小可以通过使用环境变量`CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`，再MPS启动前。

##### L2 Policy for Persisting Access

这里有一个概念，access policy window，这个对象会描述一个块连续的global memory区域，以及这个区域对应的L2 cache的persisting property。

下面代码介绍使用CUDA Stream去设置L2 persisting access window。

```c
cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);  
```

后面还有类似的CUDA GraphKernelNode Example的例子。

##### L2 Access Properties

这里主要有三种类型的access properties：

- `cudaAccessPropertyStreaming`：使用Streaming property进行的内存访问不太可能持久化到L2缓存中，因为这些访问会优先被排除(these access are preferentially evicted)。
- `cudaAccessPropertyPersisting`：使用持久化属性进行的内存访问更有可能持久化到L2缓存中，因为这些访问优先保留在L2缓存的预留部分。
- `cudaAccessPropertyNormal`：此访问属性强制将以前应用的持久访问属性重置为正常状态。从以前的CUDA内核中获得的带有持久化属性的内存访问可能会在其预期使用后很长时间内保留在L2缓存中。这种使用后持久化减少了L2缓存的数量，用于后续不使用持久化属性的内核。用cudaAccessPropertyNormal属性重置访问属性窗口，就像之前的访问没有访问属性一样，删除了之前访问的持久(优先保留)状态。

##### L2 Persistence Example

下面展示CUDA kernels在CUDA Stream下使用persistent access，如何设置L2 cache 的 set-aside，最后reset。

```c
cudaStream_t stream;
cudaStreamCreate(&stream);                                                                  // Create CUDA stream

cudaDeviceProp prop;                                                                        // CUDA device properties variable
cudaGetDeviceProperties( &prop, device_id);                                                 // Query GPU properties
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // Select minimum of user defined num_bytes and max window size.

cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);               // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // Persistence Property
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // Type of access property on cache miss

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream

for(int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);                                 // This data1 is used by a kernel multiple times
}                                                                                           // [data1 + num_bytes) benefits from L2 persistence
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);                                     // A different kernel in the same stream can also benefit
                                                                                            // from the persistence of data1

stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2 

cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     // data2 can now benefit from full L2 in normal mode
            
```

##### Reset L2 Access to Normal

一个先前CUDA Kernel的persisting的L2 cache可能会继续留在L2 cache中，因此，需要reset to normal。这里有三种方式：

- 通过acces property reset 先前的persisting memory region。`cudaAccessPropertyNormal`
- Reset all persisting L2 cache，`cudaCtxResetPersistingL2Cache()`
- 自动复位，但不推荐！！不确定何时会发生。

##### Manage Utilization of L2 set-aside cache

在不同的CUDA流中并发执行的多个CUDA内核可能会为它们的流分配不同的访问策略窗口。然而，L2预留的缓存部分被所有这些并发CUDA内核共享。因此，这个预留缓存部分的净利用率是所有并发内核单独使用的总和。当持久化访问的数量超过预留的L2缓存容量时，将内存访问指定为持久化访问的好处就会减少。（机器翻译）

为了管理 set-aside L2 cache portion，一个应用必须考虑下面的情况

- size of L2 set-aside cache
- CUDA kernel that may concurrently execute
- When and how L2 reset is required to allow normal or streaming accesses to utilize the previously set-aside L2 cache with equal priority.

##### Query L2 cache Properties

数据结构cudaDeviceProp包含了L2 cache 的 Properties。可以使用`cudaGetDeviceProperties`去查询。

CUDA Device Properties include:

- l2CacheSize: The amount of available L2 cache on the GPU.
- persistingL2CacheMaxSize: The maximum amount of L2 cache that can be set-aside for persisting memory accesses.
- accessPolicyMaxWindowSize: The maximum size of the access policy window.

##### Control L2 Cache Set-Aside Size for Persisting Memory Access

The L2 set-aside cache size for persisting memory accesses is queried using CUDA runtime API cudaDeviceGetLimit and set using CUDA runtime API cudaDeviceSetLimit as a cudaLimit. The maximum value for setting this limit is cudaDeviceProp::persistingL2CacheMaxSize.

```c
enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
}; 
```

#### Shared Memory

这里开始介绍共享内存相关的。

这里可以使用`__shared__`memory space specifier进行共享内存分配。

共享内存速度要远超global memory。可以作为临时内存，去减少global memory的方法。

下面一段大妈直接实现矩阵乘法，不适用共享内存。每个线程读取A的一行，B的一列，然后计算C中的对应元素。

![示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/matrix-multiplication-without-shared-memory.png)

```c
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}
```

随后我们给出使用共享内存的矩阵乘法代码。这里先看下基本思路，如下图

![示例图](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/matrix-multiplication-with-shared-memory.png)

```C
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
```


























## cuda编程环境搭建

**计算能力**

每一款GPU都有一个用以表示”计算能力“的版本号。形如$X.Y$的形式，$X$表示主版本号，$Y$表示次版本号。计算能力的版本号决定了GPU硬件所支持的功能，可为应用程序在运行时判断硬件特征提供依据。

**note**：计算能力和性能没有关系。计算能力版本号越大，对应的GPU架构越新。计算能力中的主版本号与GPU的核心架构相关联。

**计算能力和架构代号**

| 主计算能力($X$) | 1     | 2     | 3      | 5       | 6     | 7     | 7.5     |
| ---------- | ----- | ----- | ------ | ------- | ----- | ----- | ------- |
| 架构代号       | Tesla | Fermi | Kepler | Maxwell | Pasal | Volta | Turinig |

**GPU架构和系列对应**

每类架构有那些系列的产品，这里只看Volta和Tutinig（其他偏老）。

| 架构      | Tesla系列 | Quadro系列     | GeForce系列      | Jetson系列   |
| ------- | ------- | ------------ | -------------- | ---------- |
| Volta   | Tesla V |              |                | AGX Xavier |
| Turning | Tesla T | Quadro RTX系列 | GeForce 2000系列 |            |

**CUDA编程语言**

CUDA编程语言最初主要是基于C语言，但目前越来越多地支持C++语言。CUDA提供了两种API，即CUDA driver api 和 CUDA runtime api。其中，CUDA runtime API是在driver api上构建的更高级的api。两者在性能上几乎没有差别。

**CUDA版本**

CUDA版本类似计算能力版本也是$X.Y$的格式，但不等同于GPU的计算能力。可以这样理解：CUDA版本是GPU软件开发平台的版本，而计算能力对应着GPU硬件架构的版本。

**GPU的驱动模式**

WDDM（Windows display driver model）和TCC（Tesla computer cluster）。其中，TCC仅在Tesla、Quadro、Titan系列的GPU中可选。

```bash
sudo nvidia-smi -g GPU_ID -dm 0 # 设置为WDDM模式
sudo nvidia-smi -g GPU_ID -dm 1 # 设置为TCC模式
```

**GPU的计算模式**

GPU的计算模式默认是Default。默认模式下，同一个GPU中允许多个计算进程，但每个进程对应用程序的运行速度一般来说会降低。还有一种模式为E.Process。指的是独占进程模式（exclusive process mode）。但不适用于处于WDDM模式的GPU。独占模式下，只能运行一个计算进程独占该GPU。

```bash
sudo nvidia-smi -i GPU_ID -c 0 # default
sudo nvidia-smi -i GPU_ID -c 1 # 独占进程模式
```

##### nvcc编译CUDA程序

CUDA的编译器驱动nvcc先将源代码分离为主机代码和设备代码。主机代码完整地支持C++语法。而设备代码只部分支持C++。nvcc先将设备代码编译成PTX（parallel thread exection）伪汇编代码，再将PTX代码编译为二进制cubin目标代码。再将源代码编译为PTX代码时，需要用选项`-arch=compute_XY`指定一个虚拟架构的计算能力，用以确定代码能够使用的计算能力。将PTX编译cubin代码时，需要指定`-code=sm_ZW`指定一个真实的架构的计算能力。

可以看到，生成PTX代码，可指定一个虚拟的架构，提供一个较低的计算能力的。但要运行时，需要将PTX代码转成cubin代码，此时必须提供一个真实的架构的计算能力。真实架构的计算能力必须大于或等于虚拟架构计算能力。

```bash
-arch=compute_XY -code=sm_XY
# 简化
-arch=sm_XY
```

##### nvcc即时编译

nvcc有一种即时编译（just-in-time compilation）的机制，可以在运行可执行文件时从其中保留的PTX代码临时编译出一个cubin目标代码。

此时的编译指令选项：

```bash
-gencode arch=compute_XY code=compute_XY
```

## cuda中线程组织

##### 使用核函数的CUDA程序

GPU只是一个设备，其正常工作需要cpu的支持，GPU编程本质是异构编程。对于CPU，我们一般称其为主机host。对于GPU，称为设备device。

```cpp
int main(void)
{
    // 主机代码
    // 核函数代码
    // 主机代码
    return 0;
}
```

CUDA中的核函数与C++中的函数是类似的。但必须使用限定词`__global__`修饰。

```cpp
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n);
}
```

限定词`__global__`和`void`的次序可随意。

完成的code如下。

```cpp
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

可以看到在对核函数`hello_from_gpu`的调用中有`<<<1,1>>>`。这是因为，一块GPU中有很多计算核心（例如Tesla V100中有5120个），可以支持很多线程（thread)）。主机在调用一个核函数时，必须指明需要在设备中指派多少个线程。三个括号中的数分别表示线程块的个数、每个线程块中的线程数。一个核函数的全部线程构成一个网格（grid）。线程块的数据也就是grid size 这里就是1，每个线程块 thread block的大小也就是block size，这里也是1。

在最后调用了`cudaDeviceSynchronize();`这是一个CUDA的 runtime api。去掉该函数，将不能输出字符穿，因为调用输出函数时，输出流时先存放在缓冲区的，而缓冲区不会自动刷新。只有程序遇到某种同步操作时，缓冲区才会刷新。

##### 使用多个线程的核函数

核函数允许指派多个线程（一个GPU往往有上千个计算核心，核函数调用时指定的总的线程数至少等于计算核心数时才能有可能能充分利用GPU中的劝不计算资源，）。

**tip**:实际上总的线程数要大于计算核心才能更充分地利用GPU中的计算资源，这会让计算核内存访问之间及不同计算之间合理重叠，从而减小计算核心空闲时间。

```cpp
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello world from the gpu!\n");
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

这里指定了两个线程块，每个线程块里有4个线程，因此会打印8次相同的文字。

##### 线程索引

一个问题：既然核函数可以指派多个线程，核函数执行时如何知道自己对应那个线程呢？举个例子，linux中fork产生的子进程可以通过getpid获取自己的标识。CUDA编程也有类似的方法。

首先是线程的**组织结构**。线程的组织结构由执行配置exectution configuration，`<<<grid_size, block_size>>>`决定。**一个限制：从开普勒架构开始最大允许的线程块大小为1024，最大允许的网格大小是$2^{31}-1$，采用这个配置大约可以指派2万亿个线程**。一般来说，只要线程数比GPU中得计算核心数（几百到几千）多几倍时，就有可能充分利用GPU中的全部计算资源。

其次，就是每个线程在核函数中都有一个唯一的标识。这里我们使用了grid_size和block_size进行核函数线程配置，在核函数内部，这两个值被存储到gridDim.x（grid_size），blockDim.x（block_size）。

类似地，核函数中预定了如下标识线程的内建变量：

- blockIdx.x：该变量指定一个线程在网格中的线程块指标，其取值范围是从0到gridDim.x-1。

- threadIdx.x：该变量指定一个线程在一个线程块中的线程指标，取值范围从0到blockDim.x。

可以看到用于标识线程的内建变量blockIdx.x ~~ grid_size, threadIdx.x ~~ block_size。

```cpp
#include<stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello world from block %d and thread %d!\n", bid, tid);
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

这段程序输出

```bash
Hello world from block 0 and thread 0! 
Hello world from block 0 and thread 1! 
Hello world from block 0 and thread 2! 
Hello world from block 0 and thread 3! 
Hello world from block 1 and thread 0! 
Hello world from block 1 and thread 1! 
Hello world from block 1 and thread 2! 
Hello world from block 1 and thread 3! 

# or
Hello world from block 1 and thread 0! 
Hello world from block 1 and thread 1! 
Hello world from block 1 and thread 2! 
Hello world from block 1 and thread 3!
Hello world from block 0 and thread 0! 
Hello world from block 0 and thread 1! 
Hello world from block 0 and thread 2! 
Hello world from block 0 and thread 3! 
```

也就是说，有时是第0个线程块先完成计算，有时候是第1个线程块先计算。也就是说，每个线程块的计算是相互独立的。

##### 更复杂的线程组织

细心的读者可能注意到先前的介绍的内建变量，gridDim.x，blockDim.x，blockIdx.x，threadIdx.x。

`blockIdx`，`threadIx`都是类型为`uint3`的变量，该类型是一个结构体，具有x，y，z三个成员。`uint3`在头文件`vector_types.h`中定义：

```cpp
struct __device_builtin__ uint3
{
    unsigned int x, y, z;
};
typedef __device_builtin__ struct uint3 uint3;
```

`gridDim`，`blockDim`是类型为`dim3`的变量。该类型同样具有x，y，z三个成员。同样定义在`vector_types.h`中。

在先前的code中我们使用`<<<2, 4>>>`，2，4分别指定给`gridDim.x`，`blockDim.x`。没有指定的成员`yz`默认为1。这种情况下，网格grid和线程块block都是“一维”的。

下面展示一些dim3的创建。

```cpp
dim3 grid_size(Gx, Gy, Gz);
dim3 block_size(Bx, By, Bz);

// 第三维为1，则可简化
dim3 grid_size(Gx, Gy);
dim3 block_size(Bx, By);
```

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2022-10-19-17-50-33-image.png)

多维网格和线程块本质上还是一维的，就像多维数组本质上也是一维数组。因此，一个多维线程对应到一个一维线程：

```cpp
int tid = threadIdx.z*blockDim.x * blockDim.y +
            threadIdx.y * blockDim.x + threadIdx.x;
```

也就是说，x维度是最内层的（例如如果有两层for循环，那么x就是内层那个，或者说x是变化最快）。z维度是为最外层的。

一段伪代码：

```python
for threadIdx.z in [0,blockDim.z-1]:
    for threadIdx.y in [0,blockDim.y-1]:
        for threadIdx.x in [0,blockDim.x-1]:
            int tid = threadIdx.x + threadIdx.y*blockDim.x 
                        + threadIdx.z * blockDim.x * blockDim.y;
```

这样我们就能定位到一个线程块内的线程，计算`tid`。

当然外层还有一个grid层，也就是线程块的索引。同样地，类似下面的方式计算出`bid`。

```cpp
int bid = blockIdx.z * gridDim.x * gridDim.y
            + blockIdx.y * gridDim.x + blockIdx.x;
```

类似地写下for循环的伪code。

```python
for blockIdx.z in [0,gridDim.z-1]:
    for blockIdx.y in [0,gridDim.y-1]:
        for blockIdx.x in [0,gridDim.x-1]:
            int bid = blockIdx.x + blockIdx.y * gridDim.x
                       + blockIdx.z * gridDim.x * gridDim.y;
```

另外，一个线程块中的线程还可以细分伪不同的线程束（thread warp）。一个线程束（即是一束线程）是同一个线程块中相邻的warpSize个线程。warpSize也是一个内建变量，标识线程束大小，其值对于目前所有的GPU架构都是32。所以一个线程束就是32个连续的线程束。注意这里是针对一个线程块内的线程而言的，具体来说，一个线程块中0~31个线程属于第0个线程束，32~63个线程属于第1个线程束。

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2022-10-19-20-24-27-image.png)

##### 网格和线程块大小限制

CUDA中对能定义的网格大小和线程块大小做了限制。网格grid在x、y、z这三个方向最大允许值分别为$2^{31}-1$，$65536$，$65536$。线程块在x、y、z这三个方向最大允许值分别为1024、1024、64。另外线程块总的大小，即`blockDim.x*blockDim.y*blockDim.z`不能大于1024。

##### cuda中的头文件

nvcc编译`.cu`文件时，将自动包含必要的CUDA头文件，例如`<cuda.h>`和`<cuda_runtime.h>`。而其中`<cuda.h>`包含了`<stdlib.h>`。

## 简单cuda程序基本框架

##### 简单的cuda程序例子

先前我们只介绍了cuda相关的一些概念，但CUDA毕竟是面向科学计算的。这里我们给出一个简单的数组相加的例子。

```cpp
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


const double EPSILON=1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 10000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*)malloc(M);
    double *h_y = (double*)malloc(M);
    double *h_z = (double*)malloc(M);

    for(int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    double *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, M);
    cudaMalloc((void**)&d_y, M);
    cudaMalloc((void**)&d_z, M);
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    cudaMemcpy(d_z, h_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}


void __global__ add(const double *x, const double *y, const double *z)
{
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    z[n] = x[n] + y[n];
}
```

**为什么cudaMalloc()需要一个双重指针作为变量？**

因为cudaMalloc()函数改变指针本身的值，而不改变指针的指向内容。而改变指针本身的值需要双重指针，获取到指针的地址。

**cudaMemcpy()**

该函数API如下

```cpp
cudaError_t cudaMemcpy(void *dst, 
                        const void *src, 
                        size_t count, 
                        enum cudaMemcpyKind kind);
```

其中，`cudaMemcpyKind`有以下几个值：`cudaMemcpyHostToHost`，`cudaMemcpyHostToDevice`，`cudaMemcpyDeviceToHost`，`cudaMemcpyDeviceToDevice`，`cudaMemcpyDefault`。

##### 核函数的要求

- 核函数返回值为void，且要使用`__global__`关键字修饰，支持一些C++中的一些关键字`static`。次序无要求。

- 函数名无特殊要求，支持C++中的重载。

- 不支持可变数量的参数列表。

- 除非使用统一内存编程机制，否则传给核函数的数组（指针）必须指向设备内存。

- 计算能力3.5前，核函数间不能互相调用。3.5后引入动态并行（dynamic parallelism）机制，核函数内部可以调用其他核函数。

- 无论从主机调用，还是设备调用，核函数都是在设备中指向的，调用核函数必须指定执行配置。

##### 核函数中if语句的必要性

先前我们的code中对于`grid_size = N / block_size;`的计算，如果N不能被整除，问题很大，例如，当N为$10^8+1$，block_size为$128$时，`grid_Size`为781250，余数是1，显然`grid_size`应该取781251。但这样先前的代码又会出现越界。因此核函数中需要支持if语句。上面代码几处地方需要修改下：

```cpp
// host code
int grid_size = (N-1) / grid_size + 1;
// kernel code
const int n = blockIdx.x * blockDim.x + threadIdx.x;
if(n >= N) return;
z[n] = x[n] + y[n];
```

##### 自定义设备函数

核函数可以调用不带执行配置的自定义函数，这样的自定义函数称为设备函数（device function）。设备函数再设备中执行，并在设备中被调用。

**函数执行空间标识符**：CUDA程序中，由以下标识符确定一个函数在那里被调用，以及在那里执行：

- `__global__`修饰的函数为核函数，主机调用，设备中执行。如果是动态并行，则也可以在核函数中调用自己或其他核函数。

- `__device__`修饰的函数为设备函数，只能被核函数或其他设备函数调用，设备中执行。

- `__host__`修饰的函数就是主机端的普通C++函数，主机中被调用。对于主机端函数，该修饰符可省略。之所以提供该修饰符，是存在同时使用`__host__`和`__device__`修饰函数，使得该函数既可以一个C++的普通函数，又是一个设备函数。减少冗余代码。编译器将针对主机和设备分别编译该函数。

- 不能使用`__device__`和`__global__`修饰一个函数，即不能将一个函数同时定义为设备函数和核函数。

- 不能使用`__host__`和`__global__`修饰一个函数，即不能将一个函数同时定义为主机函数和核函数。

- 编译器把设备函数当作内联函数（inline function）或费内联函数，但可以使用修饰符`__noinline__`建议一个设备函数作为非内联函数（编译器不一定接受）。也可以使用修饰符`__forceinline__`建议一个设备函数为内联函数。

##### 例子：为数组相加的核函数定义一个设备函数

```cpp
// 版本1，有返回值的设备函数
double __device__ add1_device(const double x, const double y)
{
    return (x + y);
}

void __global__ add1(const double *x, const double *y, double *z, const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        z[n] = add1_device(x[n], y[n]);
    }
}

// 版本2： 用指针的设备函数
void __device__ add2_device(const double x, const double y, double *z)
{
    *z = x + y;
}

void __global__ add2(const double *x, const double *y, double *z, const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        // add2_device(*(x + n), *(y + n), z + n);
        add2_device(x[n], y[n], &z[n]);
    }
}

// 版本3: 用引用的设备函数
void __device__ add3_device(const double x, const double y, double &z)
{
    z = x + y;
}

void __global__ add3(const double *x, const double *y, double *z, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        add3_device(x[n], y[n], z[n]);
    }
}
```

## cuda 程序的错误检测

##### 一个检测CUDA运行时错误的宏函数

```cpp
#pragma once
#include <stdio.h>

#define CHECK(call) \
do  \
{   \
    const cudaError_t error_code = call;    \
    if(erro_code != cudaSuccess) \
    {   \
        printf("CUDA Error:\n");    \
        printf("File: %s\n", __FILE__); \
        printf("Error code: %d\n", error_code); \
        printf("Error text: %s\n", cudaGetErrorString(error_code)); \
        exit(1);    \
    }   \
} while(0);  
```

1. 该文件开头的`@pragma once`是预处理指令，其作用确保当前文件在一个编译单元中不被重复包含。该预处理指令和如下的预处理指令作用相当，但更简洁
   
   ```cpp
   #ifndef ERROR_CUH_
   #define ERROR_CUH_
   // 头文件中的内容
   #endif 
   ```

2. 使用
   
   ```cpp
   CHECK(cudaMalloc((void**) &d_x, M));
   CHECK(cudaMemcpy(d_x, h_x, cudaMemcpyHostToDevice));
   ```

##### 检查核函数

上述方法不能捕获核函数相关错误，因为核函数不返回任何值。但可以通过下面的方法捕捉调用核函数可能发生的错误。

```cpp
CHECK(cudaGetLastError());
CHECK(cudaDeviceSynchronize());
```

第一条语句作用是捕捉第二个语句前的最后一个错误。第二条语句用于同步主机和设备。因为核函数调用是异步的。

**tip：** CUDA编程中的同步处理调用`cudaDeviceSynchronize()`，还有数据传输函数`cudaMemCpy()`。除此外，还可以修改环境变量`CUDA_LAUNCH_BLOCKING`为1。

##### 用CUDA-MEMCHECK检查内存错误

CUDA提供了名为`CUDA-MEMCHECK`的工具集，具体包括`memcheck`、`racecheck`、`initcheck`、`synccheck`。使用如下

```bash
cuda-memcheck --tool memcheck [options] app_name [options]
cuda-memcheck --tool racecheck[options] app_name [options]
cuda-memcheck --tool initcheck[options] app_name [options]
cuda-memcheck --tool synccheck[options] app_name [options]

# 例子
cuda-memcheck ./a.out
```

## gpu加速关键

##### 用CUDA事件计时

CUDA提供了一种基于CUDA事件的计时方式，可以给一段CUDA代码（可能包含主机代码和设备代码）计时。例子如下

```cpp
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));
CHECK(cudaEventRecord(start));
cudaEventQuery(start); // 此处不能使用CHECK宏函数

// 需要计时的代码块

CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
printf("Time = %d ms.\n", elapsed_time);

CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));
```

其中需要注意的是`cudaEventQuery(start)`的调用，这句话对于处于TCC驱动模式下的GPU可以省略。但对于处于WDDM模式来说，必须保留。这是因为，处于WDDM驱动模式的GPU中，一个CUDA流中的操作（如这里的`cudaEventRecord()`函数）并不是直接提交给GPU执行。

## cuda的内存组织

## 全局内存的合理使用

## 共享内存的合理使用

## 原子函数的合理使用

## 线程束基本函数与协作组

## CUDA流

## 统一内存编程

## CUDA标准库

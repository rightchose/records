[参考](https://tvm.apache.org/docs/dev/tutorial/codebase_walkthrough.html)

题外话，不得不说，TVM的文档是我见到的第一个去介绍项目的`Codebase`的。

##### Codebase Structure Overview

在TVM项目的根目录，这里介绍部分子目录的内容

- `src`：C++代码，负责operator compilation和deployment runtime。
- `src/relay`：Relay的实现
- `python`：src目录下C++对象和函数的Python前端
- `src/topi`：针对标准的神经网络的计算定义 and backend schedules。

`src/relay`是负责管理计算图。而对于计算图node的编译和执行则使用`src`目录下其他的部分。`python`提供了和C++ API和driver code相关的python接口 ，用户可以用他们去执行编译。和node相关的operators被注册到`src/relay/op`。`topi`里面实现了operator，代码有C++或python实现。

当用户使用`relay.build`去调用图编译时，图中的node会被执行下面几个步骤：

- 向operator registry查询operator实现
- 为operator生成compute expression 和 schedule
- 将operator编译成目标代码

##### Vector Add Example

```python
n = 1024
A = tvm.te.placeholder((n,), name='A')
B = tvm.te.placeholder9(n,), name='B')
C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name='C')
```

这里的`A`,`B`,`C`都是`tvm.tensor.Tensor`类型，定义在`python/tvm/te/tensort.py`中。Python的`Tensor`的后端为`C++ Tensor`，实现在`include/tvm/te/tensor.h`和`src/te/tensor.cc`。TVM中的所有的Python类型都可以视为和其有着相同名称的底层C++类型的句柄。

去翻阅Python中的`Tensor`类型的定义，你会发现其是`Object`的一个子类。

```python
@register_object
class Tensor(Object, _expr.ExprOp):
    """Tensor object, to construct, see function.Tensor"""

    def __call__(self, *indices):
       ...
```

对象的原型是将C++类型的暴露给Python这类前端语言。TVM实现Python Wrapping的方式并不是直接的。这一块在TVM Runtime System中有，详细的介绍在`python/tvm/_ffi`中。

我们使用`TVM_REGISTER_*`宏去将C++函数暴露给前端语言，in the form of a PackedFunc。`PackedFunc`是另一种机制，TVM用于实现C++和Python之间互操作的机制。这使得从Python调用C++代码十分容易。你也可以检查`FFI Navigator`，这个可以让你在Python 和 C++ FFI之间导航。

一个`Tensor`对象有着一个和其相关的`Operation`对象，定义在`python/tvm/te/tensor.py`，`include/tvm/te/operator.h`，`src/tvm/te/operation`子目录。`Tensor`是它对应的`Operation`对象的输出。相应的，每个`Operation`对象都有着`input_tensor()`方法，会返回它的输入`tensor`列表。因此，我们可以通过这种方式追踪`Operation`的依赖。

我们将和输出张量`C`相关的`Operation`传入`tvm.te.create_schedule()`函数，位于`python/tvm/te/schedule.py`。

```python
s = tvm.te.create_schedule(C.op)
```

类似的，这个python函数对应的C++函数在`include/tvm/schedule.h`中。

```c++
inline Schedule create_schedule(Array<Operation> ops) {
	return Schedule(ops);
}
```

`Scheduler`包含一系列的`Stage`和输出`Operation`。

`Stage`对应到一个`Operation`。在上面的向量相加例子中，有两个`placeholder ops`，一个`compute op`。因此schedule `s`包含三个`Stage`。每个`Stage`包含一个loop nest structure的信息，其类型（`Parallel`，`Vectorized`，`Unrolled`)以及下一个Stage的计算地址（如果有）。

`Schedule`和`Stage`定义在`tvm/python/te/schedule.py`，`include/tvm/te/schedule.h`和`src/te/schedule/schedule_ops.cc`。

为了保持简洁，我们使用`create_schedule()`创建的默认`schedule`s上调用`tvm.build(...)`。

```python
target = "cuda"
fadd = tvm.build(s, [A, B, C], target)
```

`tvm.build()`定义在`python/tvm/driver/build_module.py`，接受`schedule`,`input`,`output`,`target`，返回一个`tvm.runtime.Module`对象，该对象包含一个编译好的函数，可以使用函数调用语法调用。

`tvm.build()`函数可以被划分为下面两步：

- Lowering，将high level，initial loop nest structure转换成final，low level IR。
- Code generation，从low level IR生成目标机器的代码。

Lowering的执行交由`tvm.lower()`函数，定义在`python/tvm/build_module.py`。首先，执行绑定推理（bound inference is performed），an initial loop nest structure is created.

```python
def lower(sch,
          args,
          name="default_function",
          binds=None,
          simple_mode=False):
   ...
   bounds = schedule.InferBound(sch)
   stmt = schedule.ScheduleOps(sch, bounds)
   ...
```

Bound inference干了下面这样一件事，所有的循环绑定和 中间buffers的sizes会被推导确定。如果你的目标为CUDA后端，使用共享内存，它的最小的size会在这一步被确定。Bound inference实现在`src/te/schedule/bound.cc`，`src/te/schedule/graph.cc`和`src/te/schedule/message_passing.cc`。想要更多了解bound inference如何工作，可以去参考[InferBound Pass](https://tvm.apache.org/docs/arch/inferbound.html#dev-inferbound-pass)。

`stmt`，`ScheduleOps`的输出，代表an initial loop nest structure。如果你在你的`schedule`中使用`reorder`或`split`原语，那么initial loop nest会影响它们。`ScheduleOps`定义在`src/te/schedule/schedule_ops.cc`。

接下来，我们对`stmt`使用一些列的lowering passes。这些passes的实现在`src/tir/pass`子目录下。例如，如果你要将`vectorize`或`unroll`原语加入到你的`schedule`中，它们会应用到下面的循环 向量化和unrolling过程中。

```python
...
stmt = ir_pass.VectorizeLoop(stmt)
...
stmt = ir_pass.UnrollLoop(
    stmt,
    cfg.auto_unroll_max_step,
    cfg.auto_unroll_max_depth,
    cfg.auto_unroll_max_extent,
    cfg.unroll_explicit)
...
```

在`lowering`完成后，`build()`函数会生成目标机器上的代码。这些代码包含SSE或AVX指令（如果你的目标机器为`x86`）,PTX指令（CUDA）。除了目标机器代码外，`TVM`也会生成host side code（负责内存管理，kernel启动等）。

代码生成由`build_module()`函数执行，定义在`python/tvm/target/codegen.py`。在C++这边，实现在`src/target/codegen`。`build_module()`Python函数会使用`src/target/codegen/codegen.cc`里的`Build()`函数。

`Build()`这个函数会在`PackedFunc`注册器中寻找目标平台的code generator。例如，注册在`src/codegen/build_cuda_on.cc`的`codegen.build_cuda`函数如下

```python
TVM_REGISTER_GLOBAL("codegen.build_cuda")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildCUDA(args[0]);
  });
```

上面代码中的`BuildCUDA()`会通过定义在`src/codegen/codegen_cuda.cc`中的`CodeGenCUDA`l类使用lowered IR 生成CUDA Kernel，使用NVRTC对kernel进行编译。如果你的目标后端使用LLVM（覆盖x86，ARM，NVPTX，AMDGPU），code generator会使用`CodeGenLLVM`，定义在`src/codegen/llvm/codegen_llvm.cc`。`CodeGenLLVM`将`TVM IR`转换成`LLVM IR`，使用一系列的LLVM优化 passes，生成目标机器的代码。

`Build()`函数返回`runtime::Module`对象，定义在`include/tvm/runtime/module.h`和`src/runtime/module.cc`。`Module`是底层特定目标的`ModuleNode`对象的容器。每个后端实现了一个`ModuleNode`子类，用于目标平台的runtime API调用。例如，对于CUDA后端实现了`CUDAModuleNode`类，定义在`src/runtime/cuda/cuda_module.cc`，用于管理CUDA driver API。`BuildCUDA()`函数会使用`runtime::Module`对`CUDAModuleNode`进行包装，return it to the Python side。LLVM 后端实现了`LLVMModuleNode`，位于`src/codegen/llvm/llvm_module.cc`，其可以解决编译的代码中JIT 执行。其他的`ModuleNode`可以在`src/runtime`中寻找。

返回的`module`，可以被认为是编译函数和device API的combination，被TVM的NDArray对象调用。

```python
dev = tvm.device(target, 0)
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
output = c.numpy()
```

TVM分配设备内存和管理内存变换。为了达到这一点，每个后端都需要提供一个`DeviceAPI`类，定义在`include/tvm/runtime/device_api.h`，包含内存管理方法。例如，CUDA后端实现了`CUDADeviceAPI`，定义在`src/runtime/cuda/cuda_device_api.cc`，使用`cudaMalloc`,`cudaMemcpy`等。

当你第一次调用编译好的模块`fadd(a, b, c)`，`ModuleNode`的`GetFunction()`方法获得一个`PackedFunc`（可以用于kernel调用）。例如，`src/runtime/cuda/cuda_module.cc`，CUDA的后端实现`CUDAModuleNOde::GetFunction()`如下

```c++
PackedFunc CUDAModuleNode::GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) {
  auto it = fmap_.find(name);
  const FunctionInfo& info = it->second;
  CUDAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.launch_param_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}
```

`PackedFunc`的重载`operator()`会被调用，它会调用`CUDAWrappedFunc`的`operator()`（位于`src/runtime/cuda/cuda_module.cc`），在这里我们可以看到`cuLaunchKernel` driver call：

```c++
lass CUDAWrappedFunc {
 public:
  void Init(...)
  ...
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void** void_args) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }
    CUstream strm = static_cast<CUstream>(CUDAThreadEntry::ThreadLocal()->stream);
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    CUresult result = cuLaunchKernel(
        fcache_[device_id],
        wl.grid_dim(0),
        wl.grid_dim(1),
        wl.grid_dim(2),
        wl.block_dim(0),
        wl.block_dim(1),
        wl.block_dim(2),
        0, strm, void_args, 0);
  }
};
```



#### 总结

上面的介绍从一个向量加法开始，介绍了模型从计算图到代码生成，以及最后的部署全过程。后面考虑绘制一下流程图。
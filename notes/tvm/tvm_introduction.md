##### TVM Introduction
[参考](https://tvm.apache.org/docs/tutorial/introduction.html)

现代编译器可以抽象出前端，中端，后端。前端负责将不同语言进行代码生成，产生Intermediate Representation。中端，接受IR，进行不同编译器后端可以共享的优化，如常量替换，死代码消除，循环优化，吐出优化后的IR。后端，接受优化后的IR，进行不同硬件平台相关的指令生成，得到目标文件。

个人理解的这三个阶段划分，前端和后端是类似翻译的作用，前端将不同语言转成同一种描述，后端将该种描述转到不同硬件平台。中端对描述进行了优化。

具体到深度学习编译器，前端负责将不同的机器学习框架模型转成同一种描述。中端负责优化，后端翻译到不同硬件平台。而深度学习中的IR其实就是计算图，Graph IR。前端也就是将不同框架模型的Graph表示转换成 unified Graph IR。整体来说，深度学习编译器和传统编译器的思路一致，唯一的就是大家的翻译内容来源和翻译目标有所区别。其架构图如下。

![tvm架构图](https://tvm.apache.org/images/nnvm/nnvm_compiler_stack.png)



这里放一张TVM官方教程中的图，展示一下TVM优化过程。

![TVM](https://raw.githubusercontent.com/apache/tvm-site/main/images/tutorial/overview.png)

1  → 2：前端将不同模型转换成Graph IR。在TVM为Relay。这一步完成后，深度学习框架相关的东西消失了。

2 &rarr; 3：中端，relay lowers to Tensor Expression(TE)。lowering 也就是from higher-level representation to  lower-level representation。这一步也就将先前的model graph IR 分割成许多的subgraphs，然后再转换成Tensor Expression（TE）。TE是一种domain-specific language，用于描述计算的。TE also provides several *schedule* primitives to specify low-level loop optimizations, such as tiling, vectorization, parallelization, unrolling, and fusion. 

3 &rarr;4：中端，使用auto-tuning module寻找最好的schedule。TVM中目前有的AutoTVM或AutoScheduler。

4 &rarr;5:  中端，选择优化配置用于模型编译。在tuning后，auto-tuning module会生成json format record。这一步会使用这些json去优化每个subgraph。

5&rarr;6：中端，lower to tensor intermediate representation(TIR)，也就是TVM中的low-level intermediate representation。TIR随后会被low-level的optimization passes优化。貌似pass是编译原理里一个术语。

6&rarr;7：后端，优化的TIR会被翻译到目标得硬件平台。TVM支持的后端有LLVM，NVCC等。


### Extra
##### scheduler

TVM可以将深度学习模型中的计算（本质计算图）转化为Graph IR(Relay)，然后通过TVM提供的指令生成模块将Graph IR翻译成特定硬件可执行的指令或者代码。总的来说的TVM的思想可以总结为表示和调度分离，所谓表示就是IR，调度就是scheduler。同时，在高性能计算方面TVM提供了多种调度源语（**scheduler**），包含了大多数常见的优化手段如算子融合，读写缓存，分块计算，并行计算等等，这些计算方法都可以通过scheduler进行实现。

计算过程中使用了**一系列不同的优化手段**，这些优化算法的集合就可以统称为**scheduler**。






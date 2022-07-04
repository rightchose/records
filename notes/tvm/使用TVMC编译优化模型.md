[参考](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html)

首先介绍TVMC，the TVM command line driver。TVMC能够使用TVM的auto-tuning，compiling，profiling 和 execution of model的特性，以命令行的接口。

介绍一下本节内容，

- 编译一个在TVM runtime上的预训练的ResNet-50 v2 model。

- 使用真实图片送入编译的模型，解释模型输出和性能。

- 在CPU上使用TVM tune model。

- 使用TVM收集的tuning data 去 re-compile and optimized model。

- 运行优化的模型，对比模型输出和性能

  
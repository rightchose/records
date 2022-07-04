[原文](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html)

**TVMC**

全称 TVM command line driver，TVM命令行驱动。TVMC可以看做是一个工具，通过它，我们可以以命令行的形式使用TVM的auto-tuning、compiling、profiling以及execute model的一些features。

下面的内容以下面的方式组织

1、编译一个预训练好的ResNet-50 v2模型（for TVM runtime ？只需要TVM runtime支持？）。

2、使用真是图片去测试编译的模型，并解释输出，测量模型性能。

3、使用TVM优化model在CPU上的运行。

4、基于TVM收集到的tuning data重新去编译模型。

5、重新执行第二个环节。

### 预备

TVMC是一个python application， part of TVM python package。安装完tvm的python packge后，可以使用tvmc命令。s


在学习linux下C++时候经常会遇到make，makefile，cmake，nmake这些东西。这里看了知乎的一篇[文章](https://zhuanlan.zhihu.com/p/111110992)。

1 **gcc**
`gcc`可以认为是编译器。早期`gcc`定位为编译C语言，全称为`GNU C Compiler`。后来支持C++、Go等多种编程语言。因此重新定义为`GNU Compiler Collection`，GNU编译器套件。对于一些高级语言而言，程序从代码到可执行文件需要预处理、编译、汇编、连接（以C语言为例）。对于不同的语言，使用`gcc`进行编译时，会调用不同的程序，例如对于C语言，会调用`cc1`，对于C++而言会调用`cc1plus`，对于Object-C会调用`cc1obj`，fortran是`f771`。gcc实际是对这些后台程序的包装，它会根据不同的参数去调用预编译程序`cc1`，汇编器`as`、链接器`ld`。不过这里可以简单理解为`gcc`负责将C或C++代码转换成可执行文件。
2 **make**
`make`是一个智能批处理工具，配合`makeifle`使用。大型项目中，使用`gcc`对代码进行逐个编译时，工作量巨大且繁琐。使用`make`工具依据`makefile`文件编写的规则去处理。
3 **makefile**
`makefile`包含了编译项目中文件的规则。相较对每个文件去单独使用`gcc`会简单很多，但依旧工作量很大，同时换个平台`makefile`又要重新修改。
4 **cmake**
`cmake`可以去生成`makefile`给`make`使用。同时也能跨平台去生成`makefile`。`cmake`会使用到`CMakeLists.txt`文件去生成makefile。
5 **CMakeLists.txt**
这个东西就要由程序员去编写了。
6 **nmake**
nmake是Microsoft Visual Studio中的附带命令，需要安装VS，实际上可以说相当于linux的make

#### 附加
1、除了gcc有时候后，我们还会听到`g++`。
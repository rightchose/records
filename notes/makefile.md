[reference](https://seisman.github.io/how-to-write-makefile/index.html)不错的教程

#### 基础

##### 基本规则

```sh
target ...: prerequisites ...
	command
	...
```

target：可以是一个object file(目标文件)，也可以是一个执行文件，还可以是一个标签（label）。标签有着特殊的性质。

prerequisites：生成该target所依赖的文件 和/或 target。也就是当前target可以依赖先前生成的target（目标文件/执行文件/标签）。

command：生成该target要执行的命令。

> prerequisites中如果有一个以上的文件比target文件要新的话，command所定义的命令就会被执行。
>
> 这就是makefile的规则，也就是makefile中最核心的内容。

一个样例

```makefile
edit : main.o kbd.o command.o display.o \
        insert.o search.o files.o utils.o
    cc -o edit main.o kbd.o command.o display.o \
        insert.o search.o files.o utils.o

main.o : main.c defs.h
    cc -c main.c
kbd.o : kbd.c defs.h command.h
    cc -c kbd.c
command.o : command.c defs.h command.h
    cc -c command.c
display.o : display.c defs.h buffer.h
    cc -c display.c
insert.o : insert.c defs.h buffer.h
    cc -c insert.c
search.o : search.c defs.h buffer.h
    cc -c search.c
files.o : files.c defs.h buffer.h command.h
    cc -c files.c
utils.o : utils.c defs.h
    cc -c utils.c
clean :
    rm edit main.o kbd.o command.o display.o \
        insert.o search.o files.o utils.o
```

上面这个makefile文件在执行`make`命令后，会生成edit目标文件。make会执行文件中的规则，依据target和prerequisites文件的产生时间判断是否执行规则。这里要注意的是`clean`虽然是target，但是它只是一个动作名字，make不会去分析它的依赖性。

##### make如何工作

编写了makefile后我们需要使用make命令，make命令具体做的事情如下：

- make会在当前目录下找名字叫“Makefile”或“makefile”的文件。

- 如果找到，它会找文件中的第一个目标文件（target），在上面的例子中，他会找到“edit”这个文件，并把这个文件作为最终的目标文件。

- 如果edit文件不存在，或是edit所依赖的后面的 `.o` 文件的文件修改时间要比 `edit` 这个文件新，那么，他就会执行后面所定义的命令来生成 `edit` 这个文件。

- 如果 `edit` 所依赖的 `.o` 文件也不存在，那么make会在当前文件中找目标为 `.o` 文件的依赖性，如果找到则再根据那一个规则生成 `.o` 文件。（这有点像一个堆栈的过程）

- 当然，你的C文件和H文件是存在的啦，于是make会生成 `.o` 文件，然后再用 `.o` 文件生成make的终极任务，也就是执行文件 `edit` 了。

  所以，make是依据edit的依赖一层层地去寻找需要执行的命令，类似于树。因此上面的例子中`clean`后面的命令不会被执行，因为edit并不依赖于它。不过我们可以通过执行`make clean`显式地清楚目标文件和中间文件，再执行`make`来重新编译。

  基本上到了这里，我们已经可以处理绝大多数场景了。

  

#### 进阶

这些介绍makefile编写的一些高级技巧

##### 1、变量

上面的makefile文件中，存在大量冗余，比如下面的`.o`文件，很多重复的部分，因此我们可以引入变量。

```makefile
edit : main.o kbd.o command.o display.o \
        insert.o search.o files.o utils.o
    cc -o edit main.o kbd.o command.o display.o \
        insert.o search.o files.o utils.o
```

下面是使用变量

```makefile
objects = main.o kbd.o command.o display.o \
     insert.o search.o files.o utils.o
```

这样原本的makefile我们可以简化为下面的形式，一方面编码减少，另一方面也方便维护，如果后续edit需要增加或删除依赖我们也只需要修改`objects`变量即可。

```makefile
objects = main.o kbd.o command.o display.o \
    insert.o search.o files.o utils.o

edit : $(objects)
    cc -o edit $(objects)
main.o : main.c defs.h
    cc -c main.c
kbd.o : kbd.c defs.h command.h
    cc -c kbd.c
command.o : command.c defs.h command.h
    cc -c command.c
display.o : display.c defs.h buffer.h
    cc -c display.c
insert.o : insert.c defs.h buffer.h
    cc -c insert.c
search.o : search.c defs.h buffer.h
    cc -c search.c
files.o : files.c defs.h buffer.h command.h
    cc -c files.c
utils.o : utils.c defs.h
    cc -c utils.c
clean :
    rm edit $(objects)
```

##### 2、make的自动推导

make看到一个`.o`文件，会自动将`.c`文件加入到依赖，并且相应的`cc -c xxx.c`也会被推导出来。于是上面的makefile进一步简化。例如对于

```makefile
objects = main.o kbd.o command.o display.o \
    insert.o search.o files.o utils.o

edit : $(objects)
    cc -o edit $(objects)
# main.o : main.c defs.h
#     cc -c main.c
main.o : defs.h
kbd.o : defs.h command.h
command.o : defs.h command.h
display.o : defs.h buffer.h
insert.o : defs.h buffer.h
search.o : defs.h buffer.h
files.o : defs.h buffer.h command.h
utils.o : defs.h

.PHONY : clean
clean :
    rm edit $(objects)
```

##### 3、清除目标文件

每个makefile文件都应该有一个清除目标文件的规则。这样不仅便于重编译，也很利于保持文件的清洁。

```makefile
clean:
	rm edit $(objects)
```

更好的做法使用`.PHONY`，表示`clean`是一个伪目标，也就是false target。所以后面有关`clean`的规则项并没有上面的prerequisites。这里我们在`rm`前加上了`-`号，表示，也许某些文件出现问题，但不要管，继续做后面的事。另外，`clean`的规则不要放到文件开头，否则，make的默认目标就是`clean`，显然我们不想如此。因此，一般我们将`clean`放到makefile的最后。

```makefile
.PHONY: clean
clean:
	-rm edit $(objects)
```

##### 4、引用其他makefile

makefile中使用`include`关键字将别的makfile包含进来。基本语法如下

```makefile
include <filename>
```

具体如何使用举个例子。假如你有这样几个makefile：`a.mk`、`b.mk`、`c.mk`还有一个`foo.make`，以及一个变量`$(bar)`，其包含了`e.mk`,`f.mk`。下面的语句

```makefile
include foo.make *.mk $(bar)
```

等价于

```makefile
include foo.make a.mk b.mk c.mk e.mk f.mk
```

对于make命令如何去寻找这些makefile。如果这些文件没有指定绝对路径或相对路径，make会在当前目录下首先寻找，如果当前目录下没有找到，那么make还会在下面的几个目录下找：

- 如果make执行时，有 `-I` 或 `--include-dir` 参数，那么make就会在这个参数所指定的目录下去寻找。
- 如果目录 `<prefix>/include` （一般是： `/usr/local/bin` 或 `/usr/include` ）存在的话，make也会去找。

如果make没有找到对应文件，并不会立即报错，而是生成一条警告信息，它会继续载入其他makefile文件，完成所有makefile文件导入后依旧无法解决，才会出现报错。

同样我们也可以在`include`前加上`-`让make忽略无法读取的文件。

##### 4.1、引入makefile后的make工作方式

//TODO

##### 5、在规则中使用通配符

make支持三个通配符`*`，`?`，`~`。



##### 6、makefile的关键字

`wildcard`，`patsubst`

##### 7、文件搜寻

makefile文件中的特殊变量`VPATH`，如果没有指明该变量，make只会在当前目录搜寻依赖文件和目标文件。定义了该变量，make则会在当前目录找不到的情况下到指定目录中去寻找文件。

```makefile
VPATH = src:../headers
```

上面的定义指定两个目录，“src”和“../headers”，make会按照这个顺序进行搜索。目录由“冒号”分隔。（当然，当前目录永远是最高优先搜索的地方）





##### 5、变量进阶

makefile中的变量定义，可以不按照顺序。因此可以有下面的形式,

```makefile
foo = $(bar)
bar = $(ugh)
ugh = Huh?
```

这样做的可以允许我们将变量的真实值推到后面来定义。

```makefile
CFLAGS = $(include_dirs) -O
include_dirs = -Ifoo -Ibar
```

`include_dirs`会被展开`-Ifoo -Ibar -O`。

但这样也会造成递归定义问题。

```makefile
A = $(B)
B = $(A)
```

但make是有能力检查出这个问题。另外就是在变量种使用函数，这种方式会让我们的make运行时非常慢，更糟糕的是，他会使用得两个make的函数“wildcard”和“shell”发生不可预知的错误。因为你不会知道这两个函数会被调用多少次。

为了避免上面的这种方法，我们可以使用make中的另一种用变量来定义变量的方法 `:=` 操作符：

```makefile
x := foo
y := $(x) bar
x := later
```

等价于

```makefile
y := foo bar
x := later
```

值得一提的是，这种方法，前面的变量不能使用后面的变量，只能使用前面已定义好了的变量。如果是这样：

```makefile
y := $(x) bar
x := foo
```

那么，y的值是“bar”，而不是“foo bar”。



##### 4、makefile里有什么

makefile里主要包含五个东西：显式规则、隐晦规则、变量定义、文件指示和注释。

- 显式规则。也就是上面介绍的规则。
- 隐晦规则。来自于make的自动推导功能。
- 变量定义，类似C语言中的宏。
- 文件指示。包含三个部分，1、引用外部makefile，类似C语言的include。2、根据某些情况指定makefile中有效部分，类似C语言中的预编译#if一样。3、定义一个多行的命令。
- 注释。makefile使用`#`注释。

##### 5、在规则中使用通配符









##### 4、另类风格的makefile



##### 2、.PHONY


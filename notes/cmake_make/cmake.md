cmake一些内置变量

`PROJECT_SOURCE_DIR`：项目目录。

`PROJECT_BINARY_DIR`：执行cmake的目录。二者区别可以参考([CMake中PROJECT_SOURCE_DIR与PROJECT_BINARY_DIR的区别 - 掘金](https://juejin.cn/post/6844903999448055815))

`CMAKE_ROOT`：cmake安装的目录

cmake一些语法记录：

```cmake
# CMake 最低版本号要求
cmake_minimum_required(VERSION X.X)

# 项目信息
project(Demo1)

# 为项目配置CTest/CDash
# 在顶级CMakeLists.txt中配置，导入后，会自动创建一个BUILD_TESTING  option
# 后续可判断是否使用测试
include(CTest)
if(BUILD_TESTING)
    # CMake Code to create tests ...
endif()
# 至于CDash，to enable submission to a CDash Server, create a CTestonfig.cmake
# example
set(CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set(CTEST_SUBMIT_URL "http://my.cdash.org/submit.php?project=MyProject")

# 指定生成目标，将main.cc编译成一个名称为Demo的可执行文件
add_executable(Demo main.cc)
add_executable(Demo main.cc MatchFunction.cc)


# 查找当前目录下的所有源文件，并将名称保存到DIR_SRCS变量
aux_source_directory(. DIR_SRCS)
add_executable(Demo ${DIR_SRCS})

# 添加math子目录，指明本项目包含一个子目录math
# 这样math目录下的CMakeList.txt文件和源代码也会被处理
add_subdirectory(math)

# 添加链接库
# 指明可执行文件Demo需要一个名为MatchFunctions的链接库
target_link_libraries(Demo MatchFunctions)

# 生成链接库，将目录中的源文件编译成静态链接库
add_library(MathFunctions ${DIR_LIB_SRCS})

# 将指定目录添加到编译器的头文件搜索路径下
include_directories(math)

# 将当前目录加入到cmake的include目录
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# set给一般变量赋值
set(FOO, "x") // FOO的作用域为当前作用域
set(FOO, "x", PARENT_SCOPE) // FOO作用域跳上一级

# set赋值给缓存变量（cache variables）
# 缓存变量？在第一次运行cmake时，缓存变量会存放到CmakeCache.txt（编译目录下）
# 再次运行cmake时，这些变量会直接使用缓存值。缓存变量在整个cmake运行过程中都可以起作用。
# 当使用CACHE时，且缓存中没有该变量，变量被创建并存入缓存；
# 如果原缓存中有该变量，也不会改变原缓存中该变量的值，除非后面使用FORCE。

set(FOO, "x" CACHE <type>)
# 原缓存中没有FOO则将FOO赋值为x且存入cache中。
# 原缓存中有FOO则不做任何改变，即便原cache中FOO存的不是x。

set(FOO, "x" CACHE <type><docstring> FORCE) 　　　
# 即便原cache中存在FOO也会创建另一个FOO

# <type>分为以下几种类型：
# FILEPATH/PATH/STRING/BOOL/INTERNAL

# 添加配置头文件，用于处理CMake对源码的设置
# cmake使用config.h.in生成config.h文件
configure_file(
    "${PROJECT_SOURCE_DIR}/config.h.in"
    "${PROJECT_BINARY_DIR}/config.h"
)

# 自定义编译选项,默认ON
option (USE_MYMATH "USE provided math implementation“ ON)

# 选择语句
if (USE_MYMATH)
    include_directories("${PROJECT_SOURCE_DIR}/math")
    add_subdirectory(math)
    set(EXTRA_LIBS ${EXTRAL_LIBS} MathFunctions)
endif(USE_MYMATH)


# 指定MathFunctions库的安装路径
# 生成MathFunctions函数库的libMathFunction.so会被复制到/usr/local/bin
# MathFunctions.h会被复制到/usr/local/include中
# 可以通过修改CMAKE_INSTALL_PREFIX变量来指定文件拷贝到那里
install(TARGETS Mathfunctions DESTINATION bin)
install(FILES Mathfunctions.h DESTINATION include)


# 测试
# 启用测试
enable_testing()

# 测试程序是否成功运行
add_test(test_run Demo 5 2)

# 支持gdb
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


# 添加版本号
set(Demo_VERISON_MAJOR 1)
set(Demo_VERISON_MINOR 0)
```

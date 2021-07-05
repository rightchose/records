vscode打开项目时经常会看到一个.vscode目录，里面有一个`launch.json`和一个`setting.json`文件。

这里看一下我的一个项目中的默认生成的`launch.json`文件

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}", // 当前文件
            "console": "integratedTerminal" // vscode中启动terminal
        }
    ]
}
```

这里只需要关注一下`name`，`type`，`request`，`program`，`args`这几个参数。

`name`提供该debug配置的名称。

`type`调试的类型，这里是python项目。

`request`有两个可选值，`launch`和`attach`，前者会打开这个程序进入调试，后者表示程序已经打开。

`program`表示程序的启动入口。`${file}`表示当前激活的文件。当然也可以使用绝对路径，或者是相对于`workspace`的相对路径，例如`${workspace}/main.py`。

`args`指定要传给python程序的参数。例如

```json
{
	// ...
	"args": [
        "--p", "22"
    ]
    // ...
}
```

其他一些不怎么用的:

`console`指定程序的输出，一半默认`internalConsole`在vscode中启动终端输出。

介绍一下vscode的`task.json`文件。

vscode可以自定义task完成一些代码或者项目生成，编译，测试，打包等的工作。通过这种方式完成代码生产自动化，可以在做这些事情时不用临时再敲命令，或者写代码。

vscode提供[Task auto-detection](https://code.visualstudio.com/docs/editor/tasks#_task-autodetection)，但更多情况下，我们需要依据自己的需求去编写相关的配置文件[Custom tasks](https://code.visualstudio.com/docs/editor/tasks#_custom-tasks)。

`vscode`下，`ctrl+shift+p`输入task，然后选择Configure Task，然后生成一个`tasks.json`文件。

```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "echo",
            "type": "shell",
            "command": "echo Hello"
        }
    ]
}
```

修改一下

```json
{
  // See https://go.microsoft.com/fwlink/?LinkId=733558
  // for the documentation about the tasks.json format
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run tests",
      "type": "shell",
      "command": "./scripts/test.sh",
      "windows": {
        "command": ".\\scripts\\test.cmd"
      },
      "group": "test",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}
```

解释一下参数含义。

`label`任务的名称

`type`任务的类型。对于定制的任务，这个参数可以是`shell`或者`process`。如果是`shell`，则这个命令会被命令行中被执行。例如`bash`，`cmd`，`PowerShell`。

`command`实际执行的命令。

`windows`在windows上执行的command。例如node在windows和linux下的写法。

```json
{
  "label": "Run Node",
  "type": "process",
  "windows": {
    "command": "C:\\Program Files\\nodejs\\node.exe"
  },
  "linux": {
    "command": "/usr/bin/node"
  }
}
```

`group`表明该任务属于那个分组，这里是`test`分组。属于`test`分组的可以通过使用`Run Test Task`执行。

`presentation`定义任务输出如何在用户界面处理。这里`reval`指定`always`表明终端会一直显示输出结果，`panel`设定为`new`表明，对于每个新的任务一个新的终端会被创建。

除此之外还有以下一些参数。

`options`重载默认的`cwd`（current working directory)，`env`（environment variables），`shell`（shell）。可以为每个任务设置，也可以设置`globally`，或者在每个平台上。

`runOptions`定义一个任务什么时候如何去执行。

#### 有时候也需要一些任务的组合

这里[参考](https://code.visualstudio.com/docs/editor/tasks#_compound-tasks)

这里使用`dependsOn`参数。具体使用如下

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Client Build",
      "command": "gulp",
      "args": ["build"],
      "options": {
        "cwd": "${workspaceFolder}/client"
      }
    },
    {
      "label": "Server Build",
      "command": "gulp",
      "args": ["build"],
      "options": {
        "cwd": "${workspaceFolder}/server"
      }
    },
    {
      "label": "Build",
      "dependsOn": ["Client Build", "Server Build"]
       
    }
  ]
}
```

这里首先定义了两个任务，分别为服务端和客户端构建。然后又定义了第三个任务`Build`并指定了`dependsOn`参数，使用前面定义的两个任务，这两个任务并行地执行。

但有时候我们并不像多个任务并行执行，例如在vscode中配置latex的时候，这个时候我们想要任务依据顺序串行执行。这个时候只需要使用`dependsOrder`参数设置为`sequence`。

#### 输出行为

有时候你需要去控制终端面板输出的行为，例如你想最大化你的编辑空间，对于任务的输出只需要知道结果。这个时候需要使用`presentation`参数去控制任务的输出行为。下面是该参数的一些属性。

`reveal`控制终端输出结果是否在前端显示。可选参数`always`一直在前端显示，`never`用户必须使用View》Terminal，`silent`只有当输出没有errors或者warning，才会显示。

`focus`控制终端是否关注于输入。默认`false`。

`echo`控制执行的命令是否输出到终端，默认true。

`showReuseMessage`控制是否显示`Terminal will be reused by tasks, press any key to close it`这段话。

`panel`控制终端是否能够共用。`shared`多个任何可以共享一个终端。`dedicated`为一个特殊任务提供一个单独的终端。当该任务再次执行，依旧使用该终端。`new`每次执行任务都会使用一个新的终端。

`clear`控制在任务执行前，终端是否会clear。默认false。

`group`控制被执行任务属于的分组。

#### 运行行为

[参考](https://code.visualstudio.com/docs/editor/tasks#_run-behavior)

#### 修改vscode中的auto-detected tasks

vscode中默认的一些tasks设置也许不满足你的要求。可以去修改响应的配置文件。[参考](https://code.visualstudio.com/docs/editor/tasks#_run-behavior)

#### 附录

1、vscode中的task.json中，经常会使用到一些Variables Reference。例如`${file}`代表当前文件。

这里可以去看下官方的[文档](https://code.visualstudio.com/docs/editor/variables-reference)

2、有关vscode插件开发可以参考[这里](https://code.visualstudio.com/api)
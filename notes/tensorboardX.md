事实上，这篇文章应该是torch.utils.tensorboard的笔记。

先看下tensorboardX和tensorboard，https://blog.csdn.net/weixin_43002433/article/details/107109776这篇文章中有两者的区别。tensorboardX是在tensorboard上的封装以便在pytorch中使用。

```python
# official
from tensorboard import SummaryWriter
# unofficial
from tensorboardX import SummaryWriter
```

而torch.util.tensorboard时pytorch官方与tensorboard合作的。使用方式如下：

```python
# pytorch with tensorboard official
from torch.utils.tensorboard import SummaryWriter
```

本文主要是介绍官方的。

##### 准备工作

需要有较新版本的pytorch，以及安装tensorboard。tensorboard安装使用下面命令即可。

```
pip install tensorboard
```

##### 写入

tensorboard的使用一般是用其SummaryWriter类。

这样就会在当前目录下创建文件夹，后续对writer的操作都会写入相应的目录

```python
from torch.utils.tensorboard import SummaryWriter
log_dir = 'runs/experiment1'
writer = SummaryWriter(log_dir)
```

这里一般为了监视模型的训练，主要使用SummaryWriter的add_scalar方法。除此之外还有更多方法，例如`add_graph`可视化网络，`add_image`可以在训练过程中增加图片。具体请参考pytorch官方文档https://pytorch.org/docs/stable/tensorboard.html?highlight=torch%20utils%20tensorboard

```python
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```

然后启动tensorboard。

```shell
tensorboard --logdir=runs --port 6666
```

随后终端会输出浏览器访问地址。

##### tensorboard命令参数

`--logdir`为日志文件目录。

`--host`访问地址

`--port`访问端口

`CUDA_VISIBLE_DEVICES=id`指定GPU。否则会默认占用全部GPU。

#### vscode下使用tensorboard

vscode对tensorboard做了很好的支持。当你导入相应的包时，vscode会在相应行提示。直接就可以在vscode中查看，同时也默认支持了remote-ssh。因此很可以很轻松地在本地查看，服务器开发。

#### bug?

在一些场景下，无法正常使用。例如，当仅仅想在tensorboard中查看模型可视化图时，在浏览器访问提示的端口，没有结果。后面试了下加下面的代码，就行了。感觉很奇怪，为什么不支持仅查看模型图。

```
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```


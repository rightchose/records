这里学习下tensorflow的saved model和ckpt的区别，对应的官方文档的[saved model](https://www.tensorflow.org/guide/saved_model?hl=zh-cn)、[检查点](https://www.tensorflow.org/guide/checkpoint?hl=zh-cn)

首先，需要说明“保存TensorFlow 模型”这一短语通常表示保存以下两种元素之一：

- 检查点

- SavedModel

  检查点可以保存模型所使用的参数也就是tf.Varaible对象的确切值。检查点不包含模型所定义计算的任何描述，因此通常仅在将使用保存参数值的源代码可用时才有用。

  另一方面，除了参数值（检查点）之外，SavedModel 格式还包括对模型所定义计算的序列化描述。这种格式的模型独立于创建模型的源代码。因此，它们适合通过 TensorFlow Serving、TensorFlow Lite、TensorFlow.js 或者使用其他编程语言（C、C++、Java、Go、Rust、C# 等 TensorFlow API）编写的程序进行部署。

### 检查点

一些准备工作。

```python
import tensorflow as tf
class Net(tf.keras.Model):
  """A simple linear model."""

  def __init__(self):
    super(Net, self).__init__()
    self.l1 = tf.keras.layers.Dense(5)

  def call(self, x):
    return self.l1(x)
net = Net()
```

使用tf.keras保存默认检查点。

```
net.save_weights('easy_checkpoint')
```

我们也可以手动去编写检查点。

准备

```
def toy_dataset():
  inputs = tf.range(10.)[:, None]
  labels = inputs * 5. + tf.range(5.)[None, :]
  return tf.data.Dataset.from_tensor_slices(
    dict(x=inputs, y=labels)).repeat().batch(2)
```

```
def train_step(net, example, optimizer):
  """Trains `net` on `example` using `optimizer`."""
  with tf.GradientTape() as tape:
    output = net(example['x'])
    loss = tf.reduce_mean(tf.abs(output - example['y']))
  variables = net.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss
```

我们使用`tf.train.Checkpoint`对象手动创建一个检查点，这个对象api如下，也就是我们可以依据自己的需求传入参数，保存需要的东西。有点类似pytorch种的torch.save保存字典类型。可以自定义key-value。

```
tf.train.Checkpoint(
    **kwargs
)
```

下面我们创建检查点，并使用`tf.train.CheckpointManager`去管理。

```
opt = tf.keras.optimizers.Adam(0.1)
dataset = toy_dataset()
iterator = iter(dataset)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
```

使用时我们使用更方便的checkpoint manager对象，下面的函数接受网络和ckpt manager。通过manager我们可以很容易地访问最新的ckpt。

```
def train_and_checkpoint(net, manager):
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  for _ in range(50):
    example = next(iterator)
    loss = train_step(net, example, opt)
    ckpt.step.assign_add(1)
    if int(ckpt.step) % 10 == 0:
      save_path = manager.save()
      print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
      print("loss {:1.2f}".format(loss.numpy()))
```

**加载机制**

TensorFlow 通过从加载的对象开始遍历带命名边的有向计算图来将变量与检查点值匹配。边名称通常来自对象中的特性名称，例如 `self.l1 = tf.keras.layers.Dense(5)` 中的 `"l1"`。[`tf.train.Checkpoint`](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint?hl=zh-cn) 使用其关键字参数名称，如 [`tf.train.Checkpoint(step=...)`](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint?hl=zh-cn) 中的 `"step"`。

![](https://tensorflow.google.cn/images/guide/whole_checkpoint.svg?hl=zh-cn)

优化器为红色，常规变量为蓝色，优化器插槽变量为橙色。其他节点（例如，代表 [`tf.train.Checkpoint`](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint?hl=zh-cn) 的节点）为黑色。



### Saved model

SavedModel 包含一个完整的 TensorFlow 程序——不仅包含权重值，还包含计算。它不需要原始模型构建代码就可以运行，因此，对共享和部署（使用 [TFLite](https://tensorflow.google.cn/lite?hl=zh-cn)、[TensorFlow.js](https://js.tensorflow.google.cn/?hl=zh-cn)、[TensorFlow Serving](https://tensorflow.google.cn/tfx/serving/tutorials/Serving_REST_simple?hl=zh-cn) 或 [TensorFlow Hub](https://tensorflow.google.cn/hub?hl=zh-cn)）非常有用。

可以使用低级`tf.saved_model`保存和加载，高级`tf.keras.Model`API进行。

**Keras**

一些准备。

```python
import os
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()
```

```python
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)
```

```python
file = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
plt.imshow(img)
plt.axis('off')
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis,...])
```

```python
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
```

```python
pretrained_model = tf.keras.applications.MobileNet()
result_before_save = pretrained_model(x)

decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]

print("Result before saving:\n", decoded)
```

保存模型使用下面的代码

```python
mobilenet_save_path = os.path.join(tmpdir, "mobilenet/1/")
tf.saved_model.save(pretrained_model, mobilenet_save_path)
```

上面的代码打印了签名也就是signatures，这个对象是一个字典类型。

**在tensorflow serving中运行Saved Model**

可以通过 Python 使用 SavedModel（下文中有详细介绍），但是，生产环境通常会使用专门服务进行推理，而不会运行 Python 代码。使用 TensorFlow Serving 时，这很容易从 SavedModel 进行设置。






















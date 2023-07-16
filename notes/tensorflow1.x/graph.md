
#### tensorflow中的Graph，GraphDef，MetaGraphDef，Checkpoint
`Meta Graph`, `Frozen Grpah`

[参考](https://www.cnblogs.com/gnivor/p/13747024.html)

##### 1、Graph
Graph是一些Operation和Tensor的集合。

``` python
import tensorflow as tf

# tf.placeholder(dtype, shape=None, name=None)
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.placeholder(tf.float32)
d = a * b + c
e = d * 2
```


##### 2、GraphDef
`GraphDef`是`Graph`的序列化表示。`Graphdef`又是由许多的`NodeDef`的`Protocal Buffer`组成，概念上`NodeDef`与Python Graph的Operation相对应。

``` python
node {
  name: "Placeholder"    # 注：这是一个叫做 "Placeholder" 的node
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "Placeholder_1" # 注：这是一个叫做 "Placeholder_1" 的node
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "mul"          # 注：一个 Mul（乘法）操作
  op: "Mul"
  input: "Placeholder" # 使用上面的node（即Placeholder和Placeholder_1）
  input: "Placeholder_1" # 作为这个Node的输入
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
```

以上三个 NodeDef 定义了两个 Placeholde r和一个Multiply.
Placeholder 通过 attr(attribute的缩写)来定义数据类型和 Tensor 的形状.
Multiply 通过 input 属性定义了两个 placeholder 作为其输入.

##### 3、Meta Graph
Meta graph 的官方解释是：一个 Meta Graph 由一个计算图和其相关的元数据构成, 其包含了用于继续训练，实施评估和(在已训练好的的图上)做前向推断的信息。
实现上，MetaGraph是一个MetaGraphDef（同样由Protocol Buffer来定义），包含四种主要的信息（Buf）：
- MetaInfoDef： 存储一些元信息，比如版本和其他用户信息。
- GraphDef：MetaGraph的核心内容之一
- SaveDef：图的Saver信息（比如最多同时保存的check-point数量，需保存的Tensor名字等，但并不保存Tensor中的实际内容）
- CollectionDef，任何需要特殊注意的 Python 对象，需要特殊的标注以方便import_meta_graph 后取回(如 train_op, prediction 等等)



##### 4、CheckPoint
Checkpoint 里全面保存了训练某时间截面的信息，包括参数，超参数，梯度等等. tf.train.Saver()/saver.restore() 则能够完完整整保存和恢复神经网络的训练.
Checkpoint 分为两个文件保存Variable的二进制信息. ckpt 文件保存了Variable的二进制信息，index 文件用于保存 ckpt 文件中对应 Variable 的偏移量信息.


##### 3、Meta Graph
- Metainfo
- Collectioninfo
- GraphDef

```
tf.train.export_meta_graph
tf.trian.import_meta_graph
```


##### Saved Model
- ckpt
- GraphDef
```
builder = tf.saved_model.Builder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
    builder.add_meta_graph_and_variables(sess, [tags], signature, asserts)
    builder.save()

```

##### Frozen Graph
tvm仅接受Frozen PB作为输入
```
tf.graph_util.convert_variables_to_constants(
    sess, input_graph_def,
    output_node_name,
    variables_name_white_list=None,
    variables_name_blacklist=None
)
```
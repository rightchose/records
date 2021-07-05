### torch.scatter
`torch.scatter`中文含义散开。这个函数理解起来有点麻烦，先按照官方文档。

`Tensor.scatter_(dim, index, src, reduce=None)->Tensor`，将tensor`src`中的所有值依据参数`index`指定的索引写入到`self`中。对于`src`中的每个值，其输出到`self`中的位置索引计算，有两种情况，当`dimension!=dim`时，`src`中的值输出到`self`的索引就是的其在`src`中的位置。当`dimension==dim`时，则将`src`的值依据`index`输出到`self`。

这里依旧云里雾里。看个例子，对于一个3-D张量`self`，

```python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

当dim指定为0时，对于`self`第一维的位置，也就是`dimension=dim`。`src`中的值依据`index`输入到`self`的指定位置。

`self`，`index`，`src`（如果时张量的话）应当有相同维度数。同时`index.size(d)<=src.size(d)`对于任意的d，也就是`index`的每一维度都不能大于`src`。另外，当`d!=dim`时也不能大于`self`的。另外，`index`和`src`是不支持broadcast的。同时`index`中的值也必须要在0和`self.size(dim)-1`之间。这句话很好理解。因为从上面的描述中，不能看出`index`是为了将`src`中的值映射到`self`的`dim`维度的。

最后，`reduce`这个参数为可选参数。可以对所有输入到`self`的`src`的值一个额外的`reduction`操作。

举个栗子，当`reduction`操作为`multiplication`时（等同于`scatter_add_`）：

```python
self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2
```

#### note

另外`backward`只有在`src.shape==index.shape`时有用。

##### official exemplar

```shell
>>> src = torch.arange(1, 11).reshape((2, 5))
>>> src
tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
>>> index = torch.tensor([[0, 1, 2, 0]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
tensor([[1, 0, 0, 4, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 3, 0, 0]])
```

我们可以把`scatter`这个过程想象成有两个篮子，分别为`src`和`self`。我们要做的事是将`src`中的东西按照一个策略`index`放入到`self`中。上面这个例子，就是我们按照公式$self[index[0][i]][i]=src[0][i]$。对于`i`我们从0-5遍历，但是`index`在第二维合法索引只有0-4，这也就是上面的提到的要求，在`dim`指定为0的情境下，`index.size(0)<=src.size(0)`以及`index.size(0)<=self.size(0)`。这么想，上面的代码做的事情实际上就是将`src`的第一行的元素依次取出，然后按照`index`第一行元素的值，得到他们在`self`中应该防止的行数。

```shell
>>> src = torch.arange(1, 11).reshape((2, 5))
>>> src
tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
>>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
tensor([[1, 2, 3, 0, 0],
        [6, 7, 0, 0, 8],
        [0, 0, 0, 0, 0]])
```

再看看这个例子，这次·index有两行了，同时`dim`指定为1了。这个时候依据公式$self[i][index[i][0]]=src[i][0]$和公式$self[i][index[i][1]]=src[i][1]$。首先依据第一个公式，我们从列的角度遍历`src`，对于`src`的第一列我们依次将器按照index的第0列将其放入到`self`中。这里`src`的1，6分别依据`index`的0，0放入到了`self`的每一行的第一列。随后，再看2，7依据1，1依次放入了`self`的每一行的第2列，3，8依据2，4依次放入到了每一行的2，4列。

```
>>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
...            1.23, reduce='multiply')
tensor([[2.0000, 2.0000, 2.4600, 2.0000],
        [2.0000, 2.0000, 2.0000, 2.4600]])
```

再看一个列子，这里指定了`reduce`参数。先不管它，如果则，上面的代码做的事情是将`src`（这里`src`是1.23，并不是张量，在对`src`进行取数的时候每次返回的都是这个标量，可以把它认为是一个万能的张量，但是无论取它上面位置的值，结果都是1.23）中的每一列的依据`index`放入到每一行的每一列，不过受限于`index`。实际上只对`src`的第一列做了这种操作。此时，再看`reduce`参数，设置为`multiply`。也就是将`self`原有的数据和放入的数据相乘。

```
>>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
...            1.23, reduce='add')
tensor([[2.0000, 2.0000, 3.2300, 2.0000],
        [2.0000, 2.0000, 2.0000, 3.2300]])
```

再看这个同上面的例子相同，不过`reduce`参数变成了`add`。

#### 代码实践

那么这么复杂的一个函数，这么复杂的机制，究竟在什么情景下有用呢？

就我所知道的，主要是在目标检测的代码中有关`anchortarget`生成的部分会使用到。

举个场景，我们用图片提取的特征图回归目标检测中的boxes。其中，拿anchor free中的`fcos`（一个目标检测算法）来讲，特征图的每个位置`pos`对应到原始图像中，如果其在原始图像中，落在了目标的框中，我们认为其要去预测这个目标框（当然实际要比这个复杂多了）。

```python
areas_min_ind = torch.min(areas,dim=-1)[1]
reg_targets =torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)    
```

上面这行代码中，`areas`是一个`[bs, h*w, m]`的张量，其中，`h*w`就是特征图的长宽乘积。`m`假定每个图片中，要坚持的目标数目都相同。`area_min_ind`找出`areas`中每个`pos`对应的真实框面积最小的。我们认为，每个`pos`要预测包含其的真实框中最小的真实框（这里简化了很多）。这里的代码首先对`areas_min_ind`进行维度扩张，和`areas`保持一致，也就是上面提到的`self`，`src`，`index`的维度要保持一致。此时，`index`的shape为`[bs, h*w, 1]`。然后设置`dim=-1`也就是2。使用公式$self[i][j][index[i][j][0]]=src[i][j][k]$，其中`src`是1。公式可以写成$self[i][j][index[i][j][0]]=1$，这样其实就是对`self`的第三维度中的m个位置，依据`index`选出一个值为1。

上面可能太抽象了，举个栗子

```python
import numpy as np
import torch
import numpy as np
from numpy.core.numeric import zeros_like
import torch
from torch._C import dtype
bs = 2
h, w = 3, 3
m = 2
areas = np.random.randint(0, 20, (bs, h*w, m))
areas = torch.from_numpy(areas)
areas_min_value,areas_min_ind = torch.min(areas, dim=-1)
masks = torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(-1), 1)
print(masks.float())
print(masks.shape)
target = areas[masks] # [-1]
target = target.reshape((bs, h*w))
print((areas_min_value == target).sum() == (bs* h*w))
```

这段代码证明了，通过`scatter`函数得到了一个掩码`mask`，然后利用`mask`选出了`areas`中最后一维度最小的那些值。和使用`torch.min`结果一致。

`scatter`很适合依据一个规则，对`self`进行处理。

再补充一个`scatter`用于`one-hot`编码。

```python
import torch
bs = 4
num_class = 5
label = torch.tensor([1, 2, 3, 4]) # bs
one_hot = torch.zeros((bs, num_class))
# 这里停顿一下，思考
# 我们这里要做的事情是让one_hot依据label然后在每一行中选择一个位置置为1
# 也就是self[i][index[i][j]] = 1
# 然后i应当0-3，则index也就是label的size应该为[4,1]
# 我们的dim应该是1，同时label也要和one_hot的维度保持一致
one_hot.scatter_(1, label.unsqueeze(-1), 1)
print(one_hot)
```


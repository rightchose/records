这个记录主要是分析以`pytorch`中的提供的分类损失函数。其他的就不分析。

先列出`pytorch`中的分类损失函数。

`nn.BCELoss`，`nn.functional.binary_cross_entropy`，`nn.BCEWithLogitsLoss`，`nn.CrossEntropyLoss`，`nn.functional.mse_loss`，`nn.functional.nll_loss`，`nn.MSELoss`，

`nn.NLLLoss`，`nn.functional.l1_loss`，`nn.functional.nll_loss`。

`pytorch`的loss有两类，一类是模块`nn`下的，一类是`nn.functional`下的。

下面开始介绍可以用于分类的损失函数。

##### 负对数似然函数

负对数似然函数可以用于多分类损失函数。

负对数似然函数依据标签值`y`的值取出预测为`y`的概率，再取负数。例如某3分类问题，对于单个样本，预测的结果为0.1，0.2，0.6。标签为1，那么损失就为，-0.2。

`pytorch`提供的了`nn.NLLoss`和`nn.functional.nll_loss`用于计算。

具体可以参考这个[blog](https://blog.csdn.net/qq_22210253/article/details/85229988)，很详细。

##### 交叉熵损失函数

交叉熵损失函数是再负对数似然函数的基础上，先对预测值进行`softmax`然后再取`log`，然后再应用负对数似然函数。

##### 二分类交叉熵损失函数（BCELoss)

对于一个样本$(x,y)$，算法的预测为$\hat y$，交叉熵损失函数记为$L$。
$$
L = -[y\log {\hat y} + (1-y)log(1-\hat y)]
$$
当$y=1$时，$L=-log\hat y$，此时损失函数的值绘制成图如下:

![](https://raw.githubusercontent.com/rightchose/records/main/assert/bce_loss1.png)

当$\hat y$越接近1的时候损失函数越小，越接近0的时候越大。

对于这个损失函数，`pytorch`中提供了`nn.BCELoss`，`nn.functional.binary_cross_entropy`可以实现

```python
# nn.BCELoss
criterion = nn.BCELoss()
pred = torch.tensor(0.01, dtype=torch.float32)
label = torch.tensor(1, dtype=torch.float32)
loss = criterion(pred, label)
print(loss) 
>> tensor(4.6052)
# nn.functional.binary_cross_entropy
loss nn.fucntional.binary_cross_entropy(pred, label)
print(loss)
>> tensor(4.6052)
```

`pytorch`中还提供了`nn.BCEWithLogitsLoss`和`nn.functional.binary_cross_entropy_with_logits`相对于`BCELoss`会先对数据求个`Sigmoid`在去做`BCELoss`，

其实，`BCELoss`虽然全称为`Binary Cross Entropy Loss`，但实际上也是能做多分类的。

##### huber loss

公式如下
$$
l_{n}=\left\{\begin{array}{ll}
0.5\left(x_{n}-y_{n}\right)^{2}, & \text { if }\left|x_{n}-y_{n}\right|<\text { delta } \\
\text { delta } *\left(\left|x_{n}-y_{n}\right|-0.5 * \text { delta }\right), & \text { otherwise }
\end{array}\right.
$$

##### 平滑L1损失函数

$$
l_{n}=\left\{\begin{array}{ll}
0.5\left(x_{n}-y_{n}\right)^{2} / \text { beta }, & \text { if }\left|x_{n}-y_{n}\right|<b e t a \\
\left|x_{n}-y_{n}\right|-0.5 * \text { beta, } & \text { otherwise }
\end{array}\right.
$$






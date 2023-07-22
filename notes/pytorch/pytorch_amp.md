由于模型出结果要太久，于是尝试使用混合精度去加速模型训练。

##### 1、使用apex

1、安装`apex`

```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

2、代码中使用`apex`

[介绍](https://zhuanlan.zhihu.com/p/79887894)，[快速上手](https://zhuanlan.zhihu.com/p/140347418)

三行代码快速使用，在原有的训练代码中添加下面的内容。

```python
from apex import amp
# Added after model and optimizer construction
model, optimizer = amp.initialize(model, optimizer, flags...)
...
# loss.backward() changed to:
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

##### 2、使用pytorch原生

pytorch自1.6后新增一个子模块`amp`，支持混合精度训练。

[参考](https://bbs.cvmart.net/articles/2807)

使用

```python
scaler = torch.cuda.amp.GradScaler()

# forward
with torch.cuda.amp.autocast():
    pred = model(input)
	loss = loss_fn(pred, gt)

scaler.scale(loss).backward()

scaler.step(optimizer)
scaler.update()
```

##### 3、两种方案对比

可以看看这个链接https://www.codenong.com/cs109485088/。

目前打算在自己的模型上试下这两个。
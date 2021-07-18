#### one hot编码的几种实现方式
##### torch.scatter函数
```
import torch
bs = 4
num_class = 5
label = torch.tensor([1, 2, 3, 4]) # bs
one_hot = torch.zeros((bs, num_class))
one_hot.scatter_(1, label.unsqueeze(-1), 1)
print(one_hot)
```
##### torch.eye
```
label = torch.tensor([2, 1, 0])
one_hot = torch.eye(3)[label]
print(one_hot)
```
##### way3
```
label = torch.tensor([2, 1, 0])
one_hot = (torch.arange(1,5)[None,:]==label[:,None]).float()
print(one_hot)
```

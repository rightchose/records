#### sed

[参考1](https://wangchujiang.com/linux-command/c/sed.html)

[参考2](http://c.biancheng.net/view/4028.html)

功能强大的流式文本编辑器，能够配合正则表达式使用。处理时，把当前处理的行存储在临时缓冲区中，称为“模式空间”（pattern space），接着用`sed`命令处理缓冲区中的内容，处理完后，接着处理下一行，直到文件末尾。除非使用重定向存储输出，否则不会改变文件内容。

###### 基本格式

```
sed [选项] [脚本命令] 文件名
```


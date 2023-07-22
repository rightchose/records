#### wget

[参考](https://blog.csdn.net/freeking101/article/details/53691481)

[参考](https://www.bandwagonhost.net/8072.html)

wget支持http、https、ftp和ftps等常用网络协议去检索文件。

##### 简单使用

我们可以使用下面命令去下载`redis`的源码，使用`-P`参数指定下载目录。

```sh
wget -P /usr/local/src http://download.redis.io/releases/redis-4.0.9.tar.gz
```

如果我们想要下载的文件按照我们的要求命名，可以使用`-O`参数。

```sh
wget  http://download.redis.io/releases/redis-4.0.9.tar.gz  -O /usr/local/src/redis.tar.gz
```

##### 复杂场景

有些文件需要我们去登录用户信息，才能去下载。此时需要用到wget的设置cookie功能。


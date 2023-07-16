#### 参考[B站自制深度学习推理框架](https://www.bilibili.com/video/BV118411f7yM/?spm_id_from=333.999.0.0&vd_source=fde8df1a8f3f532205a406a1d84a479c)

#### 1、环境搭建
系统环境： ubuntu 22.04

0、配置apt-get源

[参考](https://developer.aliyun.com/article/704603)

1、 docker安装

[参考](https://yeasy.gitbook.io/docker_practice/install/ubuntu)

2、使用docker环境

```
# 拉取镜像
docker pull registry.cn-hangzhou.aliyuncs.com/hellofss/kuiperinfer:latest
# 启动容器
sudo docker run --name vimrc-learn --network host --cpus=6 -ti registry.cn-hangzhou.aliyuncs.com/hellofss/kuiperinfer:latest /bin/bash
```

配置免密登录
```
# 本地主机生成公钥
ssh-kengen # 一路回车
cat id_rsa.pub # 复制公钥

# remote 

/usr/sbin/sshd -p port

# 将本地的公钥粘贴进去
vim ~/.ssh/authorized_keys

# 本地
ssh -p port root@localhost

# 退出容器，但不关闭容器
ctrl + d

```


3、编译开发
```
cd code
git clone --recursive https://github.com/zjhellofss/KuiperInfer.git
cd KuiperInfer
git checkout -b 你的新分支 study_version_0.02 (如果想抄本项目的代码，请使用这一步切换到study tag)
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DDEVELOPMENT=OFF ..
make -j16
```

4、启动docker

每次机器重启，运行的容器都要重新启动下。
```
sudo docker ps -a
sudo docker start 83c5307ecb57
sudo docker exec -ti 83c5307ecb57 /bin/bash 
/usr/sbin/sshd -p 8801
```


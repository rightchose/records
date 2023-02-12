##### 1、配置环境变量

`.bashrc`文件存储了每次新建终端需要执行的命令。

`/etc/profile`文件存储了所有用户的配置。

因此，针对个人用户而言，可以去修改`.bashrc`。针对服务器所有者而言可以去修改`/etc/profile`。

`tips`：基本上大家会使用一些第三方的bash例如`zsh`。此时应该修改的是`~/.zshrc`。

关于更深入的介绍可以看这个

https://www.cnblogs.com/youyoui/p/10680329.html

##### 2、screen

参考https://www.cnblogs.com/mchina/archive/2013/01/30/2880680.html

GNU screen是一款由GNU计划开发的用于命令行终端切换的自由软件。

只要screen本身没有终止，其内运行的会话都可以恢复。离线再次登录主机后执行screen -r就可以恢复会话。离开时执行分离命令detach。

**创建一个新的窗口**，最好创建的时候取个名称。然后便进入该窗口，退出窗口可以用`crtl+A+D`。执行的命令不会中断。

```shell
screen -S windows_name
```

随后可以在screen创建的窗口执行一个命令，然后退出。

退出后，通过下面命令查看screen 创建的窗口。

```shell
screen -ls
```

再次进入之前的窗口

```shell
scrren -r windows_id/windows_name
```

**关闭并杀死窗口**`crtl+A+K`或者用下面的命令

```shell
screen -X -S windows_id/windows_name quit
```

如果某个会话死掉了，这时screen -list会显示该会话为dead状态。使用screen -wipe命令清除该会话。

##### 3、screen高级应用

[参考]https://www.cnblogs.com/mchina/archive/2013/01/30/2880680.html

**会话共享**

```shell
screen -x
```

**会话锁定与解锁**

使用快捷键`ctrl+A+S`锁定，使用`ctrl+A+Q`解锁。使用`ctrl+A+X`锁定需要密码解锁。

**发送命令到screen会话**

```
screen -S windows_id/windows_name -X screen cmd
```

##### 4、linux下文件操作

`unzip`解压文件，例如解压一个文件下的文件`unzip \* -d output_dir`

但有些压缩文件无法解压，这时如果是`.gz`压缩包（不带tar），

```
gzip xxx.gz -d [解压位置]
```

解压`.tar.gz`压缩包，需要使用`tar`命令的`-z`和`-f`选项（解压需要`-x`)

```
tar -zxf xxx.tar.gz -C [解压位置]
```

`cp`复制文件, 

`rm`删除文件，删除所有的文件`rm -rf`

`mv`移动文件。

##### 5、gpu使用情况工具

https://github.com/wookayin/gpustat

`gpustat`一个不错的查看gpu使用情况的工具。

##### 6、杀死gpu上的所有程序

```shell
ps -a | cut -c 1-6 | xargs  kill -9
```

##### 7、ln命令

ln命令（link file）也就是软链接的意思。

创建软链接：

```
ln -s [源文件或目录] [目标文件或目录]
```

删除软链接

```
rm -rf 软链接名称
```

修改软链接

```
ln -snf [新的源文件或目录] [目标文件或目录]
```

##### 8、实时监控gpu

```shell
watch -n1 nvidia-smi
```

##### 9、scp

用于可以在两台主机间传输文件，速度要比使用xftp快非常多。

```
scp -P 10110 -r '/home/mr/data/whispers/test_unzip' root@xx.xxx.xxx.xx:/home/mr/data/whispers/test 
```

##### 10、cut

```
du -h train/*/HSI-FalseColor | cut -c 1-2 | awk '{sum+=$1} END {print "Sum = ", sum}'
```

##### 11、awk

参考一下[blog](http://www.ruanyifeng.com/blog/2018/11/awk.html)

> [`awk`](https://en.wikipedia.org/wiki/AWK)是处理文本文件的一个应用程序，几乎所有 Linux 系统都自带这个程序。

##### 12、GPU操作

使用GPU时，遇到了ERR的问题，于是查了查，貌似机器散热出了问题。

```bash
nvidia-smi -r
sudo nvidia-smi -pm 1    # 把GPU的persistent mode（常驻模式）打开，这样才能顺利设置power limit
sudo nvidia-smi -pl 150    # 把功率限制从默认的250W调整到150W，也可以设置其他值啦，自己斟酌
```

##### 13、解决存在于gpu中的死进程

经常会出现程序结束，但是仍会留下进程占用gpu，一般情况下在主机执行命令`nvidia-smi`查看到进程号，然后执行`ps -aux | grep pid`，就可以知道是什么程序，然后`kill -9 pid`即可。

但有时候kill不掉，换一种方法。执行命令` fuser -v /dev/nvidia*`。可以看到每个卡中的进程。此时用这里的`pid`进行kill。

一般来说，不知道我的习惯那一步出错了，经常会在gpu中留下死的进程，占用显存。一般得用后面的方法才能kill掉。

##### 14、linux管理用户

为新成员创建用户，例如创建用户`xkchai`，下面这行命名会创建`xkchai`用户，同时并为其产生相应的目录。`-d`表示指定用户的目录，`-m`表示如果不存在该目录，则自动创建。`-s`表示该用户登录的Shell是`/bin/sh`。

```shell
useradd -d /home/xkchai -m xkchai -s /bin/sh
```

如果将用户加入到root用户组，则使用下面的命令

```shell
useradd -d /home/xkchai -m xkchai -s /bin/sh -g group -G wheel
```

删除账户

```shell
userdel xkchai
```

如果需要连同用户的文件一同删除加上`-r`参数即可。

```shell
userdel -r xkchai
```

给用户设置密码

```
sudo passwd xkchai
```

##### 15、systemctl和service

[参考](https://www.cnblogs.com/shijingjing07/p/9301590.html)

linux的服务管理有两种方式service和systemctl。

service命令去/etc/init.d目录下执行相关程序。例如启动ssh的服务器端程序

```shell
/etc/init.d/ssh start
# 等价
service ssh start
```

对于`systemctl`，我们先介绍`systemd`。`systemd`是Linux系统最新初始化系统(init)，作用是提高系统的启动速度，尽可能启动较少的进程，尽可能多进程并发启动。

而`systemd`对应的进程管理命令是`systemctl`。

`systemctl`兼容了`service`，也就是systemctl也会去/etc/init.d目录下，查看执行相关程序。

```shell
systemctl ssh start
systemctl ssh stop
```

systemctl命令管理systemd的资源Unit。

##### 16、tmux使用

类似于先前的`screen`，[参考](https://www.ruanyifeng.com/blog/2019/10/tmux.html)

**安装**

```sh
# Ubuntu or Debian
apt-get install tmux
# Centos
yum install tmux
```

**创建会话**

```sh
tmux
```

上面的方式会默认创建一个编号为0的会话，如果已有该会话，则编号顺序递延。但是编号缺乏足够信息，不便于管理，我们最好为会话绑定一个名称。

```shell
tmux new -s <session-name>
```

**关闭会话**

```sh
exit
```

也可以`ctrl`+`d`。会话内进程也会终止。

**离开会话**

```shell
tmux detach
```

也可以`ctrl`+`b`+`d`。会话内进程依旧存在。

**重入会话**

```shell
tmux attach -t <session-id>
tumx attach -t <session-name>
```

**会话管理**

```shell
# 查看所有会话
tmux ls
# or
tmux list-session
# 杀死会话
tmux kill-session -t <session-id>
tmux kill-session -t <session-name>
# 会话切换
tmux switch -t <session-id>
tmux switch -t <session-name>
# 会话重命名
tmux rename-session-t <session-id> <new-name>
```

**窗口划分**

```shell
# 划分上下两个窗格
tmux split-window
# 划分左右两个窗格
tmux split-window -h
```

窗口划分后，我们需要移动光标位置。

```
# 光标切换到上方窗格
tmux select-pane -U
# 下方
tmux select-pane -D
# 左边窗格
tmux select-pane -L
# 右边窗格
tmux select-pane -R
```

##### 1、配置环境变量

`.bashrc`文件存储了每次新建终端需要执行的命令。

`/etc/profile`文件存储了所有用户的配置。

因此，针对个人用户而言，可以去修改`.bashrc`。针对服务器所有者而言可以去修改`/etc/profile`。

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

#### 8、实时监控gpu

```shell
watch -n1 nvidia-smi
```






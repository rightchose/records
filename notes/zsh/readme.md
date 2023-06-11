##### zsh安装（oh my zsh）

##### 1、ubuntu
- [参考](https://zhuanlan.zhihu.com/p/514636147)
- [参考](https://segmentfault.com/a/1190000015283092)
```bash
apt-get update
apt-get install zsh
sh -c "$(curl -fsSL https://gitee.com/shmhlsy/oh-my-zsh-install.sh/raw/master/install.sh)"

# 修改默认bash为zsh
chsh -s `which zsh`
# 恢复默认
chsh -s /bin/bash
```

**插件**
```bash
# theme
git clone https://github.com/bhilburn/powerlevel9k.git ~/.oh-my-zsh/custom/themes/powerlevel9k
# zsh-syntax-highlighting，命令行语法高亮
 git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```
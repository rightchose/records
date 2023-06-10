### windows swl2安装
```
wsl --install

dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

wsl --set-default-version 2
```

### ms 应用商店下载ubuntu

1、添加中文支持
```
apt-get update && apt-get install language-pack-zh-hans
```
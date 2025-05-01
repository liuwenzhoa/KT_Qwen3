# 网站服务启动指南

本文档提供了本项目Web服务的搭建与运行步骤。

## 1. 启动Web服务

### 1.1 编译前端代码

在编译前端代码前，请确保已安装[Node.js](https://nodejs.org) 18.3或更高版本。

注意：Ubuntu或Debian GNU/Linux软件仓库中的Node.js版本过低，可能导致编译报错。建议通过Nodesource仓库安装新版Node.js，安装前请先卸载旧版本。

```bash

  # sudo apt-get remove nodejs npm -y && sudo apt-get autoremove -y
  sudo apt-get update -y && sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/nodesource.gpg
  sudo chmod 644 /usr/share/keyrings/nodesource.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_23.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
  sudo apt-get update -y
  sudo apt-get install nodejs -y

```

安装npm后，进入`ktransformers/website`目录：

```bash
cd ktransformers/website
```

然后使用以下命令安装Vue CLI：

```bash
npm install @vue/cli
```

现在可以构建前端项目：

```bash
npm run build
```
最后可以带网站一起安装ktransformers：
```
cd ../../
pip install .
```

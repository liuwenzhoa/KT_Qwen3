# Docker

## 前提条件
* 您的系统上必须安装并运行Docker。
* 创建一个文件夹用于存储大型模型和中间文件（例如 /mnt/models）。

## 镜像
我们的项目提供了Docker镜像，您可以通过以下命令拉取Docker镜像：
```
docker pull approachingai/ktransformers:0.2.1
```
**注意**：在此镜像中，我们在支持AVX512指令集的CPU上编译了ktransformers。如果您的CPU不支持AVX512，建议在容器内的/workspace/ktransformers目录中重新编译并安装ktransformers。

## 本地构建Docker镜像
 - 在[此处](../../Dockerfile)下载Dockerfile

 - 完成后，执行
   ```bash
   docker build -t approachingai/ktransformers:0.2.1 .
   ```

## 使用方法

假设您已安装[nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)，可以在Docker容器中使用GPU。
```
docker run --gpus all -v /path/to/models:/models --name ktransformers -itd approachingai/ktransformers:0.2.1
docker exec -it ktransformers /bin/bash
python -m ktransformers.local_chat --gguf_path /models/path/to/gguf_path --model_path /models/path/to/model_path --cpu_infer 33
```

更多操作选项可以查看[README](../../README.md) 
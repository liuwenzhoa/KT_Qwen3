# Makefile使用说明
## 目标
### flake_find:
```bash
make flake_find
```
查找`./ktransformers`目录下所有的Python文件，列出不符合PEP8标准的错误、警告和严重问题（及其代码）。目前我们已将这些问题列表添加到`.flake8`文件的`extend-ignore`部分，使flake8暂时忽略它们（我们计划在未来改进这些问题）。

### format:
```bash
make format
```
我们使用black格式化`./ktransformers`目录下的所有Python文件。它遵循PEP8标准，但通过在`pyproject.toml`文件中添加以下内容，我们将行长度修改为120：
```toml
[tool.black]
line-length = 120
preview = true
unstable = true
```

### dev_install:
```bash
make dev_install
```
以开发模式安装包。这意味着包是以可编辑模式安装的。因此，如果您修改了代码，无需重新安装包。我们建议开发者使用此方法安装包。 
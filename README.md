# DeepZLab
这是一个用于深度学习学习的库，旨在提供灵活的模块化支持。核心功能包括多个模型的灵活调用，支持各种深度学习框架，能够方便地在不同模型之间切换。同时，库还实现了自定义的数据集 dataloader，简化了数据加载和预处理过程。为了进一步提高效率，已上传多个数据集，便于快速开始实验。此外，库提供了定制化的 trainer 和 main 脚本，允许用户根据自己的需求调整训练流程和模型参数，适应不同的任务和需求。



# English Description:
This is a deep learning learning library designed to provide flexible, modular support. Key features include the flexible invocation of multiple models, making it easy to switch between different models with support for various deep learning frameworks. The library also implements a customized dataset dataloader that simplifies data loading and preprocessing. To facilitate quick experimentation, several datasets have been uploaded. Additionally, the library offers a customizable trainer and main script, allowing users to adjust training workflows and model parameters according to their specific needs, making it adaptable to various tasks and requirements.



# 实现需求

1. main.py不需要动，只需要修改配置文件，就可以实现更换models和task
2. 配置参数，可以是配置文件，也可以是命令行的，命令行优先级更高
3. 只需要model_name和输入输出设置就可以调用各种模型





```
args, train_dl, val_list, model, train_one_epoch, val_one_epoch
```





## Usage

使用SpiderNet，在Chesapeake数据集

1. 修改model_name即可



## Changelog

### 2025-09-29

- 新增 `clip_raster_by_polygon` 方法
  - 仅支持按 **单个矢量多边形** 裁剪栅格。
- 原有的 `clip_raster_by_shp` 方法更名为 **`clip_raster_by_rectangle`**，以更准确地反映其“按外包矩形裁剪”的功能。

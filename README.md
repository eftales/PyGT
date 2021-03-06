# PyGT
PyTorch Geometric Temporal 急速入门

## 官方网站
PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

PyTorch Geometric Temporal: https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/installation.html#

CUDA: https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal

## 安装
PyTorch=1.9.0,CUDA=11.1。实测可以通过用 `cpu` 取代下面安装链接中的 `cu111` 来安装 cpu 版本的库。 

torch-geometric-temporal 一定要加参数 `--no-deps`，否则会强制安装 torch-geometric==1.X。

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-geometric==2.0.3
pip install torch-geometric-temporal==0.42 --no-deps
```

```
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
pip install torch-geometric==2.0.3
pip install torch-geometric-temporal==0.42 --no-deps
```

## 卸载

```
pip uninstall torch torchvision torchaudio torchtext torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric torch-geometric-temporal -y
```

## Tips
文件夹 `1-gru/` 演示了如何使用 GRU 进行预测。

文件夹 `2-pyg/` 演示了 pyg 的基本使用方法。

文件夹 `3-pygt/` 演示了 pygt 的基本使用方法。

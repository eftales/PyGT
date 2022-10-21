# PyGT
PyTorch Geometric Temporal 急速入门

## 官方网站
PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

PyTorch Geometric Temporal: https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/installation.html#

CUDA: https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal

## 安装
Python=3.8, PyTorch=1.10.2, CUDA=11.3。实测可以通过用 `cpu` 取代下面安装链接中的 `cu113` 来安装 cpu 版本的库。 

安装 torch-geometric-temporal 时若不加 `--no-deps` 参数会导致强制安装当前 torch-geometric-temporal 指定的 torch-geometric 版本，但是有的时候这两个版本配合的并不好。经过测试 0.54.0 版本可以完美适配。

```
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.2+cu113.html

# pip install torch-geometric -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-geometric-temporal==0.54.0 # 会直接安装配套的 torch-geometric 版本
```

```
pip install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.2+cpu.html
# pip install torch-geometric
pip install torch-geometric-temporal==0.54.0 # 会直接安装配套的 torch-geometric 版本
```

## 卸载

```
pip uninstall torch torchvision torchaudio torchtext torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric torch-geometric-temporal -y
```

## Tips
文件夹 `0-rnn/` 演示了如何从零实现一个 RNN。

文件夹 `1-gru/` 演示了如何使用 GRU 进行预测。

文件夹 `2-pyg/` 演示了 pyg 的基本使用方法。

文件夹 `3-pygt/` 演示了 pygt 的基本使用方法。

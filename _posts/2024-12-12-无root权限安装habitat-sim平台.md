---
layout:     post
title:      tf相关依赖项的匹配选择
subtitle:   解决方案日志
date:       2024-12-12
author:     lihan
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - blog
    - 配环境
---

没有root权限并且服务器一会儿抽风有显示器转发一会儿又没有，看着一堆教程中动不动出现的sudo只想砸键盘。。struggle了好几天最后终于解决了，所以来这里update一下经验。

首先Python3.9是不行的（）让我们创建一个新的conda环境：

# 创建新的conda虚拟环境
conda create -n habitat_env python=3.7 cmake=3.14.0 -y

# 激活虚拟环境
conda activate habitat_env 

# 安装PyTorch及相关库
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch -y

接着获取Habitat-Sim的源代码，并切换到稳定的发布版本：

# 克隆Habitat-Sim仓库
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim

# 切换到指定的标签
git checkout tags/v0.1.7

# 安装habitat-sim依赖
pip install -r requirements.txt

接着就是非常tricky的。。无root手动安装CUDA、nvidia_driver、OpenGL和EGL

# 运行命令以检查驱动程序的版本
nvidia-smi

# 直接从 NVIDIA 下载驱动程序安装二进制文件。您可以通过在谷歌上搜索 nvidia driver <version-id> 来做到这一点。该文件应为 
NVIDIA-Linux-x86_64-<version-id>.run

# 在不运行的情况下提取二进制文件。这将创建一个名为 NVIDIA-Linux-x86_64-<version-id>
sh NVIDIA-Linux-x86_64-<version-id>.run --extract-only  

# 创建一些 SimLink
cd NVIDIA-Linux-x86_64-<version-id>
ln -s ./libEGL.so.<version-id>  libEGL.so.1 # note: there might be a libEGL_nvidia*, do not symlink it as libEGL.so.1 instead!
ln -s ./libGL.so.<version> libGL.so.1      

# 将此文件夹添加到 LD 搜索路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/NVIDIA-Linux-x86_64-<version-id>

# 检查链接器
cd
ldd $(python -c "import habitat_sim; print(habitat_sim._ext.habitat_sim_bindings.__file__)")

# 请注意， libEGL.so.1 和 libOpenGL.so.0 libGLdispatch.so.0 应该都指向 /path/to/NVIDIA-Linux-x86_64-<version-id> 中的那些。

然后安装OpenGL和EGL，使用Conda从conda-forge渠道安装OpenGL库，同时从KhronosGroup的EGL-Registry克隆EGL头文件：

# 使用conda安装OpenGL库
conda install -c conda-forge mesalib

# 克隆EGL头文件仓库
git clone https://github.com/KhronosGroup/EGL-Registry.git

接着配置环境变量：

# 设置环境变量以链接到CUDA和EGL头文件
export CUDA_HOME=$CUDA_HOME:/path/to/cuda/
export PATH=$PATH:/path/to/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cuda/lib64
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/path/to/EGL-Registry-main/api
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/path/to/EGL-Registry-main/api
export CMAKE_INCLUDE_PATH=$CMAKE_INCLUDE_PATH:/path/to/EGL-Registry-main/api
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/NVIDIA-Linux-x86_64-545.29.06/

最后，编译并安装Habitat-Sim和Habitat-Lab：

# 在Habitat-Sim目录下编译和安装
cd habitat-sim
python setup.py install --headless --with-cuda --cmake-args="-DEGL_LIBRARY=/path/to/NVIDIA-Linux-x86_64-545.29.06/libEGL.so" # 一定要加上编译参数关于EGL的/path/to/NVIDIA-Linux-x86_64-545.29.06/编译位置指定

# 在Habitat-Lab目录下进行开发安装
cd ../habitat-lab
python setup.py develop --all

截至2024.12.5此方法一直有效，但基于某些懂的都懂的原因极有可能很快失效，如有失效请联系本人前来update：↓
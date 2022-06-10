## DeepMD配置

1、配置相关环境变量，并且创建⼀个虚拟python环境（位于ENV_PREFIX⽂件下）（这是在72的机⼦上的）：  

```shell
export ENV_PREFIX=/$PWD/hov
export CUDA_HOME=/usr/local/cuda-10.1/
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_OPERATIONS=NCCL
conda env create --prefix $ENV_PREFIX --file env_only_conda.yml --force
```

其中env_only_conda.yml为  

```
name: null
channels:
- pytorch
- conda-forge
- defaults
dependencies:
- cmake=3.17
- cudatoolkit=10.1
- cudnn=7.6
- cupti=10.1
- cxx-compiler=1.1
- jupyterlab=2.2
- mpi4py=3.0 # installs cuda-aware openmpi
- nccl=2.5
- nodejs=14
- nvcc_linux-64=10.1
- pip=20.1
- python=3.7
- pytorch=1.5
- tensorboard=2.1
- torchvision=0.6
```

​	2、安装tensorflow-gpu，mxnet，horovod
下载地址：
https://pypi.org/project/tensorflow-gpu/
我是通过windows下来之后，通过finalshell的中linux的rz命令进⾏⽂件传输，将./tensorflow_gpu-2.2.0-cp37-cp37mmanylinux2010_x86_64.whl传输到⾃⼰的⽬录下，然后通过下⾯命令安装：  

3、配置deepmd环境：  

```shell
git clone --recursive https://github.com/deepmodeling/deepmd-kit.git deepmd-kit
cd /home/asc22g0/CK/deemd-kit
python -m pip install .
```

4、测试结果：  

```shell
horovodrun --check-build #（查看horovodrun框架的build）

dp -h #(⾮deepmd-kit⽬录下，看是否能正常运⾏)
```

![image-20220128190743561](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220128190743561.png)

![image-20220128190810453](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220128190810453.png)





启动需要使用这个命令

```shell
export CUDA_HOME=/usr/local/cuda-10.1/
source activate /home/asc22g0/GC/hov
```








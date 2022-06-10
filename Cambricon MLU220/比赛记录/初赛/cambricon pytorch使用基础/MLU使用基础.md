# MLU使用基础

Cambricon PyTorch主要使用场景为模型推理和模型训练。

针对上述两种应用场景的差异，寒武纪提供了CNML和CNNL两套算子库：

- CNML通过将模型转换成计算图，利用融合操作等功能对计算图进行离线的编译优化，生成可端到端部署的推理引擎。
- CNNL为通用算子库，提供手工优化的基本算子或简单的融合算子，保证每个算子的单次运行延时尽可能低。

![../_images/CNML_and_CNNL.PNG](https://www.cambricon.com/docs/pytorch/_images/CNML_and_CNNL.PNG)

## 基础使用

### 导入torch_mlu相关包

```
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import torchvision.models as models
```

### 设置MLU基本信息

```
ct.set_core_number(args.core_number)                 # 设置MLU core number
ct.set_core_version(args.mcore)                      # 设置MLU core version
ct.set_input_format(args.input_format)               # 设置输入图片的通道顺序，以决定首层卷积对三通道输入的补齐通道顺序。默认是RGBA顺序。
```

以上代码为推理的基本参数设置，例如推理的核数、推理核版本、输入数据模式。该部分只适用于推理模式。

### 加载模型

Cambricon PyTorch模型在torchvision库中均已提供。使用时通过torchvision导入模型，具体命令如下：

```
torch.set_grad_enabled(False) #注意：在运行MLU推理融合模式时，这个条件是必须要设置的。
net = getattr(models.quantization, 'net_name')(pretrained=True, quantize=True)
```

其中， `models.quantization` 为torchvision导入后的模型；`getattr(models.quantization,net_name)` 将返回指定的网络对象；`pretrained=True` 将在初始化对象过程中同时加载权重；`quantize=True` 表示加载量化的权重。

# 推理快速入门

PyTorch使用由多层互连计算单元组成的神经网络（模型）进行模型推理。PyTorch提供了设计好的模块和类便于快速创建网络模型，例如 `torch.nn` 类。本节使用 `torch.nn` 类来定义MNIST网络模型并介绍模型推理的具体方法。 完整代码可以参见本节末尾的 [推理完整代码](https://www.cambricon.com/docs/pytorch/pytorch_5_quickguide/Pytorch_quickguide.html#id12)。

## 编译和安装

使用模型训练前，要先编译与安装Cambricon PyTorch，并进入PyTorch的虚拟环境。

## 导入必要模块

执行以下命令导入模块：

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct
```

## 定义和初始化模型

MINIST模型使用PyTorch的卷积运算从输入图像中提取某些特征（例如边缘检测、清晰度、模糊度等）进行图像识别。

定义模型的 `Net` 类，需执行以下步骤：

1. 编写引用nn.Module的__init__函数。

   在__init__中定义连接在网络中的所有层。这里将遵循标准MINST算法使用卷积创建输入图像通道为1，输出10个标签目标的网络模型，这些标签代表数字0到9。

   ```
   class Net(nn.Module):
     def __init__(self):
       super(Net, self).__init__()
   
       # First 2D convolutional layer, taking in 1 input channel (image),
       # outputting 32 convolutional features, with a square kernel size of 3
       self.conv1 = nn.Conv2d(1, 32, 3, 1)
       # Second 2D convolutional layer, taking in the 32 input layers,
       # outputting 64 convolutional features, with a square kernel size of 3
       self.conv2 = nn.Conv2d(32, 64, 3, 1)
   
       # Designed to ensure that adjacent pixels are either all 0s or all active
       # with an input probability
       self.dropout1 = nn.Dropout2d(0.25)
       self.dropout2 = nn.Dropout2d(0.5)
   
       # First fully connected layer
       self.fc1 = nn.Linear(9216, 128)
       # Second fully connected layer that outputs our 10 labels
       self.fc2 = nn.Linear(128, 10)
   
   my_nn = Net()
   print(my_nn)
   ```

2. 编写forward函数。该函数会将输入数据传递到网络的计算图，完成模型的前向推理过程。

   ```
   class Net(nn.Module):
       def __init__(self):
         super(Net, self).__init__()
         self.conv1 = nn.Conv2d(1, 32, 3, 1)
         self.conv2 = nn.Conv2d(32, 64, 3, 1)
         self.dropout1 = nn.Dropout2d(0.25)
         self.dropout2 = nn.Dropout2d(0.5)
         self.fc1 = nn.Linear(9216, 128)
         self.fc2 = nn.Linear(128, 10)
   
       # x represents our data
       def forward(self, x):
         # Pass data through conv1
         x = self.conv1(x)
         # Use the rectified-linear activation function over x
         x = F.relu(x)
   
         x = self.conv2(x)
         x = F.relu(x)
   
         # Run max pooling over x
         x = F.max_pool2d(x, 2)
         # Pass data through dropout1
         x = self.dropout1(x)
         # Flatten x with start_dim=1
         x = torch.flatten(x, 1)
         # Pass data through fc1
         x = self.fc1(x)
         x = F.relu(x)
         x = self.dropout2(x)
         x = self.fc2(x)
   
         # Apply softmax to x
         output = F.log_softmax(x, dim=1)
         return output
   ```

3. 将上述模型定义保存为minist.py。

## 保存权重

保存随机初始化的权重，此处命名为 `test_weights.pth`。

```
output=net(input_data)
torch.save(net.state_dict(), 'test_weights.pth') # 其中：net.state_dict()获取各层参数，path是文件存放路径(通常保存文件格式为.pt或.pth) 功能：保存训练完的网络的各层参数（即weights和bias)
```

## 模型量化

利用 [模型量化工具](https://www.cambricon.com/docs/pytorch/pytorch_11_tools/Pytorch_tools.html#id1) 对模型的权重进行量化，并保存量化后的权重。此处命名为test_quantization.pth。

```
mean = [0]
std = [1/255]
net.load_state_dict(torch.load('test_weights.pth', map_location='cpu'), False)
net_quantization = mlu_quantize.quantize_dynamic_mlu(net, {'mean':mean, 'std':std, 'firstconv':True}, dtype='int8', gen_quant=True)
torch.save(net_quantization.state_dict(), 'test_quantization.pth')
```

`mean` 和 `std` 是对输入图片进行预处理的数组值，表示对输入图片的每个通道处理的均值和标准值。MNIST网络使用的数据是一通道的，因此只需传入一个通道的值。如果是3个通道的输入，可设置为 `mean = [0,0,0]`, `std = [1/255, 1/255, 1/255]`。

将保存的test_weights.pth加载，将其中的参数加载进net中，然后使用 `quantize_dynamic_mlu` 接口对net进行模型量化，并保存量化后的模型。

## 模型推理

模型推理首先要对图像进行前处理。该过程会调整输入图像的大小，并根据数据集的mean和std对图片做均值处理。前处理流程如下所示：

```
# step 1
net = mlu_quantize.quantize_dynamic_mlu(net)
# step 2
net.load_state_dict(torch.load('test_quantization.pth'))
input_data=torch.randn((1,1,28,28))
# step 3
net_mlu = net.to(ct.mlu_device())
input_mlu = input_data.to(ct.mlu_device())
# step 4
output=net_mlu(input_mlu)
print(output.cpu())
```

运行 `python mnist.py` 进行模型推理。

显示如下结果：

```
CNML: 7.9.2 1a1e33b
CNRT: 4.8.2 6d9ad7c
tensor([[-2.3040, -2.3272, -2.3617, -2.3555, -2.2696, -2.2130, -2.2238, -2.1935,
         -2.3787, -2.4267]])
```

# 训练快速入门

本节以MNIST为例说明具体的训练流程。

## 编译和安装

使用模型训练前，要先编译与安装Cambricon PyTorch，并进入PyTorch的虚拟环境。

## 导入必要模块

执行以下命令导入模块：

```
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from torch import optim
from torchvision import transforms
import torch.nn.functional as F

import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct
ct.set_cnml_enabled(False)
```

其中，`mlu_quantize` 模块对nn.Conv2d与nn.Linear算子进行自适应量化，可以在保证精度的前提下加快网络训练速度。 `ct` 模块用于管理MLU设备，将数据、模型在CPU与MLU间进行拷贝，并对MLU设备、MLU Queue等模块进行管理。 `ct.set_cnml_enabled(False)` 设置为运行训练模式，该设置必须添加。

## 定义和初始化模型

MINIST示例模型使用PyTorch构建模块，从输入图像中提取某些特征（如边缘检测、清晰度、模糊度）进行图像识别。

定义模型的 `Net` 类，需执行以下步骤：

1. 编写一个继承nn.Module的__init__函数。在__init__函数中定义了连接在网络中的基本层。

2. ```
   class Net(nn.Module):
       def __init__(self):
         super(Net, self).__init__()
         self.conv1 = nn.Conv2d(1, 32, 3, 1)
         self.conv2 = nn.Conv2d(32, 64, 3, 1)
         self.dropout1 = nn.Dropout2d(0.25)
         self.dropout2 = nn.Dropout2d(0.5)
         self.fc1 = nn.Linear(9216, 128)
         self.fc2 = nn.Linear(128, 10)
   ```

3. 编写forward函数。该函数会将输入数据传递到网络的计算图中并完成模型前向推理。

   ```
   def forward(self, x):
     x = self.conv1(x)
     x = F.relu(x)
     x = self.conv2(x)
     x = F.relu(x)
     x = F.max_pool2d(x, 2)
     x = self.dropout1(x)
     x = torch.flatten(x, 1)
     x = self.fc1(x)
     x = F.relu(x)
     x = self.dropout2(x)
     x = self.fc2(x)
     output = F.log_softmax(x, dim=1)
     return output
   ```

3.将上述模型定义保存为minist.py（需要与推理的mnist.py相区别）。

## 准备数据集以及数据预处理模块

```
data_tf = transforms.Compose(
              [transforms.ToTensor(),
              transforms.Normalize([0.1307],[0.3081])])
train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=64, shuffle=False)
```

此处使用PyTorch中预处理操作，`data_tf` 将输入数据转化为Tensor、并进行Normalize预处理，然后将数据按batch_size大小加载。

为了保证训练效果，在训练集上用 `shuffle=True` 将训练图片打乱，在测试集上则不打乱图片。

训练集与测试集均直接从网上下载，并自动进行预处理，因此用户必须保证机器联网。

如果无法访问互联网资源，可将相应的数据提前拷贝至 `mnist.py` 所在目录，按 `./data/raw/****` 方式存放，并将上面的train_set/test_set设置如下：

```
train_set = mnist.MNIST('./data', train=True)
test_set = mnist.MNIST('./data', train=False)
```

## 模型量化

生成Net对象net，并使用mlu_quantize模块中adaptive_quantize接口处理net。该接口需传入 `model` 与 `steps_per_epoch` 参数，默认进行8位量化。如果需要进行更高位数的量化，可以在该接口中传入 `bitwidth`，并将其设置为16或者31。

```
net_orig = Net()
net_quantize = mlu_quantize.adaptive_quantize(model=net_orig, steps_per_epoch=len(train_data), bitwidth=8)
net = net_quantize.to(ct.mlu_device())
optimizer_orig = optim.SGD(net_quantize.parameters(), 1e-1)
optimizer = ct.to(optimizer_orig, torch.device("mlu"))
```

以上代码构建了前向网络模型、损失函数与优化器，并将其拷贝上MLU。

## 定义训练与验证模块

将以下代码加入到 `mnist.py` 以构建训练模块和验证模块。

```
nums_epoch = 10  # 此处设置的10个训练epoch
save_model = True  # 保存模型开关

def train(model, train_data, optimizer, epoch):
  model = model.train()
  for batch_idx, (img, label) in enumerate(train_data):
      img = img.to(ct.mlu_device())
      label = label.to(ct.mlu_device())
      optimizer.zero_grad()
      out = model(img)
      loss = F.nll_loss(out, label)
      loss.backward()
      optimizer.step()
      if batch_idx % 100 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(img), len(train_data.dataset),
              100. * batch_idx / len(train_data), loss.item()))

def validate(val_loader, model):
  test_loss = 0
  correct = 0
  model.eval()
  with torch.no_grad():
      for images, target in val_loader:
          images = images.to(ct.mlu_device())
          target = target.to(ct.mlu_device())
          output = model(images)
          test_loss += F.nll_loss(output, target, reduction='sum').item()
          pred = output.argmax(dim=1, keepdim=True)
          pred = pred.cpu()
          target = target.cpu()
          correct += pred.eq(target.view_as(pred)).sum().item()
  test_loss /= len(val_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(val_loader.dataset),
      100. * correct / len(val_loader.dataset)))
```

`nums_epoch` 为训练的epoch数，用来控制训练的轮数。此处只作为演示用途，用户根据实际情况设置该值。

`save_model` 用来设置是否保存模型，默认值为True。

## 模型训练执行 `python mnist.py`，可得到如下结果（此处只显示第0个epoch的训练和验证数据）：

```
Train Epoch: 0 [0/60000      (0%)]    Loss: 2.291011
Train Epoch: 0 [6400/60000  (11%)]    Loss: 0.513938
Train Epoch: 0 [12800/60000 (21%)]    Loss: 0.485264
Train Epoch: 0 [19200/60000 (32%)]    Loss: 0.259880
Train Epoch: 0 [25600/60000 (43%)]    Loss: 0.246993
Train Epoch: 0 [32000/60000 (53%)]    Loss: 0.273036
Train Epoch: 0 [38400/60000 (64%)]    Loss: 0.095428
Train Epoch: 0 [44800/60000 (75%)]    Loss: 0.102112
Train Epoch: 0 [51200/60000 (85%)]    Loss: 0.161822
Train Epoch: 0 [57600/60000 (96%)]    Loss: 0.354688

Test set: Average loss: 0.0650, Accuracy: 9812/10000 (98%)
```

## 保存模型

在验证接口后，可以将训练好的模型保存到指定位置，此处命名为 `model.pth`。

```
if epoch == nums_epoch - 1:
    checkpoint = {"state_dict":net.state_dict(), "optimizer":optimizer.state_dict(), "epoch": epoch}
    torch.save(checkpoint, 'model.pth')
```

当 `epoch` 达到 `nums_epoch-1` 时，会在本地保存模型，名为model.pth，该模型是直接保存的MLU模型。


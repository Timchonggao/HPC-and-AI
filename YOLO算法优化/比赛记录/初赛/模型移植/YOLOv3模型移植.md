
## Pytorch YOLOv3 移植步骤

> 此demo以yolov3为例，介绍将模型移植到MLU上的步骤，详情参考用户手册和教学视频。

### 1.配置环境

声明环境变量及进入虚拟环境（该操作每次进入docker都需要进行）

```shell
   cd /workspace/volume/private/sdk/cambricon_pytorch
   source env_pytorch.sh
```

source /workspace/volume/private/sdk/cambricon_pytorch/env_pytorch.sh

### 2. 准备模型

demo位置：/workspace/volume/private/00_Yolov3_example

准备yolov3的模型：需要将原darknet框架生成的yolov3权重yolov3.weight转换为pytorch可读取的pth格式。
本demo已经准备好yolov3.pth模型，并保存在/workspace/volume/private/00_Yolov3_example/model/online/路径下。
【注】 模型必须以pth的格式（pth中只有权重，不保存模型结构）保存。


### 3. 在线推理

在线推理的示例代码在/workspace/volume/private/00_Yolov3_example/online/yolov3目录下。

3.1 **模型量化**

   ```shell
   cd /workspace/volume/private/00_Yolov3_example/online/yolov3
   bash quantize.sh #进行量化
   ```
   模型在MLU上运行需要先进行量化，cambricon-pytorch提供了相应的接口来量化模型的权重。

   ```python
   models.object_detection.yolov3(weight_path,pretrained=True, img_size, conf_thres, nms_thres)
   mean = [0.0, 0.0, 0.0]
   std  = [1.0, 1.0, 1.0]
   # 调用量化接口进行量化
   qconfig = {'use_avg':False, 'data_scale':1.0, 'mean': mean, 'std': std, 'per_channel': per_channel, 'firstconv':True}
   quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype=dtype, gen_quant=True)   
   ......
   # 保存量化模型
   checkpoint = quantized_model.state_dict()
   torch.save(checkpoint,'{}/yolov3_int8.pth'.format(opt.quantized_model_path))
   ```

   量化过程需要在CPU上完成，此demo量化后的模型yolov3_int8.pth保存在/workspace/volume/private/00_Yolov3_example/model/online/目录下。

3.2 **在线推理**

   完成量化后即可加载量化模型进行推理。
   【注】 可通过python test.py -h 查看参数定义

   ```shell
   bash run_online_accuracy.sh     #在线推理精度测试
   bash run_online_performance.sh  #在线推理性能测试
   ```
   模型和数据需要通过`.to(torch_mlu.core.mlu_model.mlu_device())`来指定设备为MLU。
   ```python
   model = models.quantization.object_detection.yolov3(quantized_weight_path,pretrained=True,quantize=True,img_size,conf_thres,nms_thres)
   model.to(torch_mlu.core.mlu_model.mlu_device())
   ```
   ```python
   ......
   imgs = Variable(imgs.type(torch.HalfTensor)) if opt.half_input and opt.mlu else Variable(imgs.type(Tensor))
   imgs = imgs.to(torch_mlu.core.mlu_model.mlu_device())
   outputs = model(imgs)
   ......
   ```

   利用 JIT 模块可以实现融合模式。融合模式会对整个网络构建一个静态图，并对静态图进行优化，有效提高性能。

   ```python
   # trace network
   example = torch.randn(opt.batch_size, 3, img_size, img_size).float()
   trace_input = torch.randn(1, 3, img_size, img_size).float()
   if opt.half_input:
       example = example.type(torch.HalfTensor)
       trace_input = trace_input.type(torch.HalfTensor)
   model = torch.jit.trace(model, trace_input.to(torch_mlu.core.mlu_model.mlu_device()), check_trace = False)
   ```
### 4.离线推理
离线推理的代码在/workspace/volume/private/00_Yolov3_example/offline/yolov3/src目录下。
4.1 **编译代码**
   ```shell
   cd /workspace/volume/private/00_Yolov3_example/offline/
   mkdir build
   cd build
   cmake ..
   make
   cd ..
   ```
   完成编译后会在/workspace/volume/private/00_Yolov3_example/offline/build/yolov3/src目录下生成yolov3_offline_multicore可执行文件。
4.2 **生成离线模型及推理**

   ```shell
   cd /workspace/volume/private/00_Yolov3_example/offline/yolov3/

   #生成batch_size=1,core_number=1的离线模型yolov3.cambricon并保存在/workspace/volume/private/00_Yolov3_example/model/offline/目录下。
   bash run_get_accuracy_offlinemodel.sh

   #进行离线推理精度测试
   bash run_offline_accuracy.sh

   #生成batch_size=16,core_number=16的离线模型yolov3.cambricon保存在../../model/offline/目录下。
   bash run_get_performance_offlinemodel.sh

   #进行离线推理性能测试
   bash run_offline_performance.sh
   ```

   通过调用`torch_mlu.core.mlu_model.save_as_cambricon(model_name)`接口，在进行jit.trace时会自动生成离线模型。生成的离线模型一般是以model_name.cambricon命名的离线模型文件，其中包含一个名为 model_name 的模型。

   ```python
   torch_mlu.core.mlu_model.save_as_cambricon('yolov3')
   ```
离线模型运行代码的编写可以参考[cnrt文档]([离线模型示例程序 — 寒武纪运行时库用户手册 4.10.0 文档 (cambricon.com)](https://www.cambricon.com/docs/cnrt/user_guide_html/example/offline_mode.html))中的示例。



## 模型量化工具说明

https://www.cambricon.com/docs/pytorch/pytorch_11_tools/Pytorch_tools.html#id1



## 在线逐层和融合推理

在线推理指使用原生PyTorch提供的Python API直接运行网络。在线推理包括逐层模式和融合模式两种。

逐层模式使用Python API逐个调用每个算子时，每个MLU算子都在底层的C++接口中经过创建句柄、编译指令、拷贝输入、前向计算、拷贝输出等过程。逐层模式便于观察数据的流动和变化，但是效率较低。

融合模式将所有算子作为一个fusion算子，只对fusion算子执行编译指令过程，减少了小算子之间的 数据拷贝（不仅是主从设备间，还包括RAM和DDR之间的拷贝），极大地提高了效率。使用JIT模式只需对整个网络进行 一次编译，避免了多次编译产生的开销。

```python
from torch_mlu.core.quantized.fuse_modules import fuse_modules

class ConvBnReLU3dModel(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
        stride, padding, dilation, groups):
    super(ConvBnReLU3dModel, self).__init__()
    self.conv = nn.Conv3d(in_channel, out_channel, kernel_size,
                         stride=stride, padding=padding,
                         dilation=dilation, groups=groups)
    self.bn = nn.BatchNorm3d(out_channel)
    self.relu = nn.ReLU()

    def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

model = ConvBnReLU3dModel(3, 16, 3, 1, 1, 1, 1)
# 完成conv、bn、relu三个算子的融合，将三个算子替换为一个大算子
fuse_modules(model, ['conv','bn','relu'], inplace=True)

# 对融合后模型量化
...
```

torch.ops.torch_mlu.yolov5_detection_output

## CPU 模式



## 模型量化

```shell
python detect_true.py --quantization true --mlu false
```

## 模型在线模式逐层推理

```shell
python detect_true.py --quantization false --mlu true 
```

## 在线模式融合推理

```shell
python detect_true.py --quantization false --mlu true --jit true
```

## 生成离线融合模型

```shell
python detect_true.py --quantization false --mlu true --jit true --save_offline_model true
```

## 在线精度测试

```shell
python detect_true.py --quantization false --mlu true --jit true --save_offline_model true --compute_map true
```

python test.py --mlu false --jit false --batch_size 1 --core_number 1 --image_number 2 --half_input 1 --quantized_mode 1 --quantization false --input_channel_order 0 --compute_map true --run_mode false



## 调整图片大小

```python
img = Image.open("./data/images/bus.jpg")
img = img.resize(640,640)
img = np.array(img)
cv2.imwrite("mlu_out_bus.jpg", img)
```

## 指定宽高

```python
import cv2

img = cv2.imread("./data/images/bus.jpg")

 
width = 640
height = 640
dim = (width, height)
 
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite("mlu_out_bus.jpg", resized)
 

```

![image-20211003223715091](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20211003223715091.png)	



## CPU模式输出：

```
CPU pred:  tensor([[[2.03060e+00, 2.58950e+00, 5.68426e+00,  ..., 6.13818e-04, 6.63433e-04, 5.86481e-04],
         [1.12860e+01, 2.17579e+00, 2.07703e+01,  ..., 5.91946e-04, 6.69680e-04, 5.57139e-04],
         [2.13187e+01, 8.55774e+00, 2.38936e+01,  ..., 4.60353e-04, 4.43738e-04, 1.23018e-03],
         ...,
         [5.61544e+02, 6.02964e+02, 1.38623e+02,  ..., 1.50212e-03, 1.10932e-03, 1.05219e-03],
         [5.83089e+02, 6.05413e+02, 1.61815e+02,  ..., 1.00942e-03, 1.94915e-03, 3.16580e-03],
         [6.20349e+02, 6.14521e+02, 1.87576e+02,  ..., 5.79984e-04, 3.06690e-03, 5.72206e-03]]])
```

采用NMS的输出：

```
pred:  [tensor([[-3.82996e-02,  3.49270e+02,  4.00604e+02,  5.57070e+02,  8.86674e-01,  5.90000e+01],
        [ 3.41624e+02,  2.89382e+02,  4.29599e+02,  4.29386e+02,  8.71249e-01,  5.80000e+01],
        [ 1.84692e+02,  2.10511e+02,  2.43130e+02,  3.06823e+02,  8.65403e-01,  5.80000e+01],
        [ 2.44686e+02,  3.09430e+02,  3.49442e+02,  3.96126e+02,  8.14127e-01,  5.60000e+01],
        [ 2.03786e+02,  2.82465e+02,  2.31900e+02,  3.06749e+02,  6.13899e-01,  7.50000e+01],
        [ 9.71712e+01,  2.69701e+02,  1.14281e+02,  3.08640e+02,  5.63304e-01,  3.90000e+01],
        [ 5.23086e+02,  2.23578e+02,  5.49140e+02,  2.58848e+02,  4.08943e-01,  7.30000e+01],
        [ 5.07562e+02,  3.25459e+02,  5.38962e+02,  3.65992e+02,  3.70526e-01,  7.30000e+01],
        [ 4.82933e+02,  3.32511e+02,  5.01575e+02,  3.65669e+02,  3.11960e-01,  7.30000e+01],
        [ 4.88454e+02,  3.28686e+02,  5.10809e+02,  3.66023e+02,  3.09061e-01,  7.30000e+01],
        [ 5.26068e+02,  3.26506e+02,  5.51967e+02,  3.67010e+02,  2.82414e-01,  7.30000e+01],
        [ 4.75004e+02,  3.32590e+02,  4.92899e+02,  3.65072e+02,  2.67434e-01,  7.30000e+01],
        [ 4.69098e+02,  3.31544e+02,  4.83421e+02,  3.64727e+02,  2.64623e-01,  7.30000e+01],
        [ 5.10570e+02,  2.71053e+02,  5.19643e+02,  3.05280e+02,  2.60286e-01,  7.30000e+01]])]
```



## MLU 输出

     MLU pred:  tensor([[[[18.00000]],     [[-0.25522]],
     [[-0.12722]],
    
     ...,
    
     [[ 0.41615]],
    
     [[-0.27357]],
    
     [[-0.24154]]]])



## Pytorch YOLOv3 移植步骤

> 此demo以yolov3为例，介绍将模型移植到MLU上的步骤，详情参考用户手册和教学视频。

### 1.配置环境

声明环境变量及进入虚拟环境（该操作每次进入docker都需要进行）

```shell
   cd /workspace/volume/private/sdk/cambricon_pytorch
   source env_pytorch.sh

```

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


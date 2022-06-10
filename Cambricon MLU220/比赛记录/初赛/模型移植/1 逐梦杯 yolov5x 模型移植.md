### 移植YOLOv5x模型

首先将赛题代码复制到私有目录下，防止代码修改后丢失，进入pytorch虚拟环境下进行模型开发。



#### CPU模式

所有赛题的CPU模式的精度已经经过验证，可用于移植模型优化后的参照。

修改模型路径，和图片参数后注册使用的激活函数，运行cpu模式的代码并增加打印识别框的功能。

运行结果：

```
Model Summary: 476 layers, 87730285 parameters, 0 gradients
image 1/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5/data/images/bus.jpg:
pred:  [tensor([[476.48029, 231.27528, 560.08221, 521.95844,   0.93258,   0.00000],
        [108.84730, 234.92026, 226.93616, 535.01819,   0.92405,   0.00000],
        [212.26106, 242.01077, 283.89954, 509.86032,   0.88982,   0.00000],
        [ 87.81021, 135.12836, 557.62671, 436.73010,   0.83110,   5.00000],
        [ 80.77100, 324.56339, 125.81225, 518.01605,   0.76313,   0.00000]])]
640x640 4 persons, 1 buss, Done. (1.795s)
image 2/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5/data/images/zidane.jpg:
pred:  [tensor([[3.74240e+02, 1.60377e+02, 5.71742e+02, 4.95508e+02, 9.30001e-01, 0.00000e+00],
        [2.16247e+02, 3.58464e+02, 2.62488e+02, 4.98221e+02, 9.05732e-01, 2.70000e+01],
        [5.98743e+01, 2.40431e+02, 5.40922e+02, 4.96992e+02, 8.55188e-01, 0.00000e+00],
        [4.93779e+02, 2.94994e+02, 5.66685e+02, 4.93369e+02, 4.44896e-01, 2.70000e+01]])]
640x640 2 persons, 2 ties, Done. (1.963s)
Results saved to runs/detect/exp11
Done. (3.936s)



Fusing layers...
Model Summary: 476 layers, 87730285 parameters, 0 gradients
image 1/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5/data/images/bus.jpg:
pred:  [tensor([[476.48029, 231.27528, 560.08221, 521.95844,   0.93258,   0.00000],
        [108.84730, 234.92026, 226.93616, 535.01819,   0.92405,   0.00000],
        [212.26106, 242.01077, 283.89954, 509.86032,   0.88982,   0.00000],
        [ 87.81021, 135.12836, 557.62671, 436.73010,   0.83110,   5.00000],
        [ 80.77100, 324.56339, 125.81225, 518.01605,   0.76313,   0.00000]])]
640x640 4 persons, 1 buss, Done. (1.907s)
image 2/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5/data/images/zidane.jpg:
pred:  [tensor([[3.74240e+02, 1.60377e+02, 5.71742e+02, 4.95508e+02, 9.30001e-01, 0.00000e+00],
        [2.16247e+02, 3.58464e+02, 2.62488e+02, 4.98221e+02, 9.05732e-01, 2.70000e+01],
        [5.98743e+01, 2.40431e+02, 5.40922e+02, 4.96992e+02, 8.55188e-01, 0.00000e+00],
        [4.93779e+02, 2.94994e+02, 5.66685e+02, 4.93369e+02, 4.44896e-01, 2.70000e+01]])]
640x640 2 persons, 2 ties, Done. (2.150s)
Results saved to runs/detect/exp17
Done. (4.237s)


```





开始移植到mlu平台上，模型量化，指对卷积操作时使用的数据类型变为int8类型，会导致精度产生一定的误差，但是是在允许范围内。



需要了解mlu的硬件特性，这是一款专注于深度学习加速的加速卡，因此对于卷积等运算可以采用int整型的数据，



首先加载模型并设置量化参数：

```python
    model = Model('models/yolov5x.yaml')
    state_dict = torch.load("../models/yolov5x.pt",map_location='cpu')['model'].state_dict()
    model.load_state_dict(state_dict)
    model.eval()                                  # 加载模型

    qconfig = {'use_avg':False, 'data_scale':1.0, 'firstconv':False, 'per_channel':False}
    quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype="int8",gen_quant=True) # 量化参数 需要量化的模型是什么，配置文件是什么等

```



引入mlu库和yolov5的模型定义文件

```python
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

from models.yolo import Model

```



使用有量化的模型在CPU上进行计算的运行结果：

```
image 1/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5/data/images/bus.jpg:
pred:  [tensor([[476.48029, 231.27528, 560.08221, 521.95844,   0.93258,   0.00000],
        [108.84730, 234.92026, 226.93616, 535.01819,   0.92405,   0.00000],
        [212.26103, 242.01074, 283.89951, 509.86035,   0.88981,   0.00000],
        [ 87.81021, 135.12836, 557.62671, 436.73010,   0.83110,   5.00000],
        [ 80.77100, 324.56339, 125.81226, 518.01599,   0.76313,   0.00000]])]
640x640 4 0s, 1 5s, Done. (3.803s)
image 2/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5/data/images/zidane.jpg:
pred:  [tensor([[3.74240e+02, 1.60377e+02, 5.71742e+02, 4.95508e+02, 9.30001e-01, 0.00000e+00],
        [2.16247e+02, 3.58464e+02, 2.62488e+02, 4.98221e+02, 9.05732e-01, 2.70000e+01],
        [5.98743e+01, 2.40431e+02, 5.40922e+02, 4.96992e+02, 8.55188e-01, 0.00000e+00],
        [4.93779e+02, 2.94994e+02, 5.66685e+02, 4.93369e+02, 4.44895e-01, 2.70000e+01]])]
640x640 2 0s, 2 27s, Done. (4.068s)
Results saved to runs/detect/exp16
Done. (7.956s)

```



设计逐层模式，便于后面的融合模式

加载量化模型

```python
model = Model('./models/yolov5x.yaml')
    state_dict = torch.load('yolov5x_int8.pth')
    quantized_net = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
    quantized_net.load_state_dict(state_dict,strict=False)
    quantized_net.eval()
    quantized_net.to(ct.mlu_device())
```



在YOLO.py中加入后处理算子，由bang语言编写，可以使框解码的部分从CPU挪到mlu上进行，以加快运算的速度

```python
 if x[0].device.type == 'mlu':          # 增加mlu的分支
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
                y = x[i].sigmoid()
                z.append(y)
            anchors_list=[10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
            num_anchors = len(anchors_list)
            img_h = 640
            img_w = 640
            conf_thres = 0.25
            iou_thres = 0.45
            maxboxnum = 1024

            detect_out = torch.ops.torch_mlu.yolov5_detection_output(z[0],z[1],z[2],anchors_list,self.nc,num_anchors,img_h,img_w,conf_thres,iou_thres,maxboxnum)
            return detect_out

```



测试两张图片时，需要根据detect.py中修改对应参数

```python
conf_thres = 0.25
            iou_thres = 0.45

```



将模型和输入数据拷到片上

```python
        pred = quantized_net(img.to(ct.mlu_device()))
        pred = pred.cpu()
        box = get_boxes(pred)
        print("pred: ",box)

```



使用get_boxes获得输出框结果

```python
def get_boxes(prediction, batch_size=1, img_size=640):
    '''
    Retirms detectopm with shape:
        (x1,y1,x2,y2,object_conf,class_score,class_pred)
    '''
    reshape_value = torch.reshape(prediction, (-1,1))
    num_boxes_final = reshape_value[0].item()
    print('num_boxes_final: ', num_boxes_final)
    all_list = [[] for _ in range(batch_size)]
    for i in range(int(num_boxes_final)):
        batch_idx = int(reshape_value[64+i*7+0].item())
        if batch_idx >= 0 and batch_idx < batch_size :
            bl = reshape_value[64+i*7+3].item()
            br = reshape_value[64+i*7+4].item()
            bt = reshape_value[64+i*7+5].item()
            bb = reshape_value[64+i*7+6].item()
            if bt - bl > 0 and bb - br > 0:
                all_list[batch_idx].append(bl)
                all_list[batch_idx].append(br)
                all_list[batch_idx].append(bt)
                all_list[batch_idx].append(bb)
                all_list[batch_idx].append(reshape_value[64+i*7+2].item())
                all_list[batch_idx].append(reshape_value[64+i*7+1].item())
        outputs = [torch.FloatTensor(all_list[i]).reshape(-1,6) for i in range(batch_size)]
    return outputs

```



由于CPU资源有限，所以编译的比较慢

运行结果：

```
batchNum: 1
image 1/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5_gc_test/data/images/bus.jpg: num_boxes_final:  5.0
pred:  [tensor([[476.29053, 230.71094, 560.72974, 523.50159,   0.93193,   0.00000]])]
batchNum: 1
image 2/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5_gc_test/data/images/zidane.jpg: num_boxes_final:  4.0
pred:  [tensor([[373.64774, 158.70975, 572.48346, 497.49402,   0.92274,   0.00000]])]
Results saved to runs/detect/exp25
Done. (38.616s)

```







设计融合模式，逐层算子被融合为大算子



首先需要加载模型

```python
ct.set_core_number(1)
#    ct.save_as_cambricon('yolov5x_int8_accuracy')
    trace_input = torch.randn(1,3,640,640,dtype=torch.float)
    trace_input = trace_input.to(ct.mlu_device())
    quantized_net = torch.jit.trace(quantized_net, trace_input, check_trace = False)

```



运行结果：

```
image 1/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5_gc_test/data/images/bus.jpg: num_boxes_final:  5.0
pred:  [tensor([[476.68817, 230.85555, 560.10675, 523.52972,   0.93168,   0.00000]])]
image 2/2 /workspace/volume/private/09_Yolov5x_0.06/yolov5_gc_test/data/images/zidane.jpg: num_boxes_final:  4.0
pred:  [tensor([[373.12573, 160.01617, 573.66650, 496.81299,   0.92676,   0.00000]])]
Results saved to runs/detect/exp26
Done. (56.317s)

```





生成离线模型



如何将离线模型运行在mlu上



代码逻辑依赖于cnrt运行时库，参考cnrt的用户手册，YOLOv3离线推理的样例代码



移植之后，可以尝试运行完整的数据集去验证精度





优化模型的推理性能

将batch改为16



使用分析工具进行性能分析：

|              |             |                |             |
| ------------ | ----------- | -------------- | ----------- |
| layer type   | layer count | layer time(us) | layer ratio |
| STRIDEDSLICE | 8           | 247868.6       | 32.71%      |
| ACTIVE       | 131         | 227525.8       | 30.02%      |
| TFU          | 72          | 155041.4       | 20.46%      |
| CONV         | 62          | 132281.3       | 17.46%      |
| POLL         | 159         | 113639.6       | 15.00%      |
| MULT         | 63          | 108451.4       | 14.31%      |
| ADD          | 25          | 25432.7        | 3.36%       |
| CONCAT       | 107         | 11208.6        | 1.48%       |
| PLUGIN       | 1           | 8790.8         | 1.16%       |
| INTERP       | 2           | 3137.5         | 0.41%       |
| MAXPOOL      | 3           | 1580.4         | 0.21%       |
| NOTIFY       | 162         | 1295.1         | 0.17%       |
| SPLIT        | 95          | 268.4          | 0.04%       |
| INIT_SYNC    | 1           | 14.4           | 0.00%       |
| total time   |             | 757817.9       |             |



提高性能主要应该提高TFU的比例

查看性能调优指南或者CNRT的用户使用手册





stride slice算子

将focus替换为conv加快模型的推理速度

## 在量化模型中加载新的focus权重

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.613
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.498
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.511
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.682
Mean AP: 0.613
Throughput(fps): 8.706933846524171
Latency(ms): 0.16619000000000042
```



## 调整tfu配置，对模型进行深度融合

```shell
export CNML_OPTIMIZE=USE_CONFIG:config.ini
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.613
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.498
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.511
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.682
Mean AP: 0.613
Throughput(fps): 7.29021124585599
Latency(ms): 0.17641839999999998
```

发现精度提高但是吞吐量下降

可能适合和半精度混用，有一定的准确率和较高的吞吐量










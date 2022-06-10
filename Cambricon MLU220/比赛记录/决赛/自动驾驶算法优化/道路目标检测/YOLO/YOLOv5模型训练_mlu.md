



## 训练模型

### 数据集准备

#### BDD100k

BDD100K是伯克利发布的开放式驾驶视频数据集，其中包含10万个视频和10个任务（因为把交通灯的颜色也区分了出来，实际上是13类分类任务），目的是方便评估自动驾驶图像识别算法的的进展。该数据集具有地理，环境和天气多样性，从而能让模型能够识别多种场景，具备更多的泛化能力。

![image-20211203205208896](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211203205208896.png)

前两个是label文件夹，对应于目标追踪，目标检测的标签数据，第三个是可驾驶区域的标签和图片数据集，第四个和第五个文件夹是图片文件，对应目标检测和目标追踪的数据集，目标追踪数据集为视频流的不同帧。

#### 数据预处理

Bdd100k的标签是由Scalabel生成的JSON格式，虽然伯克利提供Bdd100k数据集的标签查看及标签格式转化工具。由于没有直接从bdd100k转换成YOLO的工具，因此我们首先得使用将bdd100k的标签转换为coco格式，然后再将coco格式转换为yolo格式。

YOLO家族独特的标签数据集格式为：每个图片文件.jpg，都有同一命名的标签文件.txt。标签文件中每个对象独占一行，格式为

```xml
<object-class> <x> <y> <width> <height>。
1
```

其中：

- `<object-class>` 表示对象的类别序号：从0 到 (classes-1)。
- `<x> <y> <width> <height>` 参照图片宽度和高度的相对比例(浮点数值)，从0.0到1.0。

将bdd100k的标签格式转换为yolo的txt格式

（1）bdd 转化为coco格式

在完成bdd100k格式到yolo格式的转换后，会获得两个文件：

1. bdd100k_labels_images_det_coco_train.json
2. bdd100k_labels_images_det_coco_val.json

在执行转换的时候报错

![image-20211125213237708](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211125213237708.png)

json文件中的确有labels的属性,但是后面报错是为什么?

![image-20211128193752957](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211128193752957.png)

这10张图片没有labels键值

### 模型参数修改

copy了一份data/coco.yaml为data/diamond.yaml，修改里面的数据路径和类别数量

超参数配置文件，同样copy了一份data/hyp.scratch.yaml为data/hyp.diamond.yaml，没有做改动，使用默认超参数

网络结构配置文件，拷贝一份models/yolov5s.yaml为models/yolov5s_diamond.yaml。只需要改动配置文件中的类别参数，由coco的80类改为自己的4类。

写了一个.sh文件，train_diamon.sh

```shell
python train.py --data data/diamond.yaml \
             --hyp data/hyp.diamond.yaml \
             --cfg models/yolov5s_diamond.yaml \
             --name "yolov5s_diamond_20210128" \
             --batch-size 8
```

### train.py修改



![image-20211202154954250](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211202154954250.png)

修改default为false

![image-20211202160015632](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211202160015632.png)

device先默认设为cpu

![image-20211202155125855](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211202155125855.png)

应该修改为不加载权重数据，

![image-20211202161456730](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211202161456730.png)

注释此段代码，可能是resume时用的

### 调试过程

```shell
train: Scanning '/workspace/volume/private/GC_test/data/mybdd/images/train' images and labels... 0 found, 4265 missing, 0 empty, 0 corrupted:  18%|█▋       | 4265/23574 [03:04<11:54, 27.03it/s]train: WARNING: Ignoring corrupted image and/or label /workspace/volume/private/GC_test/data/mybdd/images/train/1f7bfc63-079c6646.jpg: cannot identify image file '/workspace/volume/private/GC_test/data/mybdd/images/train/1f7bfc63-079c6646.jpg'
train: Scanning '/workspace/volume/private/GC_test/data/mybdd/images/train' images and labels... 0 found, 23573 missing, 0 empty, 1 corrupted: 100%|███████| 23574/23574 [12:20<00:00, 31.83it/s]
train: WARNING: No labels found in /workspace/volume/private/GC_test/data/mybdd/images/train.cache. See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
train: New cache created: /workspace/volume/private/GC_test/data/mybdd/images/train.cache
Traceback (most recent call last):
  File "train.py", line 565, in <module>
    train(hyp, opt, device, tb_writer)
  File "train.py", line 204, in train
    image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
  File "/workspace/volume/private/GC_test/Yolov5_diamond/yolov5/utils/datasets.py", line 76, in create_dataloader
    prefix=prefix)
  File "/workspace/volume/private/GC_test/Yolov5_diamond/yolov5/utils/datasets.py", line 401, in __init__
    assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'
AssertionError: train: No labels in /workspace/volume/private/GC_test/data/mybdd/images/train.cache. Can not train without labels. See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

```



![image-20211202171232129](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211202171232129.png)

不使用图片缓存

将image_cache参数修改为false






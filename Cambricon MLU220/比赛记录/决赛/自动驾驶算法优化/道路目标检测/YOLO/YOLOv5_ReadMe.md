

## YOLOv5模型移植

├── README.md
├── models
│   ├── yolov5x.pt             参赛模型，md5:000570448c8bc5afc5f67f0fba973419
│   └── yolov5x.pt.original    原始模型，md5:4f7eee7ab596ed6f9496520cb304c7cb
├── val2017.txt
├── file_list
└── yolov5                     赛题源码

1 models中有两个模型

  1. yolov5x.pt是经过转换到torch 1.3能够加载的yolov5x模型
  2. yolov5x.pt.original是原始模型可以在torch 1.7中加载，如果需要使用，只需mv yolov5x.pt.original model_name.pt

> 云平台的torch版本
>
> 1.3.0a0
>
> ![image-20211124213704757](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211124213704757.png)

2 准备数据集
  在file_list同级目录下 ln -s /workspace/dataset/public/zhumeng-dataset/coco_2017 coco

> ![image-20211124213837512](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211124213837512.png)



> 运行cpu模式
>
> 修改detect.py内的模型路径为../models/yolov5x.pt
>
> 在torch内注册silu和hardswish激活函数
>
> 在 activation init文件里添加silu，hardswish



> self.parser.add_argument('--lr_use', action='store_true', default=False, help='if or not use lr_loss')  
>
> 当在终端运行的时候，如果不加入--lr_use，那么程序running的时候，lr_use的值为default: False 
>
> 如果加上了--lr_use，不需要指定True/False,那么程序running的时候，lr_use的值为True 



> cpu模式跑通





3 python test.py

  结果：
  IoU metric: bbox
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.689
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.541
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.350
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.550
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.642
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.382
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.625
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.674
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.517
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.726
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.817

4 精度测试方法: 精度测试数据集为coco2017 validation，共5000张
   以 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503 为准，参赛选手适配好模型后精度需要满足：AP >= 0.493

5 性能测试方法: 前500张，具体请看file_list

6 备注
  数据集目录结构:
  COCO_PATH/
  ├── annotations
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  └── val2017
      ├── 000000000139.jpg
      ├── 000000000285.jpg
      ├── ...
      ├── 000000581615.jpg
      └── 000000581781.jpg

github参考链接: https://github.com/ultralytics/yolov5






























































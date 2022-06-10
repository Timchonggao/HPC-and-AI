搭建环境和训练模型的一些问题：1）tensorflow和CUDA版本匹配问题 2）gpu版本可以正常训练，用了全部的四块gpu，每块gpu绑定了一个进程，每块gpu上都有数据存放，但是计算任务都是在一块gpu上完成，并且利用率不高只有30%-40% 3）gpu版本，训练模型正常无误，但是freeze模型的时候会报段错误segmentation default，这个待解决 





我之前遇到过cu值报错的问题，群里说在type_map加入这个值重新训练就能解决





**GPU利用率查看**

watch -n 1 nvidia-smi



两个idea：1.试一试c的接口 2.找找Python程序热点代码分析的工具或方法

分析程序内部的瓶颈在哪，可能是个函数耗时很长，或者是一个代码段耗时很长，这就是应该着重优化的地方。



![image-20220202173934078](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202173934078.png)










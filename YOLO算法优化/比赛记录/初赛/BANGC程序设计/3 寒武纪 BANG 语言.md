## 寒武纪 BANG 语言



![image-20210916202431633](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916202431633.png)



BANG语言执行流程

![image-20210916202600625](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916202600625.png)





寒武纪MLU多核架构

![image-20210916202711823](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916202711823.png)



MLU服务层级

![image-20210916202823464](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916202823464.png)



 MLU硬件架构

![image-20210916202946550](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916202946550.png)



MLU异构编程

![image-20210916203126655](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916203126655.png)



MLU异构编程的概念

![image-20210916203408082](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916203408082.png)



host端程序示例

![image-20210916203653166](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916203653166.png)



MLU端程序示例

![image-20210916203806915](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916203806915.png)



并行编程-任务调度类型

不同声明的任务类型所使用的的资源不同，当任务在不同队列中时，可以实现并行

![image-20210916204324390](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916204324390.png)



并行编程-任务规模

![image-20210916204417209](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916204417209.png)



并行内建变量

![image-20210916210031256](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916210031256.png)



**将任务分为三个维度**



并行内建变量-示例

![image-20210916204639736](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916204639736.png)



并行内建变量的理解：

![image-20210916205025823](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916205025823.png)



job调用的核数可以理解为设置的func_type调用的核数，总的调用核数为x/y/z三个维度的乘积



![image-20210916205850728](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210916205850728.png)



taskDimX不小于func_type



![image-20210924211341245](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210924211341245.png)



![image-20210924211451750](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210924211451750.png)



![image-20210924211552311](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210924211552311.png)



![image-20210924211620573](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210924211620573.png)



![image-20210924211817288](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210924211817288.png)



![image-20210924212105111](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20210924212105111.png)


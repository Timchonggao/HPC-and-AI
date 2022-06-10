

单 GPU 训练优化：分析程序性能瓶颈，寻找优化策略，可以参考 fgure1，视频里的提示有（可能不止这些）：

- 两次反向求导的计算量很大，占用时间比较大，可以考虑优化这部分。  
- 自定义的算子可以考虑用 kernel fussion 策略优化。  

![image-20220202013546048](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202013546048.png)





## 单GPU训练

加速训练流程:

减少计算需求:

理清训练流程和计算的地方在哪:





GPU利用率低：GPU利用率低可能原因是因为计算任务少，因此GPU没有完全利用起来。带宽允许下，可以考虑多分配一些计算任务。



程序的执行流程，函数的花费时间，具体分析函数内部的内容：

- 算法复杂度
- 分析for循环



CUDAprofile工具的使用：

- 并行计算
- 读写匹配
- 数据复用
- 调用计算算子



分析profile的结果：py-spy结果分析；nvprof分析；self-profile结果分析；TensorBoard分析

深入源码，理解TensorFlow的流程和优化方法

阅读文献，开始着手写作





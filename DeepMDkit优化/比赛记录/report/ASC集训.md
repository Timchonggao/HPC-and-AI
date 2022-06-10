

## ASC集训营

![image-20220202003615369](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202003615369.png)

![image-20220202003645949](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202003645949.png)

![image-20220202003814361](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202003814361.png)

AIMD基于量子力学的方式,使用密度泛函理论(DFT)

EFT经验力场,优点是快,但是精度比较低

![image-20220202004109325](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202004109325.png)

**计算机发展与计算规模的增长关系**

![image-20220202005039546](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202005039546.png)

**summit是一个超算机器**

**注意图的坐标的大小变化**

![image-20220202005757405](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202005757405.png)



**理解并行**

### 将模型分块,通信和计算,适合多GPU  **如何使用多GPU进行处理**

![image-20220202005957440](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202005957440.png)



**精度对比**

**算力需求对比**

![image-20220202010205999](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202010205999.png)



**一些有意思的工作**

多项式拟合简单神经网络提升速度

DPGEN生成DP的训练数据

![image-20220202010526869](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202010526869.png)



DP的KEY

### 网络训练 kernel fusion

![image-20220202010952683](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202010952683.png)



**idea:使用机器学习的方式拟合这个高维函数**



![image-20220202011214441](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202011214441.png)

数据源于DFT的精确数据



![image-20220202011331801](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202011331801.png)

![image-20220202011834598](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202011834598.png)



![image-20220202012501408](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202012501408.png)



### **可以参照这部分进行优化**

![image-20220202012607421](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202012607421.png)

第一步坐标转化为相对坐标,需要对周围原子进行计算,**最好对空间进行分块,然后用GPU处理,用GPU进行分块比较好**

网络多个的原因是因为需要考虑不同种元素的情况下需要使用不同的网络,input.json文件可配置具体细节

![image-20220202013546048](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202013546048.png)



![image-20220202013721123](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202013721123.png)



**任务为加速训练过程,训练过程分析**

**1两次反向求导**

**2不同元素需要不同网络模型**

**3自定义OP可以使用kernel fusion等优化策略**

**主要看相对提升量,绝对提升量是一个参考**

**三个数据集只需要做一套优化,然后进行测试就可以了**



具体要做什么

首先需要自己跑一个baseline,input.json文件中的type_oneside优化N的网络还是N*N的网络,优化单精度还是双精度,**这里选择一个baseline**,看代码,在代码底层进行优化,如kernel fusion ,训练加速,精度应该控制与baseline的精度差不多

评判标准是相对提升



单精度比双精度稍微快一些,且精度差不多

半精度可以更快,但是在求梯度的时候可能因为梯度比较大不能存储这么大的数,导致精度下降

混合精度优化的话,不能只修改input.json文件,需要指定网络中那一部分用什么精度进行训练,需要保证bseline和优化的一致



因为dp-kit对训练的随机数据进行处理了,如果环境以及代码一致的话,两次训练精度结果应该是一致的



优化前和优化后,精度需要保持变化不大





![image-20220202020328523](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220202020328523.png)

neuron确定网络的大小,下一层必须上一层的两倍或者一倍,原因是应为加入了残差模块

sel参数决定训练过程中的tensor的大小



可以从main.py开始看,dp train的过程为main.py train.py trainer.py model.py的过程



只优化dp train这个过程



每一次修改代码后都需要重新编译,在根目录下执行pip install. 命令进行重新编译,或者使用pip install -e.命令进行重新编译



数据格式参考



baseline也应该保证势能函数预测准确



训练次数是固定的10000次,确定次数后用训练时长来进行评判优化的结果



程序热点分析工具如nvprof









分析具体的并行的过程

具体源码并行的地方，分析具体过程，分析项目流程












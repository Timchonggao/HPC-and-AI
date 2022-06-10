

### tf.Graph().as_default()

tf.Session()创建一个会话，当上下文管理器退出时会话关闭和资源释放自动完成。 tf.Session().as_default()创建一个默认会话，当上下文管理器退出时会话没有关闭，还可以通过调用会话进行run()和eval()操作

https://blog.csdn.net/nanhuaibeian/article/details/101862790



### tf.placeholder函数说明

ensorflow的设计理念称之为计算流图，在编写程序时，首先构筑整个系统的graph，代码并不会直接生效，这一点和python的其他数值计算库（如Numpy等）不同，graph为静态的，类似于docker中的镜像。然后，在实际的运行时，启动一个session，程序才会真正的运行。这样做的好处就是：避免反复地切换底层程序实际运行的上下文，tensorflow帮你优化整个系统的代码。我们知道，很多python程序的底层为C语言或者其他语言，执行一行脚本，就要切换一次，是有成本的，tensorflow通过计算流图的方式，帮你优化整个session需要执行的代码，还是很有优势的。

   所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。

https://blog.csdn.net/kdongyi/article/details/82343712

```python

import tensorflow as tf
import numpy as np
 
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)
 
with tf.Session() as sess:
    #print(sess.run(y))  # ERROR:此处x还没有赋值
    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))

```







### 安装TensorFlow（CPU版本）

使用anaconda promot

```
conda create -n tensorflow python=3.6
activate tensorflow
pip install --upgrade --ignore-installed tensorflow    #这里默认安装的是2.6版本，使用教程和网上大多不相同，但是不影响使用TensorBoard
```







### TensorBoard使用

它刻意将模型抽象成图像，`tensor`每一步是如何流动的，一目了然。

通过适当的代码设置，还能将指定的关键变量在训练时的变化情况绘制成曲线图，以便训练完成后观察变量的变化情况，来更加准确定位问题。

这篇文章简单介绍一下`tensorboard`的基本用法。



`tb`的使用，大致归纳为三步：

- 调用`tf`中的`FileWriter`将自己关注的数据写到磁盘
- 在命令行使用`tensorboard --logdir /path/to/log`启动`tb`的`web app`
- 然后在本地浏览器输入`localhost:6006`来使用`tb`


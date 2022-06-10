

# 介绍

寒武纪机器学习库（Cambricon Neuware Machine Learning Library，[CNML](https://www.cambricon.com/docs/cnml/user_guide/glossary/index.html#term-CNML)）是一个基于寒武纪机器学习单元（Machine Learning Unit，MLU）并针对机器学习以及深度学习的编程库。CNML为用户提供简洁、高效、通用、灵活并且可扩展的编程接口，用于在MLU上加速用户各种机器学习和深度学习算法。

CNML提供了丰富的基本算子。通过组合基本算子可以实现多样的机器学习和深度学习算法。

**寒武纪软件栈**

在寒武纪软件栈中，**CNML负责对编程框架如TensorFlow提供的模型文件进行解析和编译，生成寒武纪指令等模型数据文件**。软件栈结构如下所示：

![../_images/cnml_user_concept1.png](https://www.cambricon.com/docs/cnml/user_guide/_images/cnml_user_concept1.png)

用户可以通过各种机器学习、深度学习网络和CNML提供的丰富算子构造整个网络。再通过CNML编译生成可执行镜像。该镜像可以直接运行，也可以保存为离线模型。

![../_images/cnml_user_concept2.png](https://www.cambricon.com/docs/cnml/user_guide/_images/cnml_user_concept2.png)


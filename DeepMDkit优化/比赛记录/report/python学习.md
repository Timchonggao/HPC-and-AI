



## python学习之argparse模块

argparse是python用于解析命令行参数和选项的标准模块

argparse模块的作用是用于解析命令行参数，例如 python parseTest.py input.txt output.txt --user=name --port=8080

使用步骤：

1：import argparse

2：parser = argparse.ArgumentParser()

3：parser.add_argument()

4：parser.parse_args()

首先导入该模块；然后创建一个解析对象；然后向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；最后调用parse_args()方法进行解析；



## python re.compile() 详解——Python正则表达式

### 正则表达式

正则表达式 英文名称叫 Regular Expression简称[RegEx](https://so.csdn.net/so/search?q=RegEx&spm=1001.2101.3001.7020)，是用来匹配字符的一种工具，它常被用在网页爬虫，文稿整理，数据筛选等方面，最常用的就是用在网页爬虫，数据抓取。[正则表达式](https://so.csdn.net/so/search?q=正则表达式&spm=1001.2101.3001.7020)已经内嵌在Python中，通过导入re模块就可以使用。

### re模块

正则表达式是用在findall()方法当中，大多数的字符串检索都可以通过findall()来完成。

![img](https://img-blog.csdnimg.cn/20200328113857419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0NzU1MA==,size_16,color_FFFFFF,t_70)![img](https://img-blog.csdnimg.cn/20200328113857419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0NzU1MA==,size_16,color_FFFFFF,t_70)



```python
import re
s = "a123456b"
rule = "a[0-9][1-6][1-6][1-6][1-6][1-6]b"	#这里暂时先用这种麻烦点的方法，后面有更容易的，不用敲这么多[1-6]
l = re.findall(rule,s)
print(l)
```

输出结果为

```python
['a123456b']
```



当我们在Python中使用[正则表达式](https://so.csdn.net/so/search?q=正则表达式&spm=1001.2101.3001.7020)时，re模块内部会干两件事情：

1. 编译正则表达式，如果正则表达式的字符串本身不合法，会报错；
2. 用编译后的正则表达式去匹配字符串。

那么如果一个正则表达式要重复使用几千次，出于效率的考虑，我们是不是应该先把这个正则先预编译好，接下来重复使用时就不再需要编译这个步骤了，直接匹配，提高我们的效率

预编译十分的简单,re.compile()即可；演示如下：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/1/17 15:55
# @Author  : Arrow and Bullet
# @FileName: compile.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_41800366
import re


re_telephone = re.compile(r'^(\d{3})-(\d{3,8})$')  # 编译

A = re_telephone.match('010-12345').groups()  # 使用
print(A)  # 结果 ('010', '12345')
B = re_telephone.match('010-8086').groups()  # 使用
print(B)  # 结果 ('010', '8086')
```

编译后生成Regular Expression对象，由于该对象自己包含了正则表达式，所以调用对应的方法时不用给出正则字符串。





## python logging模块

[参考](https://www.cnblogs.com/yyds/p/6901864.html)



## python typing模块

 [Python - typing 模块 —— Optional](https://www.cnblogs.com/poloyy/p/15170297.html)





### python魔术方法

python 的类中，所有以双下划线__包起来的方法，叫魔术方法，魔术方法在类或对象的某些事件发出后可以自动执行，让类具有神奇的魔力，比如常见的构造方法__new__、初始化方法__init__、析构方法__del__

__new__是开辟疆域的大将军，而__init__是在这片疆域上辛勤劳作的小老百姓，只有__new__执行完后，开辟好疆域后，__init__才能工作，结合到代码，也就是__new__的返回值正是__init__中self。https://blog.csdn.net/sj2050/article/details/81172022





### python中super().__init__()和python中继承

super().__init__()，就是继承父类的init方法，同样可以使用super()去继承其他方法。

https://blog.csdn.net/a__int__/article/details/104600972



### python中with...as的用法(包括上下文管理器try ... except等)

with…as语句执行顺序：
–>首先执行expression里面的__enter__函数，它的返回值会赋给as后面的variable，想让它返回什么就返回什么，只要你知道怎么处理就可以了，如果不写as variable，返回值会被忽略。

–>然后，开始执行with-block中的语句，不论成功失败(比如发生异常、错误，设置sys.exit())，在with-block执行完成后，会执行expression中的__exit__函数。

当with...as语句中with-block被执行或者终止后，这个类对象应该做什么。如果这个码块执行成功，则exception_type,exception_val, trace的输入值都是null。如果码块出错了，就会变成像try/except/finally语句一样，exception_type, exception_val, trace 这三个值系统会分配值。
https://blog.csdn.net/qiqicos/article/details/79200089



### cpython是什么_什么是CPython

CPython

当我们从Python官方网站下载并安装好Python 3.5后，我们就直接获得了一个官方版本的解释器：CPython。这个解释器是用C语言开发的，所以叫CPython。在命令行下运行python就是启动CPython解释器。

CPython是使用最广的Python解释器。教程的所有代码也都在CPython下执行。

IPython

IPython是基于CPython之上的一个交互式解释器，也就是说，IPython只是在交互方式上有所增强，但是执行Python代码的功能和CPython是完全一样的。好比很多国产浏览器虽然外观不同，但内核其实都是调用了IE。

CPython用>>>作为提示符，而IPython用In [序号]:作为提示符。

PyPy

PyPy是另一个Python解释器，它的目标是执行速度。PyPy采用JIT技术，对Python代码进行动态编译（注意不是解释），所以可以显著提高Python代码的执行速度。

绝大部分Python代码都可以在PyPy下运行，但是PyPy和CPython有一些是不同的，这就导致相同的Python代码在两种解释器下执行可能会有不同的结果。如果你的代码要放到PyPy下执行，就需要了解PyPy和CPython的不同点。

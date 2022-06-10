

## 一、cProfile介绍

- cProfile自python2.5以来就是标准版Python解释器默认的性能分析器。
- 其他版本的python，比如PyPy里没有cProfile的。
- cProfile是一种确定性分析器，只测量CPU时间，并不关心内存消耗和其他与内存相关联的信息。



## 二、支持的API

#### （一）run(command, filename=None, sort=-1)

##### 第一种情况：

```python
def bar(n):
    a = 0
    for i in range(n):
        a += i**2
    return a    

def foo():
    ret = 0
    for i in range(1000):
        ret += bar(i)
    return ret

def c_profile(codestr):
    import cProfile
    p = cProfile.Profile()
    p.run(codestr)
    p.print_stats()

if __name__ == '__main__':
    c_profile("foo()")

# python test.py

```

输出

```
         1004 function calls in 0.138 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.138    0.138 <string>:1(<module>)
     1000    0.137    0.000    0.137    0.000 test.py:1(bar)
        1    0.000    0.000    0.138    0.138 test.py:7(foo)
        1    0.000    0.000    0.138    0.138 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

- 第一行：1004个函数调用被监控(不涉及递归）
- ncalls：函数被调用的次数。如果这一列有两个值，就表示有递归调用，第二个值是原生调用次数，第一个值是总调用次数。
- tottime：函数内部消耗的总时间。（可以帮助优化）
- percall：是tottime除以ncalls，一个函数每次调用平均消耗时间。
- cumtime：之前所有子函数消费时间的累计和。
- filename:lineno(function)：被分析函数所在文件名、行号、函数名。

##### 第二种情况：

```python
import cProfile
import re
cProfile.run('re.compile("aaa|bbb")', 'stats', 'cumtime')
```



#### （二）runctx(command, globals, locals, filename=None）

```python
import cProfile
# 这样才对
def runRe():
    import re
    cProfile.runctx('re.compile("aaa|bbb")', None, locals())
runRe()
```

#### （三）Profile(custom_timer=None, time_unit=0.0, subcalls=True, builtins=True)

- custom_timer：是一个自定义参数，可以通过与默认函数不同的方式测量时间。
- 如果custom_timer返回的是一个整数，time_unit是单位时间换成秒数。


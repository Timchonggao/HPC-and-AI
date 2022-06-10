repo: [https://github.com/benfred/py-spy](https://link.zhihu.com/?target=https%3A//github.com/benfred/py-spy)

可以直接使用 pip 进行安装:

```bash
pip install py-spy
```

使用方法:

```bash
# 基本使用方法：
py-spy record --format speedscope -o output.json --native -- python3 xxx.py
  
# =====
# 主要参数：
# --format: 输出格式，默认是 flamegraph，推荐使用 speedscope 格式
# --output: 输出文件名
# --native: 是否采样 C/C++调用栈，不加--native 只会对 Python 调用栈进行采样
  
# =====
# 其他参数
# --rate:          采样频率，默认值为 100，打开--native 会增大 overhead，建议打开--native 时调低--rate
# --subprocesses:  是否对子进程进行采样，默认为否
# --gil:           是否只对拿到 Python GIL 的线程进行采样
# --threads:       在输出中显示 thread id
# --idle:          是否对 idle（即 off-CPU time) 的 thread 进行采样，默认为否，根据具体场景选择是否打开
  
# =====
# 例子：
py-spy record -sti --rate 10 --format speedscope --native -o output.json -- python3 xxx.py
 
# 除了在启动时运行 py-spy，还可以对一个已经运行的 python 进程进行采样，如：
py-spy record --format speedscope -o output.json --native --pid 12345
```



运行命令如

```
py-spy record --rate 10 --format speedscope -o output.json --native dp train input.json
py-spy record --rate 10 --format speedscope -o profile.svg --native dp train input.json
```



See `py-spy record --help` for information on other options including changing the sampling rate, filtering to only include threads that hold the GIL, profiling native C extensions, showing thread-ids, profiling subprocesses and more.


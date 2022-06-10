​	

# Profiling the code

Profiling the code consists in finding which parts of the algorithm **dominate the computational time**, for your particular simulation setup.

## Profiling the code executed on CPU

Dumping the profiling results in a text file allows you to quickly profile the execution of a simulation.

Run the code with [cProfile](http://docs.python.org/2/library/profile.html) :

```
python -m cProfile -s time fbpic_script.py > cpu.log
```

and then open the file `cpu.log` with a text editor.

## Profiling the code executed on GPU

Two profiling tools exists for GPU:

> - [nvprof](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview)
> - [Nsight Systems](https://docs.nvidia.com/nsight-systems/) (which can be installed from [this page](https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2019-5))

Instructions here are given for both tools.

### Getting the results in a simple text file

- For **nvprof**: First run the code with `nvprof`

  > ```
  > nvprof --log-file gpu.log dp train input.json -l train.log
  > ```
  >
  > and then open the file `gpu.log` with a standard text editor.

- For **Nsight Systems**: Run the code with `nsys profile`

  > ```
  > nsys profile --stats=true dp train input.json -l train.log
  > ```
  >
  > The profiling information is printed directly in the Terminal output.

In order to simultaneously profile the device-side (i.e. GPU-side) and host-side (i.e. CPU-side) code, you can use:

```
nvprof --log-file gpu.log python -m cProfile -s time fbpic_script.py > cpu.log
```









## Nvprof

### 使用如下命令得到简单的log文件

```
nvprof --log-file gpu.log dp train input.json -l train.log
# nvprof --log-file gpu.log python fbpic_script.py
```

```
==18027== NVPROF is profiling process 18027, command: //home/asc22g0/GC/hov/bin/python3.7 /home/asc22g0/GC/hov/bin/dp train input.json -l train.log
==18027== Profiling application: //home/asc22g0/GC/hov/bin/python3.7 /home/asc22g0/GC/hov/bin/dp train input.json -l train.log
==18027== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   12.20%  1.25667s     26264  47.847us  9.0880us  135.30us  volta_dgemm_64x64_nt
                   11.14%  1.14812s     30572  37.554us  9.9200us  128.00us  volta_dgemm_64x64_nn
                    7.77%  800.32ms     10000  80.032us  62.240us  100.48us  void dgemm_largek<bool=0, bool=1, int=4, int=4, int=4, int=3, int=3, int=16>(double*, double const *, double const *, int, int, int, int, int, int, double const *, double const *, double, double, int, int, int*, int*)
                    6.99%  720.30ms     96616  7.4550us  1.6960us  38.592us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<double const , double const >, Eigen::TensorMap<Eigen::Tensor<double const , int=1, int=1, long>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<double const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, long>(double, int=1)
                    6.88%  708.62ms    228119  3.1060us  1.2800us  249.70us  [CUDA memcpy HtoD]
                    5.80%  598.16ms     12352  48.425us  28.384us  63.200us  volta_dgemm_64x64_tn
                    5.33%  549.22ms     54792  10.023us  2.3040us  38.592us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_tanh_gradient_op<double>, Eigen::TensorMap<Eigen::Tensor<double const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<double const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, long>(double, int=1)
                    3.73%  384.17ms     52176  7.3620us  1.4080us  38.240us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, Eigen::TensorMap<Eigen::Tensor<double const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<double const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, long>(double, int=1)
                    3.68%  379.56ms     83678  4.5350us  1.3120us  206.37us  [CUDA memcpy DtoH]
                    3.01%  310.25ms     12264  25.297us  18.464us  40.000us  void gemm_kernel2x2_tile_multiple_core<double, bool=1, bool=0, bool=0, bool=1, bool=0>(double*, double const *, double const *, int, int, int, int, int, int, double*, double*, double, double, int)
                    3.00%  309.24ms     65408  4.7270us  1.5040us  16.033us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, int=2, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, int=2> const , Eigen::DSizes<int, int=2> const , Eigen::TensorMap<Eigen::Tensor<double const , int=2, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(double, int=2)
                    2.40%  247.41ms      2000  123.71us  115.46us  130.21us  void dgemm_largek<bool=0, bool=1, int=5, int=5, int=4, int=4, int=4, int=32>(double*, double const *, double const *, int, int, int, int, int, int, double const *, double const *, double, double, int, int, int*, int*)
                    2.39%  246.34ms     53000  4.6480us  4.2560us  6.0160us  void tensorflow::_GLOBAL__N__71_tmpxft_0000474a_00000000_11_dynamic_stitch_op_gpu_cu_compute_70_cpp1_ii_d0a4f2f9::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, int=8>, tensorflow::_GLOBAL__N__71_tmpxft_0000474a_00000000_11_dynamic_stitch_op_gpu_cu_compute_70_cpp1_ii_d0a4f2f9::DynamicStitchKernel<int, int const *, int=8>, tensorflow::GpuDeviceArrayStruct<int, int=8>*)

API calls:   49.29%  9.44319s    939442  10.051us  4.0850us  1.73134s  cudaLaunchKernel
                   12.20%  2.33663s         4  584.16ms  584.07ms  584.24ms  cuDevicePrimaryCtxRetain
                    8.91%  1.70784s    216118  7.9020us  2.3060us  104.91ms  cuMemcpyHtoDAsync
                    8.73%  1.67170s    422842  3.9530us     305ns  798.78us  cuEventRecord
                    6.11%  1.17059s     83678  13.989us  4.0490us  488.96us  cuMemcpyDtoHAsync
                    4.43%  848.83ms      2059  412.25us  3.5920us  1.0368ms  cuCtxSynchronize
                    3.15%  604.23ms    370698  1.6290us     378ns  1.4372ms  cuEventQuery
                    2.98%  571.71ms        12  47.643ms     582ns  373.31ms  cudaFree
                    1.24%  238.46ms    211421  1.1270us     452ns  486.91us  cuStreamWaitEvent
                    1.11%  212.24ms     19853  10.690us  5.9440us  162.83us  cudaMemcpyAsync
                    0.48%  92.498ms        65  1.4231ms  1.3112ms  2.8645ms  cuDeviceTotalMem
                    0.29%  55.095ms      8088  6.8110us  4.6550us  20.670us  cudaMemsetAsync
                    0.26%  49.788ms     22088  2.2540us  1.4380us  83.549us  cudaEventQuery
                    0.24%  46.533ms    200552     232ns     123ns  468.35us  cudaGetLastError
                    0.22%  41.658ms      2504  16.636us     138ns  4.8664ms  cuDeviceGetAttribute
                    0.16%  30.921ms     22088  1.3990us     930ns  10.004us  cudaEventRecord
                    0.08%  15.452ms         5  3.0904ms  1.4306ms  6.3375ms  cuMemHostAlloc
                    0.04%  7.0327ms        18  390.70us  359.28us  431.41us  cudaGetDeviceProperties
                    0.03%  6.0475ms         8  755.94us  650.58us  1.0038ms  cuMemAlloc
                    0.01%  2.6944ms        65  41.452us  33.592us  145.62us  cuDeviceGetName
                    0.01%  2.3750ms        12  197.92us  158.89us  255.80us  cuCtxEnablePeerAccess
                    0.01%  970.32us        46  21.093us  1.7530us  214.90us  cuStreamCreate
                    0.00%  254.90us         9  28.322us  18.488us  51.923us  cuMemGetInfo
                    0.00%  105.48us         3  35.159us  17.371us  57.523us  cudaMalloc
                    0.00%  92.800us         4  23.200us     463ns  32.865us  cuDevicePrimaryCtxGetState
                    0.00%  78.622us        49  1.6040us  1.1860us  2.5070us  cuDeviceGetPCIBusId
                    0.00%  78.392us       112     699ns     197ns  5.6810us  cuCtxSetCurrent
                    0.00%  68.613us        66  1.0390us     308ns  14.002us  cuEventCreate
                    0.00%  52.631us        90     584ns     416ns  3.9160us  cudaFuncSetAttribute
                    0.00%  23.783us        73     325ns     159ns  1.3380us  cuDeviceGet
                    0.00%  22.842us         1  22.842us  22.842us  22.842us  cuMemsetD32
                    0.00%  19.252us         1  19.252us  19.252us  19.252us  cudaMemcpy
                    0.00%  13.543us        12  1.1280us     408ns  6.1150us  cudaSetDevice
                    0.00%  13.248us        72     184ns     142ns     488ns  cuCtxGetDevice
                    0.00%  13.186us        36     366ns     218ns     962ns  cuDeviceCanAccessPeer

```

- Profiling result：是GPU（kernel函数）上运行的时间
- API calls：是在cpu上测量的程序调用API的时间

**暂时不知道怎么从这个log文件中分析**







### 基本命令:



```
nvprof --unified-memory-profiling off --log-file gpu.log dp train input.json -l train.log  #直接打印出,因为某块内存被设置了不允许分析,所以添加参数
nvprof --print-gpu-trace dp train input.json -l train.log  # 直接打印出log
```



```
Further options of potential interest:
• --print-gpu-trace: Show trace of function calls
• --openacc-profiling on: Profile OpenACC as well (on by default)
• --cpu-profiling on: Enable some CPU profiling
• --csv --log-file FILE: Generate CSV output and save to FILE; handy for plots or benchmarked analysis
• --metrics M1: Measure only metric M1 which is one of the NVIDIA-provided metrics which can be listed via
--query-metrics.
```



追踪API

```
nvprof --print-api-trace dp train input.json
```

不需要时可以通过–profile-api-trace none关掉这个功能



Event/metric总结模式

```python
nvprof --events warps_launched,local_load --metrics ipc  ./a.out（python aa.py）
```



Event/metric追踪模式

```python
nvprof --aggregate-mode off --events local_load --print-gpu-trace ./a.out（python aa.py）
```



Timeline

```
nvprof --export-profile timeline.prof ./a.out（python aa.py）
nvprof --metrics achieved_occupancy,executed_ipc -o metrics.prof <app> <app args>
nvprof --kernels <kernel specifier> --analysis-metrics -o analysis.prof <app> <app args>
```



保存为文件

```python
nvprof -o profileOutput ./a.out
nvprof --export-profile timeline.prof ./a.out
nvprof --log-file output.log ./a.out
```



```
nvprof --metrics gld_efficiency,gst_efficiency ./myproc
检测内存加载存储效率
```



```
nvprof --query-metrics
# 查看所有能用的参数命令
```



```
nvprof --metrics stall_sync ./myproc
检测核函数的线程束阻塞情况
```



```
nvprof --metrics gld_throughput ./myproc
检测内存加载吞吐量
```



```
nvprof --metrics branch_efficiency  ./myproc
检测分支分化性能
```



```
nvprof ./a.out
profiling result中显示的是kernel执行的time情况 api calls则显示的是程序调用的api所耗费的time情况
```



**使用nvprof输出kernel timeline数据**

Kernel Timeline 输出的是以gpu kernel 为单位的一段时间的运行时间线，我们可以通过它观察GPU在什么时候有闲置或者利用不够充分的行为，更准确地定位优化问题。nvprof是nvidia提供的用于生成gpu timeline的工具，其为cuda toolkit的自带工具。使用方法如下：

```bash
nvprof -o out.nvvp -f --csv --profile-from-start off python3 mnist.py
```

`-o`用于输出 `.nvvp` 文件，`-f` 用于强制覆盖， `--csv`可是在console输出除 timeline 以外数据的时候以逗号分隔数据，方便拷贝至csv文件。我们需要重点讲一下`--profile-from-start` 。

往往我们只需要监测中间模型运行的部分，忽略掉预处理的部分，让生成的timeline短一些。所以需要让nvprof不要从程序一开始就运行，而是在代码中手动设置开启。设置开启的方法是先在代码中插入如下函数：

```python
import ctypes
_cudart = ctypes.CDLL('libcudart.so')

def cu_prof_start():
  ret = _cudart.cudaProfilerStart()
  if ret != 0:
    raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
  ret = _cudart.cudaProfilerStop()
  if ret != 0:
    raise Exception('cudaProfilerStop() returned %d' % ret)
```

然后在对应位置插入：

```python
for epoch in range(EPOCHS): 
   ...
   flag = False
   cu_prof_start() 
   for images, labels in train_ds:
     if flag:
        cu_prof_stop()
     flag =  True
     train_step(images, labels) 
   ...
```

这样我们就可以只监测一个迭代步了。



### nvvp



```
nvprof --unified-memory-profiling off -o prof.nvvp dp train input.json -l train.log # 有时候使用print-gpu-trace会有问题
# nvprof --unified-memory-profiling off --print-gpu-trace -o prof.nvvp dp train input.json -l train.log # 输出可用于nvvp分析的文件
```

使用sz命令发送至本地使用nvvp软件分析



```
nvprof --export-profile timeline.prof ./a.out（python aa.py）
```





### 2.2.2. Import Session

Using the import dialog you can select one or more nvprof data files for import into the new session.

You must have one nvprof data file that contains the timeline information for the session. This data file should be collected by running nvprof with the --export-profile option. You can optionally enable other options such as --system-profiling on, but you should not collect any events or metrics as that will distort the timeline so that it is not representative of the applications true behavior.

You may optionally specify one or more event/metric data files that contain event and metric values for the application. These data files should be collected by running nvprof with one or both of the --events and --metrics options. To collect all the events and metrics that are needed for the analysis system, you can simply use the --analysis-metrics option along with the --kernels option to select the kernel(s) to collect events and metrics for. See [Remote Profiling](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#remote-profiling) for more information.



## Visual Profiler Views

The Visual Profiler is organized into views. Together, the views allow you to analyze and visualize the performance of your application. This section describes each view and how you use it while profiling your application.

### 2.4.1. Timeline View

The Timeline View shows CPU and GPU activity that occurred while your application was being profiled. Multiple timelines can be opened in the Visual Profiler at the same time in different tabs. The following figure shows a Timeline View for a CUDA application.

![Timeline View shows CPU and GPU activity that occurred while your application was being profiled.](https://docs.nvidia.com/cuda/profiler-users-guide/graphics/timeline-view.png)

The types of timeline rows that are displayed in the Timeline View are:

Process

A timeline will contain a **Process** row for each application profiled. The process identifier represents the pid of the process. The timeline row for a process does not contain any intervals of activity. Threads within the process are shown as children of the process.

Thread

A timeline will contain a **Thread** row for each CPU thread in the profiled application that performed either a CUDA driver or CUDA runtime API call. The thread identifier is a unique id for that CPU thread. The timeline row for a thread is does not contain any intervals of activity.

Runtime API

A timeline will contain a **Runtime API** row for each CPU thread that performs a CUDA Runtime API call. Each interval in the row represents the duration of the call on the corresponding thread.

Driver API

A timeline will contain a **Driver API** row for each CPU thread that performs a CUDA Driver API call. Each interval in the row represents the duration of the call on the corresponding thread.

Profiling Overhead

A timeline will contain a single **Profiling Overhead** row for each process. Each interval in the row represents the duration of execution of some activity required for profiling. These intervals represent activity that does not occur when the application is not being profiled.

Data Migration (DtoH)

A timeline will contain **Data Migration (DtoH)** row for each device. In the non-segment mode each interval on the timeline corresponds to one data migration from device to host.

Data Migration (DtoH)

A timeline will contain **Data Migration (HtoD)** row for each device. In the non-segment mode each interval on the timeline corresponds to one data migration from host to device.

Memcpy

A timeline will contain memory copy row(s) for each context that performs memcpys. A context may contain up to four memcpy rows for device-to-host, host-to-device, device-to-device, and peer-to-peer memory copies. Each interval in a row represents the duration of a memcpy executing on the GPU.

Compute

A timeline will contain a **Compute** row for each context that performs computation on the GPU. Each interval in a row represents the duration of a kernel on the GPU device. The **Compute** row indicates all the compute activity for the context. Sub-rows are used when concurrent kernels are executed on the context. All kernel activity, including kernels launched using CUDA Dynamic Parallelism, is shown on the Compute row. The **Kernel** rows following the Compute row show activity of each individual application kernel.

Stream

A timeline will contain a **Stream** row for each stream used by the application (including both the default stream and any application created streams). Each interval in a **Stream** row represents the duration of a memcpy or kernel execution performed on that stream.











## self-profile



将input.json中profile设置为true后需要使用如下命令运行dp train

```
sudo /home/asc22g0/gc/hov/bin/dp train input.json -l train_1.log # gcc版本为1.1.0
```

**setup.py中设置为cpu，生成timeline和TensorBoard**





**setup.py中设置为cuda，生成timeline和TensorBoard**



**IO和计算的分布**



什么操作比较耗时，优化IO和优化计算

计算放在GPU，理解IO的瓶颈





## CPU profile

通常在计算密集型（CPU intensive）的任务中 CPU time 会占据较大的比重，而在 I/O 密集型（I/O intensive）任务中 off-CPU time 会占据较大的比重。搞清楚 CPU time 和 off-CPU time 的区别对性能优化十分重要，比如某个程序的性能瓶颈在 off-CPU time 上，而我们选择了一个只观测 CPU time 的工具，那么很难找到真正的性能瓶颈，反之亦然。

我们知道了一个线程执行过程中的 CPU time 和 off-CPU time，如果要对程序的性能进行优化，这些还远远不够，我们需要进一步知道 CPU time 的时间段内，CPU 上到底发生了哪些事情、这些事情分别消耗了多少时间、在哪里导致了线程被 block 住了、分别 block 了多久等。我们需要性能观测工具来获得这些详细的信息。通常情况下我们也将称这种观测工具称为 profiler。

按照观测范围来分类，CPU 上的 profiler 大致可以分为两大类：进程级（per-process，某些地方也叫做应用级）和系统级（system wide），其中：

- 进程级只观测一个进程或线程上发生的事情
- 系统级不局限在某一个进程上，观测对象为整个系统上运行的所有程序

按照观测方法来分类，大致可以分为 event based 和 sampling based 两大类。其中：

- event based：在一个指定的 event 集合上进行，比如进入或离开某个/某些特定的函数、分配内存、异常的抛出等事件。event based profiler 在一些文章中也被称为 tracing profiler 或 tracer
- sampling based：以某一个指定的频率对运行的程序的某些信息进行采样，通常情况下采样的对象是程序的调用栈

即使确定了我们优化的对象属于上述的某一个类别，仍然有更细粒度的分类。在选择工具之前要搞清楚具体的优化对象是什么，单个 profiler 一般无法满足我们所有的需求，针对不同的优化对象 （比如 Python 线程、C/C++线程等） 我们需要使用不同的 profiler。并且，对于同一个优化对象，如果我们关注的信息不同，也可能需要使用不同的 profiler。

### 2.3 Python 进程模型

本文主要关注 Python（包括 C/C++ 拓展） 程序的优化，一个典型的 Python 和 C/C++ 拓展程序的进程如下图所示：

![img](https://pic3.zhimg.com/80/v2-7a5c9224fda9c58c68048fa1212ee21a_720w.jpg)

一个 Python 进程必须包含一个 Python 主线程，可能包含若干个 Python 子线程和若干个 C/C++ 子线程。因此我们进一步把优化对象细分为三类：

- Python 线程中的 Python 代码
- Python 线程中的 C/C++ 拓展代码
- C/C++ 线程

这里的 Python 线程具体指 CPython 解释器线程，而 C/C++ 线程指不包含 Python 调用栈的 C/C++ 线程。



### **三、profiler 的分类和选择**

我们从以下两个角度对 profiler 进行刻画:

- 是否支持 profiling time、off-CPU time 和 wall clock time（CPU time + off-CPU time）
- 是否支持 profiling C/C++ stack
- 是否能够从 CPython 解释器的调用栈中解析出 Python 调用栈



### **四、可视化工具**



#### 4.1 flamegraph

火焰图（flamegraph）是一个功能强大的可视化 profiling 结果的工具。它即可以对多种 profiler 的输出进行处理，也可以对处理后的结果进行可视化。它能够处理不同平台上的十多种 profiler 的原始输出，除了能够可视化 cpu 上的 profiling 结果，它也可以对一些内存 profiler 的输出结果进行可视化。



flamegraph 的主要功能就是显示 profiler 采样的调用栈的频率分布，图中纵向堆起来的代表调用栈，调用栈中的矩形块的宽度代表该函数运行时被采到的频率（某个执行路径的时间占比与它被采样到的概率成正比，因此采样频率近似等于该执行路径的时间占比）。



通过观察火焰图，我们可以看到程序都有哪些执行路径，以及每个执行路径的时间占比，然后对时间占比较大的性能瓶颈（或"热点"）进行优化，来达到优化性能的目的。



如果想深入了解 flamegraph，可以参考作者的主页或 github repo：

- homepage: [http://www.brendangregg.com/flamegraphs.html](https://link.zhihu.com/?target=http%3A//www.brendangregg.com/flamegraphs.html)
- github repo: [https://github.com/brendangregg/FlameGraph](https://link.zhihu.com/?target=https%3A//github.com/brendangregg/FlameGraph)



#### 4.2 speedscope



我们推荐把 speedscope 与 flamegraph 结合在一起使用：用 flamegraph 来处理不同工具的输出数据，用 speedscope 进行可视化。speedscope 是一个 web app，作者提供了一个可以直接使用的地址：[https://www.speedscope.app/](https://link.zhihu.com/?target=https%3A//www.speedscope.app/)，我们也可以在本地部署，但更前者更方便。

左上角可以选择三种模式:

- Time Order：即时间轴模式，从左到右代表时间的方向，中间每一列代表改时刻采样的调用栈

- Left Heavy：按照调用栈函数的时间占比（采样次数占比来估计时间占比）进行展示，即调用栈的每一层都按照左侧时间多右侧时间短的顺序来排序。点击任何一个调用栈中的函数：

- - 可以在图中左下角看到该函数在当前调用栈 (This Instance) 的总开销 (Total) 和自身开销 (Self)，以及该函数在所有出现过的调用栈 (All Instances) 中的总开销 (Total) 和自身开销 (Self)， 图中的整数代表被采样的次数，百分比为被采样的占比（近似等于时间占比）。
  - 图下方的白色框内是该函数的调用栈。

- Sandwich:用函数的总开销和自身开销来排序，点击函数可以看到该函数的调用者和被调用者

更详细的介绍可以参考 speedscope 的官方 repo：[https://github.com/jlfwong/spee](https://link.zhihu.com/?target=https%3A//github.com/jlfwong/speedscope)



### 五、性能观测工具



### 5.1 py-spy

#### 介绍

py-spy 是一个 sampling based profiler, 它的 profiling 的对象是 Python 及 C/C++ 拓展的调用栈。py-spy 的 overhead 中等，对运行的程序速度影响不算太大。且本身支持直接输出 speedscope 和 flamegraph 格式的结果。

repo: [https://github.com/benfred/py-s](https://link.zhihu.com/?target=https%3A//github.com/benfred/py-spy)

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



```
py-spy record --rate 10 --format speedscope -o output1.json --native dp train input.json
```

```
py-spy record --rate 10 --format -i speedscope -o output2.json -n dp train input.json
```

```
py-spy record --rate 10 --format speedscope -o output3.json -n -i -s dp train input.json
```



https://www.cnblogs.com/-wenli/p/13374186.html

https://github.com/benfred/py-spy





### 分析profile结果

```
py-spy record -o profile.svg --pid 12345
# https://github.com/brendangregg/FlameGraph
```

![image-20220220022652615](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220220022652615.png)

```
py-spy record --pid $PID --format speedscope -o profile.speedscope.json
#https://github.com/jlfwong/speedscope/blob/main/README.md
```

![image-20220220022637461](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220220022637461.png)



**直接使用pid线程号profile会有error**



**使用下面的方式error也有但是少**

```
py-spy record --rate 10 --format speedscope -o output1.json --native dp train input.json
```

```
py-spy record --rate 10 --format -i speedscope -o output2.json -n dp train input.json
```

```
py-spy record --rate 10 --format speedscope -o output3.json -n -i -s mpirun -np 4 sudo /home/asc22g0/CK/hov/bin/dp train input.json
```



**因为是采样的方式进行profile,所以显示的总时间可能会小一点,跑example的时候比例是对的,跑10000的时候比例不正确**

![image-20220223215045891](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220223215045891.png)



分析speedscope图

**火焰图就是看顶层的哪个函数占据的宽度最大。只要有"平顶"（plateaus），就表示该函数可能存在性能问题。**



可能存在的瓶颈：

![image-20220224014856397](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220224014856397.png)



![image-20220224014941712](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220224014941712.png)





为什么这里需要执行这么久

![image-20220224015120571](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220224015120571.png)

单个op的时间少，但是总共的op时间比较大



![image-20220224034824272](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220224034824272.png)



![image-20220224203239559](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220224203239559.png)





**首先理解清除程序的执行流程,CPU还是GPU,数据之间的关系**



 


























# PyTorch概述

PyTorch在Python中重新设计和实现Torch（一个用于机器学习和科学计算的模块化开源库。该库通过利用LuaJIT编译器提高性能），同时在后端代码中共享相同的核心C库。

# Cambricon PyTorch概述

Cambricon PyTorch借助PyTorch自身提供的设备扩展接口将MLU后端库中所包含的算子操作动态注册到PyTorch中，MLU后端库可处理MLU上的张量和神经网络算子的运算。Cambricon PyTorch会**基于CNML库在MLU后端实现一些常用神经网络算子**，并完成一些数据拷贝操作。

Cambricon PyTorch兼容原生PyTorch的Python编程接口和原生PyTorch网络模型，支持**在线逐层、在线融合和离线三种方式执行推理**，同时还支持在线逐层训练。网络的权重可以从pth格式文件中读取，**已支持的分类和检测网络结构由torchvision管理，可以从torchvision中读取**。

从PyTorch 1.3.0开始，寒武纪采用Python扩展包的形式对原生PyTorch进行支持。寒武纪将所有关于MLU的操作都放在了一个单独的Python包中，然后将该包导入到原生PyTorch以支持在MLU上的运算。

# MLU使用基础

Cambricon PyTorch主要使用场景为模型推理和模型训练。

推理场景追求高吞吐和低延时，将离线优化好的模型端到端部署到同一设备， 除处理输入输出数据外，尽量不打断程序的控制流、执行流和数据流。

训练场景更注重可扩展性，典型的加速手段是数据并行和模型并行， 由于要处理节点间任务调度、通讯和同步，通常要把网络拆解成细粒度的算子，无法做到端到端执行。

针对上述两种应用场景的差异，寒武纪提供了CNML和CNNL两套算子库：

- CNML通过将模型转换成计算图，利用融合操作等功能对计算图进行离线的编译优化，生成可端到端部署的推理引擎。
- CNNL为通用算子库，提供手工优化的基本算子或简单的融合算子，保证每个算子的单次运行延时尽可能低。

以下是两者应用场景的关系图。

![../_images/CNML_and_CNNL.PNG](https://www.cambricon.com/docs/pytorch/_images/CNML_and_CNNL.PNG)

*Cambricon CNML和CNNl关系图*

## 基础使用

### 导入torch_mlu相关包

```python
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import torchvision.models as models
```

### 设置MLU基本信息

```python
ct.set_core_number(args.core_number)                 # 设置MLU core number
ct.set_core_version(args.mcore)                      # 设置MLU core version
ct.set_input_format(args.input_format)               # 设置输入图片的通道顺序，以决定首层卷积对三通道输入的补齐通道顺序。默认是RGBA顺序。
```

以上代码为推理的基本参数设置，例如推理的核数、推理核版本、输入数据模式。该部分只适用于推理模式。

### 加载模型

Cambricon PyTorch模型在torchvision库中均已提供。使用时通过torchvision导入模型，具体命令如下：

```python
torch.set_grad_enabled(False) #注意：在运行MLU推理融合模式时，这个条件是必须要设置的。
net = getattr(models.quantization, 'net_name')(pretrained=True, quantize=True)
```

其中， `models.quantization` 为torchvision导入后的模型；`getattr(models.quantization,net_name)` 将返回指定的网络对象；`pretrained=True` 将在初始化对象过程中同时加载权重；`quantize=True` 表示加载量化的权重。

# 推理快速入门

PyTorch使用由多层互连计算单元组成的神经网络（模型）进行模型推理。PyTorch提供了设计好的模块和类便于快速创建网络模型，例如 `torch.nn` 类。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlu
import torch_mlu.core.mlu_model as ct  # 导入必要模块
import torch_mlu.core.mlu_quantize as mlu_quantize
import torchvision.models as models
ct.set_core_number(1)
ct.set_core_version("MLU270")
torch.set_grad_enabled(False)

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2)
      x = self.dropout1(x)
      x = torch.flatten(x, 1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)
      output = F.log_softmax(x, dim=1)
      return output

net=Net().eval()
torch.save(net.state_dict(), 'test_weights.pth')  # 保存随机初始化的权重，此处命名为 test_weights.pth。
input_data = torch.rand((1,1,28,28), dtype=torch.float) 

# mean 和 std 是对输入图片进行预处理的数组值，表示对输入图片的每个通道处理的均值和标准值。MNIST网络使用的数据是一通道的，因此只需传入一个通道的值。如果是3个通道的输入，可设置为 mean = [0,0,0], std = [1/255, 1/255, 1/255]。
mean = [0]
std = [1/255]


net.load_state_dict(torch.load('test_weights.pth', map_location='cpu'), False) # 将保存的test_weights.pth加载，将其中的参数加载进net中，然后使用 quantize_dynamic_mlu 接口对net进行模型量化，并保存量化后的模型
net_quantization = mlu_quantize.quantize_dynamic_mlu(net, {'mean':mean, 'std':std, 'firstconv':True}, dtype='int8', gen_quant=True) # 模型权重量化  使用 quantize_dynamic_mlu 接口替换网络中需要量化的算子为Cambricon PyTorch对应的自定义算子。
output = net_quantization(input_data)  # cpu运行量化模型的输出结果
torch.save(net_quantization.state_dict(), 'test_quantization.pth') # 保存权重数据

net_quantization.load_state_dict(torch.load('test_quantization.pth')) # 加载量化权重。
#print("step1")
net_mlu = net_quantization.to(ct.mlu_device()) # 将模型和输入数据拷贝到MLU上
#print("step2")
input_mlu = input_data.to(ct.mlu_device())
#print("step3")
output=net_mlu(input_mlu) 
print("step4")
print(output.cpu())
print("step5")

```

# 框架移植概述

为了使Cambricon PyTorch更好地在寒武纪硬件平台上运行，寒武纪对原生PyTorch做了进一步拓展，如下图所示。Cambricon PyTorch对原生PyTorch的主要修改有：添加MLU设备、实现MLU特有的在线融合模式、部分算子的分发方式、torchvison访问权限和Catch拓展包。

![../_images/catch_sf_framework.PNG](https://www.cambricon.com/docs/pytorch/_images/catch_sf_framework.PNG)

*Cambricon PyTorch 框架结构图*

# 自定义在线逐层算子

## 添加逐层算子

PyTorch逐层模式中算子间数据传递和存储的基本单元是tensor。PyTorch根据tensor中的device属性值将算子分发到不同设备。

Catch通过注册添加MLU算子方式与PyTorch源码解耦。

执行以下步骤在Catch中添加MLU算子：

1. 声明算子。

   在 `catch/torch_mlu/tools/mlu_functions.yaml` 中声明算子。

   ```
   - name: add # 算子名称
     use_mlu_dispatcher: unboxed_only # 分发类型，包括标准算子（unboxed_only）和客制化算子（custom）
     derived_type: cnml # 派生类型（cnml/cnnl/cnml&&cnnl）
     schema_string: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor # 用于算子注册的函数签名
     arguments: # 参数
     - name: self # 参数名称
       type: const at::Tensor & # 参数类型
     - name: other
       type: const at::Tensor &
     - name: alpha
       type: at::Scalar
     return_type: at::Tensor # 函数返回类型
   ```

2. 添加OpMethods基类中的CPU实现。

Catch模块中包含AtenMluType标准算子类型和AtenMluCustomType客制化算子类型，AtenMluType和AtenMluCustomType会通过OpMethods下发到推理算子或训练算子。 根据模板生成的op_methods.h算子声明，在 `catch/torch_mlu/csrc/aten/operators/op_methods.cpp` 中添加算子的CPU实现。

```
op_methods.h
virtual at::Tensor add(const at::Tensor& self, const at::Tensor& other, at::Scalar alpha);
```

```
op_methods.cpp
at::Tensor OpMethods::add(const at::Tensor& self,
                          const at::Tensor& other,
                          at::Scalar alpha) {
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output = at::add(input_cpu, other_cpu, alpha);
  return output.to(at::Device(at::Device::Type::MLU));
}
```

3. 添加wrapper。

wrapper是对算子kernel的封装，每个算子对应一个wrapper。本文分别以add和div算子为例，添加wrapper如下所示：

- CNML算子

  ```
  cnml_kernel.h
  at::Tensor cnml_add(const at::Tensor& input, const at::Tensor& other,
                      at::Scalar alpha);
  ```

  ```
  add.cpp
  at::Tensor cnml_add(const at::Tensor& input, const at::Tensor& other,
                      at::Scalar alpha_scalar) {
    TORCH_CHECK(input.dim() >= 0 || other.dim() >= 0, "dimension not support");
    at::Tensor input_ = input;
    at::Tensor other_ = other;
    auto alpha_data = alpha_scalar.to<scalar_t>();
    if (alpha_data != 1) {
      // scale_t
      other_ = cnml::ops::cnml_scale(other_, alpha_data, 0);
    }
    if (other_.dim() < 1 && other_.device().type() == c10::DeviceType::CPU) {
      auto other_scalar = other_.item();
      return cnml_add_internal(input_, other_scalar);  //调用kernel
    }
    if (input_.dim() < 1 && input_.device().type() == c10::DeviceType::CPU) {
      auto input_scalar = input_.item();
      return cnml_add_internal(other_, input_scalar);  //调用kernel
    }
  
    bool broadcast = input_.sizes() != other_.sizes();
    if (broadcast) {
      auto broadcast_size = at::infer_size(input.sizes(), other.sizes());
      at::Tensor broadcast1 = cnml::ops::cnml_expand(input_, broadcast_size, false);
      at::Tensor broadcast2 = cnml::ops::cnml_expand(other_, broadcast_size, false);
      return cnml_add_internal(broadcast1, broadcast2);  //调用kernel
    } else {
      return cnml_add_internal(input_, other_);  //调用kernel
    }
    return cnml_add_internal(input_, other_);  //调用kernel
  }
  ```

- CNNL算子

  ```
  cnnl_kernel.h
  at::Tensor cnnl_div(const at::Tensor& input, const at::Tensor& other);
  ```

  ```
  div.cpp
  at::Tensor cnnl_div(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor input_new, other_new;
    bool input_is_scalar = false, other_is_scalar = false;
    if (self.dim() == 0 && other.dim() == 0) {
      auto self_t = self.cpu();
      auto other_t = other.cpu();
      auto output = at::div(self_t, other_t);
      return output.to(at::Device(at::Device::Type::MLU));
    } else if (other.dim() == 0) {
      // self is Tensor, other is Scalar
      auto other_tensor = at::native::full(
          self.sizes(), other.item(), self.options().device(at::kCPU));
      other_new = other_tensor.to(at::Device(at::Device::Type::MLU));
      other_is_scalar = true;
    }
    input_new = input_is_scalar ? input_new : self;
    other_new = other_is_scalar ? other_new : other;
    return cnnl_div_internal(input_new, other_new); //调用kernel
  }
  ```

4. 添加kernel。

Wrapper中通过调用kernel实现算子功能。示例中分别为CNML库的add算子和CNNL库div算子。 算子的具体实现主要通过调用CNML和CNNL库的接口来完成。下面简要介绍一下两个库的逻辑。

- CNML库Kernel

![../_images/programming_model.png](https://www.cambricon.com/docs/pytorch/_images/programming_model.png)

*CNML编程逻辑*

- CNML kernel实现就是按照上述编程逻辑调用CNML库接口完成的， 在 `catch/torch_mlu/csrc/aten/operators/cnml/internal/cnml_internal.h` 和 `catch/torch_mlu/csrc/aten/operators/cnml/internal/add_internal.cpp` 中添加kernel函数的声明和实现。

  ```
  cnml_internal.h
  at::Tensor cnml_add_internal(const at::Tensor& input1, const at::Tensor& input2);
  ```

  ```
  add_internal.cpp
  at::Tensor cnml_add_internal(const at::Tensor& input1, const at::Tensor& input2) {
    auto output = at::native::empty_like(input1);
    // prepare input cnml tensor
    auto* input1_impl = getMluTensorImpl(input1);  // 获取MluTensorImpl
    auto input1_cnml = input1_impl->CreateCnmlTensor(
        CNML_TENSOR, toCnmlDataType(input1.dtype()));  // 类型自适应：toCnmlDataType()
  
    auto* input2_impl = getMluTensorImpl(input2);
    auto input2_cnml = input2_impl->CreateCnmlTensor(
        CNML_TENSOR, toCnmlDataType(input2.dtype()));
  
    // prepare output cnml tensor
    auto* output_impl = getMluTensorImpl(output);
    auto output_cnml = output_impl->CreateCnmlTensor(
        CNML_TENSOR, toCnmlDataType(output.dtype()));
  
    // End the execution flow if not MLU device
    CHECK_MLU_DEVICE(output);
  
    // setup operator
    cnmlBaseOp_t add_op;
    TORCH_CNML_CHECK(cnmlCreateAddOp(&add_op, input1_cnml, input2_cnml, output_cnml));
  
    // return to JIT if running mode is fuse
    CHECK_RETURN_TO_FUSE(add_op, output);
  
    // compile op
    TORCH_CNML_CHECK(cnmlCompileBaseOp(add_op, GET_CORE_VERSION, GET_CORE_NUMBER));
  
    auto queue = getCurQueue();
    TORCH_CNML_CHECK(cnmlComputeAddOpForward_V4(add_op,
                                                NULL,
                                                input1_impl->raw_mutable_data(),
                                                NULL,
                                                input2_impl->raw_mutable_data(),
                                                NULL,
                                                output_impl->raw_mutable_data(),
                                                queue,
                                                NULL));
    syncQueue(queue);
    TORCH_CNML_CHECK(cnmlDestroyBaseOp(&add_op));
  
    return output;
  }
  ```

  - CNNL库kernel

  CNNL库的kernel与CNML库有所不同，无需经过创建、编译、执行等多个步骤，使用较简单，但不支持融合操作。 在 `catch/torch_mlu/csrc/aten/operators/cnnl/internal/cnnl_internal.h` 和 `catch/torch_mlu/csrc/aten/operators/cnnl/internal/div_internal.cpp` 中添加kernel函数的声明和实现。

  ```
  cnnl_internal.h
  at::Tensor cnnl_div_internal(const at::Tensor& self, const at::Tensor& other);
  ```

  ```
  div_internal.cpp
  at::Tensor cnnl_div_internal(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor input_new = self;
    at::Tensor other_new = other;
    auto output = at::empty_like(self);
  
    auto input_impl = getMluTensorImpl(input_new);
    auto other_impl = getMluTensorImpl(other_new);
    auto output_impl = getMluTensorImpl(output);
  
    // get current handle
    auto handle = getCurrentHandle();
    auto queue = getCurQueue();
    CnnlTensorDescriptor desc_input;
    CnnlTensorDescriptor desc_other;
    CnnlTensorDescriptor desc_output;
  
    // get cnnl descriptor
    desc_input.set(input_new);
    desc_other.set(other_new);
    desc_output.set(output);
  
    // malloc mlu memory
    auto input_ptr = input_impl->cnnlMalloc();
    auto other_ptr = other_impl->cnnlMalloc();
    auto output_ptr = output_impl->cnnlMalloc();
  
    // workspace
    size_t workspace_size = 0;
    TORCH_CNNL_CHECK(cnnlGetDivWorkspaceSize(handle,
                          desc_input.desc(),
                          desc_other.desc(),
                          desc_output.desc(),
                          &workspace_size));
    std::vector<int64_t> space_shape;
    workspace_size /= input_impl->itemsize();
    at::Tensor temp =
        at::empty({static_cast<long int>(workspace_size)}, self.options());
    auto* temp_impl = getMluTensorImpl(temp);
    auto temp_ptr = temp_impl->cnnlMalloc();
  
  
    // set descriptor config
    TORCH_CNNL_CHECK(cnnlDiv(handle, desc_input.desc(), input_ptr,
                             desc_other.desc(), other_ptr,
                             temp_ptr, desc_output.desc(), output_ptr));
    TORCH_CNRT_CHECK(cnrtSyncQueue(queue));
    return output;
  }
  ```

### 对MLU不支持算子的处理

对于MLU暂不支持的算子，如果未在 `catch/torch_mlu/tools/mlu_functions.yaml` 中声明，程序会直接终止，并抛出无法分发到MLU设备的异常；如果已经在 `catch/torch_mlu/tools/mlu_functions.yaml` 中声明但在运行至wrapper或kernel时失败，输入数据将会拷贝到CPU上，然后调用CPU相关算子，使其在CPU上运行，最后再将输出结果拷回到MLU上。

## 算子测试

使用基于Python的unittest模块编写算子单元测试。测试时需提供相同的参数和输入数据，分别在MLU和CPU上执行算子，对比两者的输出结果。MLU和CPU计算结果因为量化等原因可能会产生差异，由于CNML和CNNL的使用场景不同，分别维护了两套单算子测试，一般情况下两者的相对误差在以下范围内均是可以接受的：CNML库单算子误差为2%以内，CNNL库单算子误差在0.3%以内（训练场景对算子精度要求较高）。

# 自定义在线融合算子

## 融合机制

Cambricon PyTorch利用JIT模块实现融合模式。JIT （Just-In-Time）是一组编译工具，用于弥补PyTorch科研与生成部署之间的差距。它**允许创建不依赖Python解释器而运行的模型，也可以进行如算子融合、冗余算子去除等优化**。 

在JIT的trace方法中，首先会对整个网络运行一遍逐层模式，同时构建一个静态图；然后对静态图进行优化（包括去除冗余算子、小算子融合为融合算子、数据块复用等）得到一个优化后的静态图；之后会根据输入数据的设备类型进行基于设备的优化，生成针对当前设备的指令。优化后的指令会缓存进一个键值map中。当下次输入数据进行计算前会进行匹配，命中则直接取出指令运行。****

### 融合模式实现方案

JIT的运行流程跟CNML中的融合模式大致相同。在CNML中，需要将所有CNML支持的算子构成一个融合子图，设置融合子图的输入和输出。针对整个融合子图生成一套指令，指令只生成一遍，之后便不再编译该子图，而是直接运行指令。利用融合子图和指令优化以实现运算加速。

类似地，PyTorch 1.3.0的JIT提供了用户自定义图优化方式和融合算子用户自定义注册方式，Cambricon PyTorch使用这些功能将MLU融合代码存放在Catch库中。



# 任务调度

为同时满足MLU端串行程序运行和并行程序运行，寒武纪提供了Queue功能，可以将计算任务或内存拷贝任务下发到特定的Queue运行。

Queue的核心思想如下：

1. 任务下发到Queue后异步执行。
2. 同一个Queue内的任务按下发先后顺序串行执行。
3. 不同Queue间的任务并行执行。

可以将需要串行执行的任务下发至同一个Queue，将需要并行执行的任务下发到不同的Queue。单个Queue内的任务将按照创建顺序执行，不同的Queue会按照相对顺序并发执行。

### 基本用法

以下为代码示例：

```python
import torch_mlu.core.mlu_model as ct
ct.set_device(0)

// 当前设备0卡使用默认Queue计算
x = torch.randn((64, 128, 24, 24), dtype=torch.float32)
x_mlu = x.to(torch.device('mlu'))
out = torch.abs(x_mlu)
...

// 设定MLU 1卡的Queue进行计算
with torch_mlu.core.mlu_model.Queue(1):
    x = torch.randn((64, 128, 24, 24), dtype=torch.float32)
    x_mlu = x.to(torch.device('mlu'))
    out = torch.abs(x_mlu)
...

// 设定MLU 2卡的Queue进行计算
with torch_mlu.core.mlu_model.Queue(0):
    x = torch.randn((64, 128, 24, 24), dtype=torch.float32)
    x_mlu = x.to(torch.device('mlu'))
    out = torch.abs(x_mlu)
...
```

# 内存管理

Cambricon PyTorch支持对MLU内存的管理，通过对MLU上的内存管理提升性能。Cambricon PyTorch提供了一系列Python API，通过接口调用对正在使用/已经缓存的内存的检测以及释放缓存的操作。

调用以下接口返回当前正在使用的MLU内存。device_index默认值为-1，表示当前使用设备id。

```
torch_mlu.core.mlu_model.memory_allocated(int device_index)
```

## 锁页内存

Cambricon PyTorch提供锁页内存对数据拷贝到设备上进行加速。Cambricon PyTorch分别支持对tensor和DataLoader锁页内存申请，接口和使用方式与原生PyTorch完全一致，当编译并导入catch之后就可以使用。

下面的用例展示了如何开启DataLoader的锁页功能并使用DataLoader导入数据：

```python
import torch
import torch_mlu
from torch.utils.data import DataLoader
import torch_mlu.core.mlu_model as ct

train_loader = DataLoader(
               train_dataset, # torch.utils.data.ImageFolder
               batch_size=16,
               shuffle=None,
               sampler=None,
               num_workers=2, # 进程数
               pin_memory=True) # 开启锁页内存

for i, (images, target) in enumerate(train_loader):
    images = Variable(images.float(), requires_grad=False)
    images = images.to(ct.mlu_device())
```

# 配置环境变量

使用以下命令进行环境变量配置。

**DEBUG**

该环境变量用来设置debug模式。设置为1表示开启debug模式，设置为0表示禁止debug模式。

```
export DEBUG=0
```

**USE_OPENCV**

该环境变量用来设置是否使用OPENCV。设置为1表示使用opencv，设置为0表示不使用。

**MAX_JOBS**

该环境变量用来设置编译最大Job数。

**USE_CUDA**

该环境变量用来设置是否使用CUDA。设置为0表示不使用CUDA，设置为1表示使用CUDA。

**DISABLE_MLU_FUSION**

该环境变量用来设置是否使能MLU图分段逻辑。设置为0表示使能MLU图分段逻辑，设置为1表示禁止MLU图分段逻辑。

# 模型推理

# Python API使用

Cambricon PyTorch 以不改变原有PyTorch接口行为为原则，在Catch扩展包中添加MLU设备以及MLU算子、在原生PyTorch上打补丁，实现了PyTorch的大部分特性。

Cambricon PyTorch支持CPU和MLU设备类型。可以**使用to方法将CPU设备上的tensor以及nn.Module转为MLU对象**

Cambricon PyTorch支持将输入tensor设置成HalfTensor或FloatTensor。Half模式仅支持MLU和CUDA，不支持CPU。建议使用HalfTensor输入，因为在MLU上Half类型的输入tensor运算比Float类型输入tensor运算性能更优。**目前每个网络的demo程序中都添加了一个half_input参数，用于将输入tensor可以设置成HalfTensor或FloatTensor。**

以加法算子为例，使用Python API运行如下：

```
import torch
import torch_mlu.core.mlu_model as ct
a = torch.rand((1,3,224,224), dtype=torch.float)
b = torch.rand((1,3,224,224), dtype=torch.float)
out_mlu = a.to(ct.mlu_device()) + b.to(ct.mlu_device())
out_cpu = out_mlu.cpu()
```

以上代码展示了如何使用MLU设备完成加法运算，`a.to(ct.mlu_device())` 操作实际上是将a转移至MLU。

# 在线推理

## 在线逐层和融合推理

在线推理指使用原生PyTorch提供的Python API直接运行网络。

在线推理包括逐层模式和融合模式两种。

逐层模式使用Python API逐个调用每个算子时，每个MLU算子都在底层的C++接口中经过创建句柄、编译指令、拷贝输入、前向计算、拷贝输出等过程。逐层模式便于观察数据的流动和变化，但是效率较低。

融合模式将所有算子作为一个fusion算子，只对fusion算子执行编译指令过程，减少了小算子之间的 数据拷贝（不仅是主从设备间，还包括RAM和DDR之间的拷贝），极大地提高了效率。使用JIT模式只需对整个网络进行 一次编译，避免了多次编译产生的开销。

# 离线推理

离线推理指序列化已编译好的算子到离线文件，生成离线模型。离线模型不依赖于PyTorch框架，只基于CNRT（Cambricon Neuware Runtime Library，寒武纪运行时库）单独运行。离线模型为.cambricon文件，生成离线模型可使用Cambricon PyTorch的Python接口将模型转为离线模型。

对于离线模型的多核设置，使用genoff.py脚本设置 `batch_size` 、 `core_number` 和 `input_format` 参数，CNRT会在推理时自动分配最优的软硬件资源。要编译示例程序，请执行 `./scripts/build_offline.sh`。

在多核离线示例程序运行中可设置以下参数：

- -batch_size：指定batch size。此处的batch_size表示每次分发给MLU的样本数量，MLU会自动分配到所使用的每个核上。
- -core_number：设置运行网络时用到的MLU核数。
- -input_format：指定运行网络时输入图片的通道顺序。

`batch_size` 和 `core_number` 参数只针对多核示例程序，在多核示例执行时添加参数运行。

## 检测网络YOLOv3示例

yolov3_offline_multicore是检测网络YOLOv3的示例为离线多核示例程序。以下为运行命令和参数详解：

```
../build/yolov3/yolov3_offline_multicore -offlinemodel yolov3.cambricon -dataset_path $COCO_PATH_PYTORCH/COCO -images file_list_for_release -labels label_map_coco.txt -simple_compile 1 -input_format 0
```

本离线示例的参数详解如下：

- -offlinemodel：离线模型的文件名称。
- -dataset_path：数据集的路径。
- -images：一个文本文件的名称，该文件中应当包含若干行，每行都是一张需要检测图片的路径。
- -labels：检测的分类标记文件。
- -simple_compile：使能简单编译功能，CNRT根据离线模型core_number和batch_size参数自动分配最优的软硬件资源。
- -input_format：指定运行网络时输入图片的通道顺序，依据该参数对输入图片进行相应的前处理。支持RGB、RGBA和ARGB三种输入图片的通道顺序。0表示运行网络时输入图片的通道顺序为RGB；1表示运行网络时输入图片的通道顺序为RGBA；2表示运行网络时输入图片的通道顺序为ARGB。

# 多核示例

多核编程模型分为在线编程和离线编程两部分介绍。但需要注意的是，core_num仅有三种数值可选：1、4、16。

## 在线多核示例




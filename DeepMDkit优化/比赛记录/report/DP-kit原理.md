

## DeepMD-kit操作探索

启动需要使用这个命令

```shell
export CUDA_HOME=/usr/local/cuda-10.1/
source activate /home/asc22g0/GC/env
```

```bash
# 常用操作
dp train input.json

dp freeze 

dp test -m frozen_model.pb -s ./data/ -n 200
```



### 输出分析



```
head lcurve.out 
#输出结果误差的变化情况
#  step      rmse_val    rmse_trn    rmse_e_val  rmse_e_trn    rmse_f_val  rmse_f_trn         lr
      0      2.65e+01    2.76e+01      6.77e-01    6.79e-01      8.38e-01    8.71e-01    1.0e-03
    100      1.00e+01    9.86e+00      2.62e-01    2.53e-01      3.17e-01    3.11e-01    1.0e-03
    200      6.61e+00    6.83e+00      9.79e-02    9.06e-02      2.09e-01    2.16e-01    1.0e-03
    300      5.93e+00    5.74e+00      2.10e-02    1.91e-02      1.88e-01    1.81e-01    1.0e-03
    400      6.82e+00    6.73e+00      1.78e-01    1.80e-01      2.15e-01    2.13e-01    1.0e-03
    500      5.44e+00    5.26e+00      4.11e-02    4.33e-02      1.72e-01    1.66e-01    1.0e-03
    600      5.06e+00    5.58e+00      5.22e-02    5.26e-02      1.60e-01    1.76e-01    1.0e-03
    700      5.33e+00    5.23e+00      1.96e-02    1.76e-02      1.69e-01    1.65e-01    1.0e-03
    800      4.64e+00    4.29e+00      5.06e-02    5.19e-02      1.47e-01    1.36e-01    1.0e-03

```



使用dp test命令的输出

```
DEEPMD INFO    Adjust batch size from 16384 to 32768
DEEPMD INFO    # number of test data : 200 
DEEPMD INFO    Energy RMSE        : 6.616563e-01 eV
DEEPMD INFO    Energy RMSE/Natoms : 3.446127e-03 eV
DEEPMD INFO    Force  RMSE        : 1.070423e-01 eV/A
DEEPMD INFO    Virial RMSE        : 7.999571e+00 eV
DEEPMD INFO    Virial RMSE/Natoms : 4.166443e-02 eV
```





## DP-kit项目结构

### example文件夹

![image-20220130194846421](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130194846421.png)

![image-20220130194908979](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130194908979.png)

data文件夹中为数据集,使用dp test,数据信息和测试信息都在这里面	

![image-20220130195027443](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130195027443.png)









## DP-kit基本原理



DP与DFT和经典力场相比，采用神经网络的方式进行拟合势函数



输入为第一性原理计算得到的高精度数据

![image-20220129111228096](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220129111228096.png)



拟合结果需要保持可扩展性：更大规模体系，上图原子中心框架可以说明，体系能量为局部能量的总和



输入不能为原子坐标，应该为符合可扩展性的输入符，对于人类是常识，但对于神经网络可能会产生不一样的输出结果，下图介绍了几种常用的描述符，DeepMD描述符是专门为神经网络设计的

![image-20220129111625005](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220129111625005.png)



描述符的作用是描述周围的环境，如何将第i个原子的周围环境描述出来

![image-20220129112032720](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220129112032720.png)





























input.json:

```json
{
    "_comment": " model parameters",
    "model": {
        "type_map":     ["O", "H"],//原子对应	
        "descriptor" :{
            "type":             "se_e2_a",  //采用环境矩阵的多少列，a代表使用全部的列
            "sel":              [46, 92],  //决定输入，邻域内最多考虑的原子个数，保持输入的一致性，不要调太大
            "rcut_smth":        0.50,    //平滑处理的参数
            "rcut":             6.00,    //考虑的最远距离，单位为0.1纳米
            "neuron":           [25, 50, 100],
            "resnet_dt":        false,//残差处理是否需要乘于可学习的参数
            "axis_neuron":      16,  //M2，得到description时的参数，M1一般比较大，M2取16
            "seed":             1,
            "precision":        float64, //设置结果精度为双精度float64或者单精度float32
            "_comment":         " that's all"
        },
        "fitting_net" : {
            "neuron":           [240, 240, 240],
            "resnet_dt":        true,
            "seed":             1,
            "_comment":         " that's all"
        },
        "_comment":     " that's all"
    },

    "learning_rate" :{
        "type":         "exp",//参数变化方式，这里是指数衰减，步长为5000
        "decay_steps":  5000,
        "start_lr":     0.001, //初始学习率
        "stop_lr":      3.51e-8,//终止学习率
        "_comment":     "that's all"
    },

    "loss" :{
        "type":         "ener",//
        "start_pref_e": 0.02,//能量
        "limit_pref_e": 1,
        "start_pref_f": 1000,//受力，为了防止过拟合，所以设置大一些
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
        "_comment":     " that's all"
    },

    "training" : {
        "training_data": {
            "systems":          ["../data/data_0/", "../data/data_1/", "../data/data_2/"],
            "batch_size":       "auto",
            "_comment":         "that's all"
        },
        "validation_data":{
            "systems":          ["../data/data_3"],
            "batch_size":       1,
            "numb_btch":        3,
            "_comment":         "that's all"
        },
        "numb_steps":   1000000,
        "seed":         10,
        "disp_file":    "lcurve.out",
        "disp_freq":    100,
        "save_freq":    1000,
        "_comment":     "that's all"
    },

    "_comment":         "that's all"
}
 

```





### 使用dp train input.json的输出分析

```
OMP: Info #249: KMP_AFFINITY: pid 36665 tid 38808 thread 1608 bound to OS proc set 8
//这段话表示线程分配
```







### 数据结构

![image-20220129172450529](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220129172450529.png)

粗体标出的为必须的数据

type.raw对应于每一个原子的坐标形式



数据处理方式dpdata

![image-20220129172930552](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220129172930552.png)





![image-20220130002902949](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130002902949.png)





![image-20220130003350740](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130003350740.png)

![image-20220130003535020](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130003535020.png)



![image-20220130011235193](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130011235193.png)



![image-20220130011355459](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130011355459.png)

![image-20220130011418982](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130011418982.png)





![image-20220130013035512](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130013035512.png)

![image-20220130013704274](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20220130013704274.png)

 




#!/bin/bash

#提交测试时，进入虚拟环境是必备步骤
cd /workspace/volume/private/sdk/cambricon_pytorch/
source env_pytorch.sh
cd -

# 你的run脚本需要按照此脚本的格式编写，如果你的脚本格式不符合此模板，或者打印信息的格式不符合要求，可能不会采纳你的成绩。

CURRENT_DIR=$(dirname $(readlink -f $0))

# 在info部分你需要打印出你所使用的模型、数据集所在的路径，以及精度计算以及性能统计的代码位置。
print_info(){
    echo "------------------------------- INFO --------------------------------"
    echo "|  dataset path: "
    echo "|     $CURRENT_DIR/data/coco"
    echo "|"
    echo "|  Online:"
    echo "|     online model path:"
    echo "|         $CURRENT_DIR/model/online/yolov3.pth,"
    echo "|         $CURRENT_DIR/model/online/yolov3_int8.pth"
    echo "|     performance test code position: "
    echo "|         $CURRENT_DIR/online/yolov3/test.py"
    echo "|"
    echo "|  Offline:"
    echo "|     offline model path: "
    echo "|         $CURRENT_DIR/model/offline/yolov3.cambricon"
    echo "|     performance test code position: "
    echo "|         $CURRENT_DIR/offline/yolov3/yolov3_offline_multicore.cpp."
    echo "|"
}

# 在run部分完成你的网络的在线和离线模式运行，你可以替换其中运行代码的部分。
run(){
    echo "------------------------------- RUN ---------------------------------"
    # 你的运行脚本部分，根据具体情况进行修改。
    echo "Running online."
    pushd $CURRENT_DIR/../online/yolov3
    # 在线模式测试性能
    bash run_online_performance.sh
    popd
    echo "Running offline."
    pushd $CURRENT_DIR/../offline/yolov3
    # 生成性能测试离线模型
    bash run_get_performance_offlinemodel.sh
    # 离线模式测试性能
    bash run_offline_performance.sh
    popd
}

do_run() {
    print_info
    run
}

do_run 2>&1 |tee 00_Yolov3_example_performance_result

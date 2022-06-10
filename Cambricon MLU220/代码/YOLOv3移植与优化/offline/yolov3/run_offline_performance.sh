#!/bin/bash
# Parameter1 non-simple_mode: 0; simple_mode:1
CURRENT_DIR=$(dirname $(readlink -f $0))
EXAMPLE_DIR=$CURRENT_DIR/../build/yolov3/src
OFFLINE_DIR=${CURRENT_DIR}/..

# configure target network_list
network_list=(
    yolov3
)

# select input format rgba argb bgra or abgr
# 0-rgba 1-argb 2-bgra 3-abgr
channel_order='0'

# configure batch_size and core_number for mlu270
bscn_list_MLU270=(
  '16 16'
)

file_list=$CURRENT_DIR/data/file_list_performance

# including: generate *.cambricon model and run *.cambricon model in MLU
do_run()
{
    echo "----------------------"
    echo "batch_size: $batch_size, core_number: $core_number, channel_order: $channel_order"
    # first remove any offline model
    /bin/rm *.cambricon* &> /dev/null

    log_file="${network}_multicore_thread.log"

    echo > $CURRENT_DIR/$log_file

    # run offline.cambricon model command
    run_cmd="${EXAMPLE_DIR}/yolov3_offline_multicore$SUFFIX -offlinemodel $CURRENT_DIR/../../model/offline/${network}.cambricon -images $file_list -labels $CURRENT_DIR/label_map_coco.txt -outputdir /home/tmp -dump 1 -simple_compile 1 -dataset_path /workspace/dataset/public/zhumeng-dataset/coco_2014/ -perf_mode 1 -perf_mode_img_num 496  -input_format $channel_order &>> $CURRENT_DIR/$log_file"

    check_cmd="python $OFFLINE_DIR/scripts/meanAP_COCO.py --file_list $file_list --result_dir /home/tmp --ann_dir /workspace/dataset/public/zhumeng-dataset/coco_2014/ --data_type val2014 &>> $CURRENT_DIR/$log_file"

    echo "run_cmd: $run_cmd" &>> $CURRENT_DIR/$log_file
    echo "check_cmd: $check_cmd" &>> $CURRENT_DIR/$log_file

    echo "running offline test..."
    eval "$run_cmd"
#    tail -n 3  $CURRENT_DIR/$log_file
    grep "yolov3_detection() execution time:" -A 2 $CURRENT_DIR/$log_file
    tail -n 5 $CURRENT_DIR/$log_file
}

clean(){
    /bin/rm /home/tmp/COCO*.txt &> /dev/null
    /bin/rm /home/tmp/yolov3*.jpg &> /dev/null
}

clean

/bin/rm *.log &> /dev/null

for network in "${network_list[@]}"; do
    echo -e "\n===================================================="
    echo "running ${network} offline multiple core ..."
    for bscn in "${bscn_list_MLU270[@]}"; do
      batch_size=${bscn:0:2}
      core_number=${bscn:3:2}
      do_run
    done
done
clean

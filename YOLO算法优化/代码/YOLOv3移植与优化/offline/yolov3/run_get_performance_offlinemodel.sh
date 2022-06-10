#生成离线模型
cd ../../online/yolov3/
python test.py --mlu true --jit true --batch_size 16 --core_number 16 --image_number 16 --half_input 1 --quantized_mode 1 --quantization false --input_channel_order 0 --compute_map false --save_offline_model true --run_mode false
cd -

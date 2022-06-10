#在线融合推理精度测试
python test.py --mlu true --jit true --batch_size 1 --core_number 1 --image_number 5000 --half_input 1 --quantized_mode 1 --quantization false --input_channel_order 0 --compute_map true --run_mode false

#在线融合推理性能测试
python test.py --mlu true --jit true --batch_size 16 --core_number 16 --image_number 496 --half_input 1 --quantized_mode 1 --quantization false --input_channel_order 0 --compute_map false --run_mode true

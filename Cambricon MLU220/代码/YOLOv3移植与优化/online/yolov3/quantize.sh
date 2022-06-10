#模型量化
python test.py --mlu false --jit false --batch_size 1 --core_number 1 --image_number 1 --half_input 1 --quantized_mode 1 --quantization true --input_channel_order 0 --quantized_model_path ../../model/online

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.yolo import Model
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import numpy as np
import os
import logging 

import tqdm
from datetime import datetime, timedelta
import json
from PIL import Image

from random import sample
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.general import coco80_to_coco91_class


from utils.parse_config import *


logging.basicConfig(level=logging.INFO, format= '%(levelname)s: %(message)s')
logger = logging.getLogger("__name__")



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_boxes(prediction, batch_size=1, img_size=640):
    '''
    Retirms detectopm with shape:
	(x1,y1,x2,y2,objects_conf,class_score,class_pred)
    '''
    reshape_value = torch.reshape(prediction, (-1,1))
    num_boxes_final = reshape_value[0].item()
    print('num_boxes_final: ', num_boxes_final)
    all_list = [[] for _ in range(batch_size)]
    for i in range(int(num_boxes_final)):
        batch_idx = int(reshape_value[64+i*7+0].item())
        if batch_idx >= 0 and batch_idx < batch_size :
            bl = reshape_value[64+i*7+3].item()
            br = reshape_value[64+i*7+4].item()
            bt = reshape_value[64+i*7+5].item()
            bb = reshape_value[64+i*7+6].item()
            if bt - bl > 0 and bb - br > 0:
                all_list[batch_idx].append(bl)
                all_list[batch_idx].append(br)
                all_list[batch_idx].append(bt)
                all_list[batch_idx].append(bb)
                all_list[batch_idx].append(reshape_value[64+i*7+2].item())
                all_list[batch_idx].append(reshape_value[64+i*7+1].item())
        outputs = [torch.FloatTensor(all_list[i]).reshape(-1,6) for i in range(batch_size)]
        return outputs

def preprocess(imgs, img_size, normalize=False):
    input_imgs = np.array([]).reshape(-1,3,img_size,img_size)
    img_shape = (img_size, img_size)

    for img in imgs:
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        if normalize:
            input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        else:
            input_img = np.pad(img, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        if normalize:
            input_img = resize(input_img, (img_shape[0], img_shape[1], 3), mode='reflect')
        else:
            input_img = cv2.resize(input_img, (img_shape[0], img_shape[1]), interpolation = cv2.INTER_AREA)
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.reshape(input_img, (-1, 3, img_shape[0], img_shape[1]))

        input_imgs = np.concatenate((input_imgs, input_img))
    # As pytorch tensor
    input_imgs = torch.from_numpy(input_imgs).float()

    return input_imgs

def load_img(index, batch_size, coco_path, list_path):
    with open(list_path, 'r') as file:
        img_files = file.readlines()

    #---------
    #  Image
    #---------
    imgs = []
    img_paths = []
    if index + batch_size <= len(img_files):
        for i in range(batch_size):
            img_path = img_files[index % len(img_files)].rstrip()
            img_path = coco_path + '/' + img_path
            img = np.array(Image.open(img_path))
            # Handles images with less than three channels
            idx = index
            while len(img.shape) != 3:
                idx += 1
                img_path = img_files[idx % len(img_files)].rstrip()
                img_path = coco_path + '/' + img_path
                img = np.array(Image.open(img_path))
            index += 1
            imgs.append(img)
            img_paths.append(img_path)
    else:
        print("Out of img list range.")
        exit(0)
    return img_paths, imgs


@torch.no_grad()
def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    #print(opt.quantization)
    if opt.quantization:
        model = Model('models/yolov5x.yaml')
        state_dict = torch.load("../models/yolov5x.pt",map_location='cpu')['model'].state_dict()
        model.load_state_dict(state_dict)
        model.eval()
        qconfig = {'use_avg':False, 'data_scale':1.0, 'firstconv':False, 'per_channel':False}   
        quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype="int8",gen_quant=True)

    if opt.mlu:
        model = Model('./models/yolov5x.yaml')
        state_dict = torch.load('yolov5x_int8.pth')
        quantized_net = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
        quantized_net.load_state_dict(state_dict,strict=False)
        quantized_net.eval()
        quantized_net.to(ct.mlu_device())
        if opt.jit:
            ct.set_core_number(16)
#            if opt.save_offline_model:
#                ct.save_as_cambricon('yolov5x_int8_accuracy')
            trace_input = torch.randn(1,3,640,640,dtype=torch.float)
            trace_input = trace_input.to(ct.mlu_device())
            quantized_net = torch.jit.trace(quantized_net, trace_input, check_trace = False)

    if not opt.quantization and not opt.mlu:    
        model = attempt_load(weights, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()



    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for batch_i in tqdm.tqdm(range(0, opt.image_number, opt.batch_size)):
        img_path, imgs = load_img(batch_i, opt.batch_size, opt.coco_path, test_path)
        print("detect.py, imgs", imgs)
    
        if opt.mlu:
            imgs = preprocess(imgs, opt.img_size, normalize=False)
        else:
            imgs = preprocess(imgs, opt.img_size, normalize=True)
    
        # imgs = Variable(imgs.type(torch.HalfTensor)) if opt.half_input and opt.mlu else Variable(imgs.type(Tensor))
        
        for img in imgs: 
            if opt.quantization:
                pred = quantized_model(img,augment=opt.augment)[0]
            
            ## 处于CPU模式
            if not opt.quantization and not opt.mlu:
                pred = model(img, augment=opt.augment)[0]
            
            # 逐层
            if opt.mlu:

                img = torch.from_numpy(img).to(device)
                pred = quantized_net(img.to(ct.mlu_device()))
                
                t3 = time_synchronized()
                #print(f'\n quantized_net. ({t3-t1:.3f}s)')
                pred = pred.cpu()
                box = get_boxes(pred)
                #print("pred: ",box)
                #print("im0s.shape:", im0s.shape)
                print(box)
                pred = box

            # CPU    
            if not opt.mlu:
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                        max_det=opt.max_det)
                print("\n"+"pred: ", pred)

            #category_idx = 6 if opt.mlu else 5
            category_idx = 5 if opt.mlu else 4
            score_idx = 5 if opt.mlu else 4 
            outputs = pred
            if outputs == None:
                continue
            for si, pred in enumerate(outputs):
                if not torch.is_tensor(pred):
                    continue
                img = np.array(Image.open(img_path[si]))
                img_h, img_w, _ = img.shape
                img_name = os.path.splitext(os.path.basename(img_path[si]))[0]
                image_id = int(img_name.split('_')[-1].lstrip('0'))
                img_ids.append(image_id)
                box = pred[:, :4].clone()   #x1, y1, x2, y2
        
                scaling_factors = min(float(opt.img_size) / img_w, float(opt.img_size) / img_h)
        
                box[:, 0] = ((box[:, 0] - (opt.img_size - scaling_factors * img_w) / 2.0) / scaling_factors) / img_w
                box[:, 1] = ((box[:, 1] - (opt.img_size - scaling_factors * img_h) / 2.0) / scaling_factors) / img_h
                box[:, 2] = ((box[:, 2] - (opt.img_size - scaling_factors * img_w) / 2.0) / scaling_factors) / img_w
                box[:, 3] = ((box[:, 3] - (opt.img_size - scaling_factors * img_h) / 2.0) / scaling_factors) / img_h
        
                for di, d in enumerate(pred):
                    box_temp = []
                    box_temp.append(round(box[di][0].item(), 3) * img_w)
                    box_temp.append(round(box[di][1].item(), 3) * img_h)
                    box_temp.append((round(box[di][2].item(), 3) - round(box[di][0].item(), 3))  * img_w)
                    box_temp.append((round(box[di][3].item(), 3) - round(box[di][1].item(), 3))  * img_h)
                    jdict.append({'image_id': image_id,
                                'category_id': coco91class[int(d[category_idx])],
                                'bbox': box_temp,
                                'score': round(d[score_idx].item(), 5)})        
                                
        if opt.quantization:
            torch.save(quantized_model.state_dict(), "yolov5x_int8.pth")
            
        if opt.save_offline_model:
            ct.save_as_cambricon('yolov5x_int8_accuracy')
        
        print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../models/yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/val22017', help='source')  # file/folder, 0 for webcam
    parser.add_argument("--data_config_path", type=str, default="./data/coco.data", help="path to data config file")
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', type=bool, default=False, help='use FP16 half-precision inference')
    parser.add_argument('--quantization', type=str2bool, default=False, help='use FP16 half-precision inference')
    parser.add_argument('--mlu', default=False, type=str2bool, help='Use mlu to train model')
    parser.add_argument('--jit', default=False, type=str2bool, help='Use jit for inference net')
    parser.add_argument('--save_offline_model', default=False, type=str2bool, help='save offline model')
    parser.add_argument('--compute_map', default=False, type=str2bool, help='compute mAP, true or false')
    parser.add_argument('--run_mode', default=False, type=str2bool, help='Whether to output performance results')
    parser.add_argument("--json_name", dest = "json_name", help ="name of the output file(.json)", default = 'results',type = str)
    parser.add_argument("--ann_dir", dest = "ann_dir", help ="The annotation file directory", default = '',type = str)
    parser.add_argument("--data_type", dest = "data_type", help ="The data type. e.g. val2014, val2015, val2017",
                    default = 'val2017',type = str)
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument('--image_number', default = 2, type=int,help='test image number')
    opt = parser.parse_args()


    opt.coco_path = '/workspace/dataset/public/zhumeng-dataset/coco_2017/'
    opt.ann_dir = '/workspace/dataset/public/zhumeng-dataset/coco_2017/'
    data_config = parse_data_config(opt.data_config_path)
    test_path = data_config["valid"]

    if opt.jit:
        torch.set_grad_enabled(False)
    all_detections = []
    all_annotations = []

    seen = 0
    total_e2e = 0
    total_hardware = 0
    jdict, stats, img_ids = [], [], []
    coco91class = coco80_to_coco91_class()

    print(opt.quantization)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)

    if opt.compute_map:
        logger.info('Compute mAP...')
        json_file_name = '%s.json'%(opt.json_name)
        print(json_file_name)
        with open(json_file_name, 'w') as file_json:
            json.dump(jdict, file_json)

        data_dir = opt.ann_dir
        data_type = opt.data_type

        ann_type = ['segm', 'bbox', 'keypoints']
        ann_type = ann_type[1]
        prefix = 'person_keypoints' if ann_type == 'keypoints' else 'instances'

        ann_file = '%s/annotations/%s_%s.json'%(data_dir, prefix, data_type)
        coco_gt = COCO(ann_file)

        coco_dt = coco_gt.loadRes(json_file_name)

        coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP = round(coco_eval.stats[1], 3)
        print('Mean AP: ' + str(mAP))
    if opt.run_mode: 
        print('Throughput(fps): ' + str(opt.image_number / total_e2e))
        #print('Latency(ms): '+ str(opt.batch_size / (opt.image_number/total_hardware) * 1000))
        print('Latency(ms): '+ str(1 / (opt.image_number/total_hardware) * 1000))
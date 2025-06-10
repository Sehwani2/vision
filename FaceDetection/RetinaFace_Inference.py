# Import modules
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from models.retinaface import RetinaFace
from data import cfg_mnet, cfg_re50

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm

# 하이퍼파라미터 설정
weight_path = '/home/ash99/vision/FaceDetection/face_detection.pth'
cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'class_weight': 1.0,
    'landm_weight': 1.0,
    'pretrain': False,
    'return_layers': {
        'stage1': 1,
        'stage2': 2,
        'stage3': 3
    },
    'in_channel': 32,
    'out_channel': 64
}
resize = 1
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750
vis_thres = 0.6

# 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RetinaFace(cfg, phase= 'test').to(device)
model.load_state_dict(torch.load(weight_path, map_location= device))
model.eval()
print("Model Loaded!")

# Retinaface Inference 함수
def retinaface_inf(test_img, model):
    img = np.float32(test_img)
    im_height, im_width, _ = img.shape
    
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = model(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # 1. ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores =  scores[inds]

    # 2. keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # 3. do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]


    # 4. keep top-K faster NMS
    dets = dets[:keep_top_k, :]

    # 5. 최종 결과 출력하기
    fps_  =  round(1/(time.time() - tic),2)
 
    for b in dets:
        if b[4] < vis_thres:
            continue
        b = list(map(int, b))
        cv2.rectangle(test_img, (b[0], b[1]), (b[2],b[3]), (0,0,255), 4)
    cv2.putText(test_img, "retinaface", (410,70),cv2.FONT_HERSHEY_DUPLEX, 1.5,(255,0,0), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(test_img, "fps : "+str(fps_), (5,70),cv2.FONT_HERSHEY_DUPLEX, 1.5,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
    return test_img

test_path = '/home/ash99/vision/FaceDetection/test7.jpg'
test_img = cv2.imread(test_path)

result_retina = retinaface_inf(test_img,model)

# plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
# plt.show()

# 이미지 저장 
output_image_path = '/home/ash99/vision/FaceDetection/detected_faces_test7.jpg'
plt.imshow(cv2.cvtColor(result_retina, cv2.COLOR_BGR2RGB))
plt.axis('off') 
plt.savefig(output_image_path)
print(f"탐지된 얼굴 이미지가 다음 경로에 저장되었습니다: {output_image_path}")
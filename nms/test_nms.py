import sys
sys.path.insert(0,'../')
from nms.nms_wrapper import nms
import numpy as np
import torch

def nms_cuda(boxes_np, nms_thresh=0.7, xyxy=True):    
    if xyxy:
        x1,y1,x2,y2,scores=np.split(boxes_np, 5, axis=1)
        boxes_np=np.hstack([y1,x1,y2,x2,scores])
    boxes_pth=torch.from_numpy(boxes_np).float().cuda()
    pick=nms(boxes_pth, nms_thresh)
    pick=pick.cpu().data.numpy()
#    print(pick)
    if len(pick.shape)==2:
        pick=pick.squeeze(1)
    return pick

if __name__=='__main__':
    bboxes=np.ones((10000,5))
    pick=nms_cuda(bboxes)

    print(pick)
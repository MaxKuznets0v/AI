import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
import cfg
import dataset_gen
import ssd

# model = s3fd().cuda()
# model.load_state_dict(torch.load(os.path.join(utils.MODEL_PATH, "s3fd_convert.pth")))
# test_im = dataset_gen.transform(os.path.join(utils.DATASET_PATH, "train/49_Greeting_peoplegreeting_49_606.jpg"), resize=False)
# gg = cv2.imread(os.path.join(utils.DATASET_PATH, "train/49_Greeting_peoplegreeting_49_606.jpg"))
# #gg = cv2.resize(gg, utils.dim)
# olist = model(test_im)
#
# print(olist[1])
# bboxlist = []
# for i in range(len(olist)//2):
#     olist[i*2] = F.softmax(olist[i*2], dim=1)
# for i in range(len(olist)//2):
#     ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
#     FB,FC,FH,FW = ocls.size() # feature map size
#     stride = 2**(i+2)    # 4,8,16,32,64,128
#     anchor = stride*4
#     for Findex in range(FH*FW):
#         windex,hindex = Findex%FW,Findex//FW
#         axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
#         score = ocls[0,1,hindex,windex]
#         loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
#         if score<0.05: continue
#         priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
#         variances = [0.1,0.2]
#         box = torch.cat((
#             priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
#             priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
#         box[:, :2] -= box[:, 2:] / 2
#         box[:, 2:] += box[:, :2]
#         x1,y1,x2,y2 = box[0]*1.0
#         #cv2.rectangle(test_im,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
#         bboxlist.append([x1,y1,x2,y2,score])
# bboxlist = np.array(bboxlist)
# if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
#
#
# def nms(dets, thresh):
#     if 0==len(dets): return []
#     x1,y1,x2,y2,scores = dets[:, 0],dets[:, 1],dets[:, 2],dets[:, 3],dets[:, 4]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1,yy1 = np.maximum(x1[i], x1[order[1:]]),np.maximum(y1[i], y1[order[1:]])
#         xx2,yy2 = np.minimum(x2[i], x2[order[1:]]),np.minimum(y2[i], y2[order[1:]])
#
#         w,h = np.maximum(0.0, xx2 - xx1 + 1),np.maximum(0.0, yy2 - yy1 + 1)
#         ovr = w*h / (areas[i] + areas[order[1:]] - w*h)
#
#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]
#
#     return keep
#
#
# keep = nms(bboxlist,0.3)
# bboxlist = bboxlist[keep,:]
# for b in bboxlist:
#     x1,y1,x2,y2,s = b
#     if s<0.5: continue
#     cv2.rectangle(gg,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
# cv2.imshow('test',gg)
# cv2.waitKey(0)
# exit()

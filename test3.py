import numpy as np
import PIL
from PIL import Image

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

#
logits_u_w = np.zeros([2,2,2])
gt_mask = np.zeros([2,2,1])

logits_u_w[:,:,0] += 1
# logits_u_w[:,:,1] += 0

logits_u_w[1,1,0] = 0.02
logits_u_w[1,1,1] = 0.98


gt_mask[1,1,0] += 1

numerator = 2*np.sum(gt_mask * logits_u_w)
denominator = np.sum(gt_mask + logits_u_w)

loss = np.mean(1 - numerator / denominator)
print(numerator,denominator,loss)




print(gt_mask)
print(logits_u_w)
#
# logits_u_w = np.sum(logits_u_w.max(axis=-1))
# print(logits_u_w)
#
# pseudo_label = np.zeros(logits_u_w.shape)
# pseudo_label[np.where((logits_u_w < 0.5).all(axis=-1))] = 1
#
# image = np.zeros([10,10,3], dtype=np.uint8)
# image[np.where((pseudo_label[0] == [1]).all(axis=-1))] = [255,255,255]
#
# image = Image.fromarray(image)
# image.show()
#
# print(image)
# print(pseudo_label.shape)
# print(pseudo_label)

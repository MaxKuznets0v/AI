import torch
import torch.nn as nn
from utils import config as cfg


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
        Compute Targets:
            1) Produce Confidence Target Indices by matching  ground truth boxes
               with (default) 'priorboxes' that have jaccard index > threshold parameter
               (default threshold: 0.5).
            2) Produce localization target by 'encoding' variance into offsets of ground
               truth boxes and their matched  'priorboxes'.
            3) Hard negative mining to filter the excessive number of negative examples
               that comes with using a large number of default bounding boxes.
               (default negative:positive ratio 3:1)
        Objective Loss:
            L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
            Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
            weighted by α which is set to 1 by cross val.
            Args:
                c: class confidences,
                l: predicted boxes,
                g: ground truth boxes
                N: number of matched default boxes
            See: https://arxiv.org/pdf/1512.02325.pdf for more details.
        """
    def __init__(self):
        super(MultiBoxLoss, self).__init__()

    def forward(self):
        """Multibox Loss
                Args:
                    predictions (tuple): A tuple containing loc preds, conf preds,
                    and prior boxes from SSD net.
                        conf shape: torch.size(batch_size,num_priors,num_classes)
                        loc shape: torch.size(batch_size,num_priors,4)
                        priors shape: torch.size(num_priors,4)

                    ground_truth (tensor): Ground truth boxes and labels for a batch,
                        shape: [batch_size,num_objs,5] (last idx is the label).
                """

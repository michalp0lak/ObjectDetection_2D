import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import product as product
import numpy as np
from math import ceil
import time


# Network blocks
def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class Prior2DBoxGenerator(object):

    def __init__(self,
                 steps,
                 aspects,
                 sizes,
                 image_size,
                 clip = False):

        super(Prior2DBoxGenerator, self).__init__()
        self.steps = steps
        self.aspects = aspects
        self.sizes = sizes
        self.image_size = image_size
        self.clip = clip
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward_v1(self, device):

        priors = []

        for k, feature_map in enumerate(self.feature_maps):

            fmap_sizes = self.sizes[k]

            x = torch.arange(0.5, feature_map[1]+0.5,1, device=device)*(self.steps[k]/self.image_size[1])
            y = torch.arange(0.5, feature_map[0]+0.5,1, device=device)*(self.steps[k]/self.image_size[0])

            grid = torch.meshgrid(y, x, indexing='ij')
            prior_centers = torch.cat([grid[1].reshape(-1,1), grid[0].reshape(-1,1)], dim=-1)
            
            prior_dims = []

            for ratio in self.aspects:
                for size in fmap_sizes:


                    if ratio == 1:
                        
                        prior_dims_e = torch.tensor([size / self.image_size[1], size / self.image_size[0]], 
                            device=device).unsqueeze(0)
                        prior_dims.append(prior_dims_e)

                    else:
                        prior_dims_v = torch.tensor([ratio*size / self.image_size[1], size / self.image_size[0]], 
                            device=device).unsqueeze(0)
                        prior_dims.append(prior_dims_v)

                        prior_dims_h = torch.tensor([size / self.image_size[1], ratio*size / self.image_size[0]], 
                            device=device).unsqueeze(0)
                        prior_dims.append(prior_dims_h)
                

            prior_dims = torch.cat(prior_dims, dim=0)
            prior_num = prior_dims.size(0)
            prior_dims = prior_dims.repeat(prior_centers.size(0), 1)
            prior_centers = torch.repeat_interleave(prior_centers, prior_num, dim=0)
            priors_fmap = torch.cat([prior_centers,prior_dims], dim=-1)
            priors.append(priors_fmap)
        
        priors = torch.cat(priors)

        if self.clip:
            priors.clamp_(max=1, min=0)

        return priors

    def forward_v2(self, device):

        priors = []

        for k, feature_map in enumerate(self.feature_maps):

            fmap_sizes = self.sizes[k]

            x = torch.arange(0.5, feature_map[1]+0.5,1, device=device)*(self.steps[k]/self.image_size[1])
            y = torch.arange(0.5, feature_map[0]+0.5,1, device=device)*(self.steps[k]/self.image_size[0])

            grid = torch.meshgrid(x, y, indexing='ij')
            prior_centers = torch.cat([grid[0].reshape(-1,1), grid[1].reshape(-1,1)], dim=-1)

            fmap_priors = []

            for size in fmap_sizes:

                prior_dims = torch.tensor([size / self.image_size[1],size / self.image_size[0]], 
                                          device=device).unsqueeze(0).repeat(prior_centers.size(0), 1)
                
                priors_single = torch.cat([prior_centers, prior_dims], dim=-1)

                for ratio in self.aspects:

                    if ratio == 1:

                        fmap_priors.append(priors_single)

                    else:
                        priors_single_v= priors_single.clone()
                        priors_single_v[:,2] *= ratio  
                        fmap_priors.append(priors_single_v)
                        priors_single_h= priors_single.clone()
                        priors_single_h[:,3] *= ratio  
                        fmap_priors.append(priors_single_h)

            fmap_priors = torch.cat(fmap_priors, dim = 1)
            priors.append(fmap_priors.view(-1,4))

        priors = torch.cat(priors)

        if self.clip:
            priors.clamp_(max=1, min=0)

        return priors


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)

def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)

class BoxCoder(object):
    """
        Bbox Coder for 2D boxes.
    """

    def __init__(self, variances):
        super(BoxCoder, self).__init__()
        self.variances = variances

    def encode(self, target_boxes, priors):
        """Encode the variances from the priorbox layers into the ground truth boxes
        we have matched (based on jaccard overlap) with the prior boxes.
        Args:
            matched: (tensor) Coords of ground truth for each prior in point-form
                Shape: [num_priors, 4].
            priors: (tensor) Prior boxes in center-offset form
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            encoded boxes (tensor), Shape: [num_priors, 4]
        """

        # dist b/t match center and prior's center
        center_diff = (target_boxes[:, :2] + target_boxes[:, 2:])/2 - priors[:, :2]
        # encode variance
        center_diff /= (self.variances[0] * priors[:, 2:])
        # match wh / prior wh
        dim_diff = (target_boxes[:, 2:] - target_boxes[:, :2]) / priors[:, 2:]
        dim_diff = torch.log(dim_diff) / self.variances[1]
        # return target for smooth_l1_loss
        return torch.cat([center_diff, dim_diff], 1)  # [num_priors,4]

    # Adapted from https://github.com/Hakuyume/chainer-ssd
    def decode(self, predicted_boxes, priors):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = torch.cat((
            priors[:, :2] +  predicted_boxes[:, :2] * self.variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp( predicted_boxes[:, 2:] * self.variances[1])), 1)

        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes
        
def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, score_thresh, top_k, overlap=0.5):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    
    scores_indx = scores > score_thresh
    orig_idx = torch.arange(scores.shape[0], device=scores_indx.device,dtype=torch.long)[scores_indx]
    scores = scores[scores_indx]
    boxes = boxes[scores_indx]  
    
    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order

    if scores.shape[0] > top_k:
        idx = idx[-top_k:]  # indices of the top-k largest vals


    w = boxes.new()
    h = boxes.new()
    
    count = 0
    while idx.numel() > 0:

        i = idx[-1]  # index of current largest val
        #keep[count] = i
        keep[count] = orig_idx[i]
        count += 1

        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view

        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()
        yy2 = boxes.new()

        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h

        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    
    return keep[:count]

def multiclass_nms(boxes, scores, score_thr, iou_thr):
    """Multi-class nms for 3D boxes.
    Args:
        boxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        scores (torch.Tensor): Multi-level boxes with shape
            (N, ). N is the number of boxes.
        score_thr (float): Score threshold to filter boxes with low
            confidence.
        iou_thr (float): IoU threshold to surpress highly overlapping bounding boxes.
    Returns:
        list[torch.Tensor]: Return a list of indices after nms,
            with an entry for each class.
    """

    idxs = []

    # For each class
    for i in range(scores.shape[1]):
        # Check for all datums if predicted score is bigger than threshold
        cls_inds = scores[:, i] > score_thr

        # If class was not predicted strongly enough for any datum
        if not cls_inds.any():
            
            # Append empty tensor and go to another class
            idxs.append(torch.tensor([], dtype=torch.long, device=cls_inds.device))
            continue

        # Tensor of original indices of datums, which were selected for given class
        orig_idx = torch.arange(cls_inds.shape[0], device=cls_inds.device,dtype=torch.long)[cls_inds]

        # Sample valid scores and boxes
        _scores = scores[cls_inds, i]
        _boxes = boxes[cls_inds, :]

        scores_sorted = torch.argsort(_scores, dim=0, descending=True)

        orig_idx = orig_idx[scores_sorted]
        boxes_sorted = _boxes[scores_sorted,:]

        box_indices = torch.arange(0, boxes_sorted.shape[0]).cuda()
        suppressed_box_indices = []

        while box_indices.shape[0] > 0:
            # If box with highest classification score is not among suppressed bounding boxes
            if box_indices[0] not in suppressed_box_indices:
                # Choose box with best score
                selected_box = box_indices[0]          
                tmp_suppress = []
                selected_iou = jaccard(boxes_sorted[box_indices], boxes_sorted[selected_box].unsqueeze(0))
                mask_iou = (selected_iou > iou_thr).squeeze(-1)

                mask_indices = box_indices != selected_box
                mask = mask_iou & mask_indices
                suppressed_box_indices.append(box_indices[mask].tolist())
            
            box_indices = box_indices[torch.logical_not(mask)]
            box_indices = box_indices[1:]
        
        suppressed_box_indices = [idx for slist in suppressed_box_indices for idx in slist]
        preserved_box_indexes = list(set(np.arange(0, boxes_sorted.shape[0]).tolist()) - set(suppressed_box_indices))
        idxs.append(orig_idx[preserved_box_indexes])

    return idxs
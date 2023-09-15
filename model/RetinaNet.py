import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F

from model.MobileNet import MobileNetV1 as MobileNetV1
from model.utils import conv_bn, conv_bn_no_relu, conv_bn1X1
from model.base_model import BaseModel
from model.utils import BoxCoder, Prior2DBoxGenerator
from model.utils import nms, jaccard, point_form, log_sum_exp
from augment.augmentation import Augmentation

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):

        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):

        conv3X3 = self.conv3X3(input)
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)

        return out

class ClassHead(nn.Module):
    def __init__(self,inchannels=512, num_classes=3, num_anchors=2):
        super(ClassHead,self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_classes*self.num_anchors, kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, self.num_classes)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)
    
class Head(nn.Module):

    def __init__(self,
                 channels,
                 steps,
                 aspects,
                 sizes,
                 variances,
                 negpos_ratio,
                 num_classes,
                 nms_top_k,
                 nms_thresh,
                 score_thr,
                 iou_thr,
                 image_size=2000):

        super().__init__()
        self.channels = channels
        self.steps = steps
        self.aspects = aspects
        self.sizes = sizes 
        self.image_size = image_size
        
        if type(self.image_size) == int:
            self.image_size = [self.image_size, self.image_size]

        self.fm_num = len(steps)
        self.num_classes = num_classes
        self.nms_top_k = nms_top_k
        self.nms_thresh = nms_thresh
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.variances = variances
        self.negpos_ratio = negpos_ratio
        self.num_anchors = 0

        for aspect in self.aspects:
            if aspect == 1: self.num_anchors += len(self.sizes[0])
            else:
                self.num_anchors += 2*len(self.sizes[0])

        assert len(self.steps) == len(self.sizes)
        
        self.ClassHead = self._make_class_head(fpn_num=self.fm_num, inchannels=self.channels, 
                                               classes_num=self.num_classes, anchors_num=self.num_anchors)

        self.BboxHead = self._make_bbox_head(fpn_num=self.fm_num, inchannels=self.channels, anchors_num=self.num_anchors)
        self.box_coder = BoxCoder(self.variances)

    def _make_class_head(self,fpn_num,inchannels,classes_num, anchors_num):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, classes_num, anchors_num))
        return classhead
    
    def _make_bbox_head(self, fpn_num, inchannels, anchors_num):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchors_num))
        return bboxhead


    def forward(self, features):
        """Forward function on a feature map.
        Args:
            x (torch.Tensor): Input features.
        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox regression
        """

        cls_score = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        bbox_pred = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)

        return cls_score, bbox_pred

    def assign_bboxes(self, predictions, targets):

        predicted_scores, predicted_bboxes = predictions

        target_bboxes = targets.boxes
        target_labels = targets.labels
        
        # Priors generator has to be initialized for every call/batch so it can adapt for various batch image size.
        # It can't be initialized statically with model initialization.
        priorsGen = Prior2DBoxGenerator(steps=self.steps,
                                        aspects=self.aspects,
                                        sizes=self.sizes,
                                        image_size=(self.image_size[0],self.image_size[1])
                                       )

        priors = priorsGen.forward_v1(device=predicted_bboxes.device)

        # Allocate matrices for priors - gt boxes matching
        batch_bboxes = torch.empty(size=(predicted_bboxes.size(0), priors.size(0), 4),
                                    device=predicted_bboxes.device, dtype=torch.float32)
       
        batch_labels = torch.empty(size=(predicted_bboxes.size(0), priors.size(0)),
                                    device=predicted_bboxes.device, dtype=torch.int64)

        # Go through each image in a batch to assign labels and bboxes of priors with targets
        for idx in range(predicted_bboxes.size(0)):
            # For multi class case add for loop through classes
 
            # jaccard index - IOU          
            overlaps = jaccard(target_bboxes[idx], point_form(priors))
        
            # Best prior for each ground truth
            best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
           
            # Ignore gt boxes which is hard to match -> on certain level of IOU (or change your priors)
            valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
            best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]

            if best_prior_idx_filter.shape[0] <= 0:
                batch_bboxes[idx] = 0
                batch_labels[idx] = 0
            else:
                # best ground truth box for each prior
                best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

                # Squeeze all
                best_truth_idx.squeeze_(0)
                best_truth_overlap.squeeze_(0)
                best_prior_idx.squeeze_(1)
                best_prior_idx_filter.squeeze_(1)
                best_prior_overlap.squeeze_(1)

                # Fills IOU value with 2, in indexes of best priors
                best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
        
                # ensure every gt matches with its prior of max overlap
                for j in range(best_prior_idx.size(0)):
                    best_truth_idx[best_prior_idx[j]] = j

                assigned_bboxes = target_bboxes[idx][best_truth_idx]         # Shape: [num_priors,4]
                assigned_labels = target_labels[idx][best_truth_idx]         # Shape: [num_priors]      
                assigned_labels[best_truth_overlap < self.iou_thr] = 0       # badly overlaping bboxes label as background examples

                bboxes_offset = self.box_coder.encode(assigned_bboxes, priors)
                batch_bboxes[idx] = bboxes_offset                # [num_priors,4] encoded offsets to learn
                batch_labels[idx] = assigned_labels.squeeze(-1)  # [num_priors] top class label for each prior
        ##########################################################################################################

        # Get positive prior matches -> iou > threshold
        positives = batch_labels != 0
        # Changing labels only to {0,1} values
        batch_labels[positives] = 1
        
        # Compute max conf across batch for hard negative mining -> to generate negative priors for learning
        batch_scores = predicted_scores.view(-1, self.num_classes)

        class_loss = log_sum_exp(batch_scores) - batch_scores.gather(1, batch_labels.view(-1, 1))
        # Hard Negative Mining
        class_loss[positives.view(-1, 1)] = 0 # filter out pos boxes for now
        class_loss = class_loss.view(predicted_bboxes.size(0), -1)
        # Sort class loss of priors in descending order for each image in batch individually
        _, loss_idx = class_loss.sort(1, descending=True)
        # Get order rank index for each prior for each image in batch individually
        _, idx_rank = loss_idx.sort(1)
        # Count of positive priors
        num_pos = positives.long().sum(1, keepdim=True)
        # Count of negative priors based on Neg-Pos ratio 
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=positives.size(1)-1)
        # Get negative indexes for each image in a batch 
        # (count varies form image to image, because of various count positives in each image)
        negatives = idx_rank < num_neg.expand_as(idx_rank)

        return batch_labels, batch_bboxes, positives, negatives

    def get_bboxes(self, scores_pred, boxes_pred, priors):

        """Get bboxes of anchor head.
        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.
        Returns:
            tuple[torch.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """

        assert scores_pred.size()[0] == boxes_pred.size()[0]

        boxes_pred = self.box_coder.decode(boxes_pred, priors)
        idxs = nms(boxes_pred, scores_pred[:,1], self.score_thr, self.nms_top_k, self.nms_thresh)
        scores = scores_pred[idxs,1]
        labels = torch.full((idxs.shape[0],), 1, dtype=torch.long)
        boxes = boxes_pred[idxs]

        return scores, labels, boxes

class RetinaDetector(BaseModel):
    
    def __init__(self,
                 device="cuda",
                 backbone='mobilenet',
                 pretrained=True,
                 classes=[],
                 augment={},
                 head={},
                 **kwargs):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaDetector,self).__init__(device=device, **kwargs)
        self.device = device
        self.backbone = backbone
        self.pretrained = pretrained

        self.classes = classes
        self.name2lbl = {n: i+1 for i, n in enumerate(self.classes)}
        self.lbl2name = {i+1: n for i, n in enumerate(self.classes)}
        self.classes_ids = [i+1 for i, _ in enumerate(self.classes)]

        self.augment = augment
        self.head = head

        self.steps = self.head.steps
        self.aspects = self.head.aspects
        self.sizes = self.head.sizes
        self.image_size = self.head.image_size
        
        if type(self.image_size) == int:
            self.image_size = [self.image_size, self.image_size]

        self.backbone_cfg = self.cfg[self.backbone]

        # Initialize required backbone - MobileNet vs ResNet50
        # MobileNet
        if self.backbone == 'mobilenet':
            backbone_model = MobileNetV1()
            if pretrained:
                checkpoint = torch.load("./weights/mobilenet.tar", map_location=self.device)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone_model.load_state_dict(new_state_dict)

        # ResNet
        elif self.backbone == 'resnet':
            import torchvision.models as models
            if pretrained:
                backbone_model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            else:
                backbone_model = models.resnet50()

        self.body = _utils.IntermediateLayerGetter(backbone_model, self.backbone_cfg['return_layers'])
        in_channels_stage2 = self.backbone_cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = self.backbone_cfg['out_channel']

        self.augmentor = Augmentation(augment)
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.head = Head(channels=out_channels, num_classes=len(self.classes_ids)+1, **head)

        self.to(self.device)


    def forward(self,inputs):
        
        out = self.body(inputs.images)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        cls_pred, boxes_pred = self.head(features)

        return cls_pred, boxes_pred

    def get_optimizer(self, cfg):

        optimizer = torch.optim.SGD(self.parameters(), **cfg)

        return optimizer, None

    def loss(self, predictions, targets):

        pred_scores, pred_boxes = predictions

        labels, boxes, pos_idx, neg_idx = self.head.assign_bboxes(predictions, targets)
    
        N = max(pos_idx.data.sum().float(), 1)

        # Confidence Loss Including Positive and Negative Examples
        loss_cls = F.cross_entropy(pred_scores[(pos_idx+neg_idx).gt(0)].view(-1, self.head.num_classes),
                                   labels[(pos_idx+neg_idx).gt(0)], reduction='sum')

        if torch.sum(pos_idx) > 0:
            # Localization Loss (Smooth L1)
            loss_box = F.smooth_l1_loss(pred_boxes[pos_idx].view(-1, 4), 
                                        boxes[pos_idx].view(-1, 4), reduction='sum')
        else:

            loss_box = boxes.sum()

        return {
                'loss_cls': loss_cls/N,
                'loss_box': loss_box/N,
               }
    
    def preprocess(self, data, attr):

        #Augment data
        data = self.augmentor.augment(data, attr['split'])

        new_data = {'image': data['image'], 'labels': data['labels'], 'boxes': data['boxes']}

        return new_data   

    def transform(self, data, attr):

        return data

    def inference_end(self, results):

        batch_score, batch_boxes = results
        batch_score = F.softmax(batch_score, dim=-1)
        inference_result = []

        priors_generator = Prior2DBoxGenerator(steps=self.steps,
                                        aspects=self.aspects,
                                        sizes=self.sizes,
                                        image_size=(self.image_size[0],self.image_size[1])
                                        )
        
        self.priors = priors_generator.forward_v1(self.device)

        for idx in range(batch_boxes.size(0)):

            boxes = batch_boxes[idx]
            scores = batch_score[idx]

            pred_scores, pred_labels, pred_boxes = self.head.get_bboxes(scores, boxes, self.priors)

            inference_result.append([])
 
            for box, score, label in zip(pred_boxes,  pred_scores, pred_labels):
        
                inference_result[-1].append({'label': label, 'score': score, 'box': box})

        return inference_result
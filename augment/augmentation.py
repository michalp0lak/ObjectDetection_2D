import torch
import torchvision.transforms.functional as F
from math import floor

import cv2
import numpy as np
import random
from augment.ops import matrix_iof 

class Augmentation():

    """Class consisting common augmentation methods for image object detection pipeline."""

    def __init__(self, cfg, seed=None):

        self.img_dim = cfg['image_size']

    def crop(self, image, boxes, labels, img_dim):

        height, width, _ = image.shape
        pad_image_flag = True

        # Crop procedure performs 250 trials to crop image, if not succeed it returns original image
        for _ in range(250):
            """
            if random.uniform(0, 1) <= 0.2:
                scale = 1.0
            else:
                scale = random.uniform(0.3, 1.0)
            """
            # Randomly select scale
            PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
            scale = random.choice(PRE_SCALES)
            # Define square shape of image
            short_side = min(width, height)
            w = int(scale * short_side)
            h = w

            # Define ROI pixels
            if width == w:
                l = 0
            else:
                l = random.randrange(width - w)
            if height == h:
                t = 0
            else:
                t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            # If any bbox covers whole ROI -> invalid cropping
            value = matrix_iof(boxes, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue
            
            # Check for bboxes located in ROI
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask_a].copy()
            labels_t = labels[mask_a].copy()

            if boxes_t.shape[0] == 0:
                continue
            
            # Crop image and bboxes
            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

	        # make sure that the cropped image contains at least one bbox > 16 pixel at training image scale
            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
            mask_b = np.minimum(b_w_t, b_h_t) > 0.0
            boxes_t = boxes_t[mask_b]
            labels_t = labels_t[mask_b]

            if boxes_t.shape[0] == 0:
                continue

            pad_image_flag = False

            return image_t, boxes_t, labels_t, pad_image_flag
        return image, boxes, labels, pad_image_flag

    def distort(self, image):

        def apply_distortion(image, alpha=1, beta=0):

            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        image = image.copy()

        if random.randrange(2):

            #brightness distortion
            if random.randrange(2):
                apply_distortion(image, beta=random.uniform(-32, 32))

            #contrast distortion
            if random.randrange(2):
                apply_distortion(image, alpha=random.uniform(0.5, 1.5))
        
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            #saturation distortion
            if random.randrange(2):
                apply_distortion(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            #hue distortion
            if random.randrange(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        else:

            #brightness distortion
            if random.randrange(2):
                apply_distortion(image, beta=random.uniform(-32, 32))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            #saturation distortion
            if random.randrange(2):
                apply_distortion(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            #hue distortion
            if random.randrange(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            #contrast distortion
            if random.randrange(2):
                apply_distortion(image, alpha=random.uniform(0.5, 1.5))

        return image

    def expand(self, image, boxes, fill, p):

        # Randomly choose if to expand
        if random.randrange(2):
            return image, boxes

        height, width, depth = image.shape

        # Get expanded dimensions
        scale = random.uniform(1, p)
        w = int(scale * width)
        h = int(scale * height)

        # Get offsets
        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        # Shift the boxes with offsets
        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)

        # Empty expanded image
        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
            
        # Fill with triplet/color
        expand_image[:, :] = fill
        # Fit original into expanded canvas
        expand_image[top:top + height, left:left + width] = image

        return expand_image, boxes_t

    def mirror(self, image, boxes):

        _, width, _ = image.shape
        
        if random.randrange(2):

            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, boxes

    def pad_to_square(self, image, pad_image_flag):

        if not pad_image_flag:
            return image

        height, width, depth = image.shape
        long_side = max(width, height)
        image_t = np.empty((long_side, long_side, depth), dtype=image.dtype)
        image_t[:, :] = 0
        image_t[0:0 + height, 0:0 + width] = image

        return image_t

    def resize(self, image, insize):

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        image = cv2.resize(image, (insize, insize), interpolation=interp_method)
        
        if len(image.shape) == 2:
            image = np.expand_dims(image,-1)

        return image
    

    def brightness_augmentor(self, image, prob = 0.33):

        # Randomly determine if brightness augmentation will be executed
        if np.random.binomial(1, prob):

            image = torch.from_numpy(image).permute(2,0,1)

            # to generate image under dark condition is more probable
            if np.random.binomial(1, 0.65):

                brightness_factor = np.random.uniform(0.2,0.7)
                image = F.adjust_brightness(image, brightness_factor)

            # to generate image bright/sunny condition is less probable
            else:

                brightness_factor = np.random.uniform(1.3, 2)
                image = F.adjust_brightness(image, brightness_factor)

            image = image.permute(1,2,0).cpu().detach().numpy()

        return image


    def small_occlusion(self, image, box):

        value = [0,32,64,128,160,192,224,255]

        h,w = image.shape[:2]

        width = box[2]-box[0]
        height = box[3]-box[1]

        occlusion_rate = np.random.uniform(0.05,0.2)
        start_pix = np.random.randint(box[0], floor(box[2]-width*occlusion_rate))

        occlusion_box = [int(start_pix),
                         int(max(0,box[1]-height)),
                         int(start_pix+floor(width*occlusion_rate)),
                         int(min(h, box[3]+height))
                        ]
        
        image[occlusion_box[1]:occlusion_box[3], 
                occlusion_box[0]:occlusion_box[2], :] = random.sample(value,1) * 3

    def big_occlusion(self, image, box):

        h,w = image.shape[:2]

        width = box[2]-box[0]
        height = box[3]-box[1]

        # Randomly sample occlusion rate
        occlusion_rate = np.random.uniform(0.25,0.65)

        # Left part of LP is going to be occluded
        if np.random.binomial(1, 0.5):

            occlusion_box = [int(max(0, box[0]-1.5*width)),
                            int(max(0,box[1]-3*height)),
                            int(box[0]+floor(width*occlusion_rate)),
                            int(min(h,box[3]+3*height))]
            
            box[0] += floor(width*occlusion_rate)
            
        # Right part of LP is going to be occluded
        else:

            occlusion_box = [int(box[2]-floor(width*occlusion_rate)),
                            int(max(0,box[1]-3*height)),
                            int(min(w, box[2]+1.5*width)),
                            int(min(h,box[3]+3*height))]

            box[2] -= floor(width*occlusion_rate)

        
        image[occlusion_box[1]:occlusion_box[3], 
                occlusion_box[0]:occlusion_box[2],:] = np.random.randint(0,255, 3)

        return box
   

    def occlusion(self, image, boxes, prob = 0.2):

        # For each box
        for i, box in enumerate(boxes):
            # Randomly applies occlusion with 20% probability
            if np.random.binomial(1, prob):
                # Big occlusion has 60% probability
                if np.random.binomial(1, 0.6):
                    box = self.big_occlusion(image, box)
                    boxes[i] = box
                # Small occlusion has 40% probability
                else:
                    self.small_occlusion(image, box)

        return image, boxes


    def augment(self, data, split):

        image = data['image'].copy()
        boxes = data['boxes'].copy()
        labels = data['labels'].copy()
        
        assert boxes.shape[0] > 0, "this image does not contain any annotation"

        image_t = image.copy()
        boxes_t = boxes.copy()
        labels_t = labels.copy()

        if split in ['training', 'validation']:
            
            # Augment brightness
            image_t = self.brightness_augmentor(image_t)

            # Generate occluded bboxes
            image_t, boxes_t = self.occlusion(image_t, boxes_t)

        # Crop
        #if split in ['training']:
        #    image_t, boxes_t, labels_t, pad_image_flag = self.crop(image, boxes, labels, self.img_dim)
        #Distort
        #if split in ['training']:
        #    image_t = self.distort(image_t)

        # Pad to square if not square 
        # (for batch processing we resize all images in batch to the same square size, so we don't have to change aspect ratio if it's padded)
        if split in ['training', 'validation']:
            image_t = self.pad_to_square(image_t, pad_image_flag=True)

        # Mirror
        #if split in ['training']:
        #    image_t, boxes_t, centers_t, directions_t = self.mirror(image_t, boxes_t, centers_t, directions_t)

        # Get image dims before resizing for bboxes coords calculation
        height, width, _ = image_t.shape

        # Resize image to required shape
        if split in ['training', 'validation']:
            image_t = self.resize(image_t, self.img_dim)

        # Cast image to float and normalize between [0,1]
        image_t = image_t.astype(np.float32)/255

        # Get relative location of target bbox
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        
        labels_t = np.expand_dims(labels_t, 1)

        return {'image': image_t, 'labels':labels_t, 'boxes': boxes_t}
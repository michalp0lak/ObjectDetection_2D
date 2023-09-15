import logging
import re
from datetime import datetime
import os 
from os.path import join
import random
import yaml
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import cv2

from torch.utils.data import DataLoader
from dataset.dataloaders import TorchDataloader, ConcatBatcher
from pipeline.utils import latest_ckpt
from utils import make_dir
from pipeline.metrics import MetricEvaluator
from pipeline.base_pipeline import BasePipeline

log = logging.getLogger(__name__)

class ObjectDetection(BasePipeline):
    """Pipeline for object detection."""

    def __init__(self, model, dataset, cfg_dict, **kwargs):

        super().__init__(model=model,
                         dataset=dataset,
                         cfg_dict=cfg_dict,
                         **kwargs)
                         
        self.ME = MetricEvaluator(self.device)

    def save_ckpt(self, epoch, save_best = False):

        ckpt_dir = join(self.cfg.log_dir, 'checkpoint/')
        make_dir(ckpt_dir)

        if save_best: path = join(ckpt_dir,'ckpt_best.pth')
        else: path = join(ckpt_dir, f'ckpt_{epoch:05d}.pth')

        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict()),
                 path)

        log.info(f'Epoch {epoch:3d}: save ckpt to {path:s}')

    def load_ckpt(self):
        
        ckpt_dir = join(self.cfg.log_dir, 'checkpoint/')
        epoch = 0
        
        if not self.cfg.inference_mode:
            
            if self.cfg.is_resume:

                last_ckpt_path = latest_ckpt(ckpt_dir)
            
                if last_ckpt_path:

                    epoch = int(re.findall(r'\d+', last_ckpt_path)[-1])
                    ckpt_path = last_ckpt_path
                    log.info('Model restored from the latest checkpoint: {}'.format(epoch))

                else:

                    log.info('Latest checkpoint was not found')
                    log.info('Initializing from scratch.')
                    return epoch, None

            else:
                log.info('Initializing from scratch.')
                return epoch, None    

        else:

            ckpt_path = self.cfg.log_dir + 'checkpoint/ckpt_best.pth'

            if not os.path.exists(ckpt_path):
                raise ValueError('There is not pretrained model for inference. Best output of training should be found as {}'.format(ckpt_path))
                    
        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device) 
        
        self.model.load_state_dict(ckpt['model_state_dict'])

        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            log.info('Loading checkpoint optimizer_state_dict')
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        return epoch, ckpt_path
    

    def transform_input_batch(self, boxes, labels=None):

        dicts = []
       
        if labels is None:
            for box,  in zip(boxes):
                dicts.append({'box': box})
        else:
            for box,  label in zip(boxes, labels):
                dicts.append({'label': label, 'box': box})

        return dicts

    def transform_for_metric(self, boxes):

        """Convert data for evaluation:
        Args:
            bboxes: List of predicted items (box, label).
        """

        box_dicts = {
            'label': torch.empty((len(boxes),)).to(self.device),
            'score': torch.empty((len(boxes),)).to(self.device),
            'box': torch.empty((len(boxes), 4)).to(self.device)
                    }

        for i in range(len(boxes)):
            box_dict = boxes[i]

            for k in box_dict:
                box_dicts[k][i] = box_dict[k]

        return box_dicts


    def run_inference(self, data):
        """Run inference on given data sample.
        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """

        self.load_ckpt()
        self.model.eval()

        # If run_inference is called on raw data.
        if isinstance(data, dict):
            batcher = ConcatBatcher(self.device)
            data = batcher.collate_fn([{
                'data': data['data'],
                'attr': data['attr']
            }])

        data.to(self.device)
        # If Pipeline should process batches of various image size, priors feature maps needs to adapt according
        # to this size. Head image_size parameter is updated with every batch, so detection head is able
        # to adapt for various image size. This update is performed here within a pipeline, when image batch is
        # formed for forward pass.
        self.model.image_size = data.images.shape[-2:]

        with torch.no_grad():

            results = self.model(data)
            boxes = self.model.inference_end(results)

        return boxes

    def show_inference(self):
        
        """
        Show_inference session randomly samples item from
        testing dataset split, predicts output and visualize 
        it with ground truth annotation. 
        """

        test_dataset = self.dataset.get_split('testing')
        
        test_split = TorchDataloader(dataset=test_dataset,
                                     preprocess=self.model.preprocess,
                                     transform=self.model.transform,
                                    )

        idx = random.sample(range(0, len(test_dataset)), 1)
        data_item = test_split.__getitem__(idx[0])
        print(idx[0])
        print(data_item['attr'])

        predicted_items = self.run_inference(data_item)
    
        data = data_item['data']

        target = [self.transform_for_metric(self.transform_input_batch(boxes = torch.Tensor(data['boxes'].tolist()),        
                                                                       labels = torch.Tensor(data['labels'].tolist())
                                                                    ))]
  
        prediction = [self.transform_for_metric(item) for item in predicted_items]

        # mAP metric evaluation for epoch over all validation data
        precision, recall = self.ME.evaluate(prediction,
                                             target,
                                             self.model.classes_ids,
                                             self.cfg.get("overlaps", [0.5]))

        print("")
        print(f' {" ": <3} "==== Precision ==== Recall ==== F1 ====" ')
        desc = ''
        for i, c in enumerate(self.model.classes):

            p = precision[i,0]
            rec = recall[i,0]

            if (p == 0) and (rec == 0):
                f1 = np.nan
            else: 
                f1 = 2*p*rec/(p+rec)
 
            desc += f' {"{}".format(c): <9} {"{:.2f}".format(p): <14} {"{:.2f}".format(rec): <11} {"{:.2f}".format(f1)}'

        print(desc)

        img = data['image']
        h,w = img.shape[:2]

        for box in data['boxes']:
            
            box[0::2] = w*box[0::2].clip(min=0, max=w)
            box[1::2] = h*box[1::2].clip(min=0, max=h)
            box = box.astype(np.int32)
            img = cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(255,0,0),1)

        for item in predicted_items[0]:

            box = item['box']
            box[0::2] = w*box[0::2].clamp(min=0, max=w)
            box[1::2] = h*box[1::2].clamp(min=0, max=h)
            box = box.detach().cpu().numpy().astype(np.int32)
            img = cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(0,0,255),1)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 1300, 1600)
        cv2.imshow('image', img) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 

    def run_testing(self):

        """
        Run test with test data split, computes precision, recall and f1 
        score of the prediction results.
        """

        test_folder = self.cfg.log_dir + "test/"
        make_dir(test_folder)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(self.device))
        log_file_path = join(test_folder, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(self.device)

        test_split = TorchDataloader(dataset=self.dataset.get_split('testing'),
                                     preprocess=self.model.preprocess,
                                     transform=self.model.transform
                                    )

        testing_loader = DataLoader(
            test_split,
            batch_size=1,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        self.load_ckpt()
        self.model.eval()

        log.info("Started testing")
        prediction = []
        target = []

        with torch.no_grad():
            for data in tqdm(testing_loader, desc='testing'):
            
                data.to(self.device)
                # If Pipeline should process batches of various image size, priors feature maps needs to adapt according
                # to this size. Head image_size parameter is updated with every batch, so detection head is able
                # to adapt for various image size. This update is performed here within a pipeline, when image batch is
                # formed for forward pass.
                self.model.image_size = data.images.shape[-2:]

                results = self.model(data)
                boxes_batch = self.model.inference_end(results)

                target.extend([self.transform_for_metric(self.transform_input_batch(boxes,  labels=labels)) 
                            for boxes, labels in zip(data.boxes, data.labels)])

                prediction.extend([self.transform_for_metric(boxes) for boxes in boxes_batch])

        # precision, recall metrics evaluation for epoch on validation split
        precision, recall = self.ME.evaluate(prediction,
                                             target,
                                             self.model.classes_ids,
                                             self.cfg.get("overlaps", [0.5]))
        
        log.info("")
        log.info(f' {" ": <3} "==== Precision ==== Recall ==== F1 ====" ')
        desc = ''
        for i, c in enumerate(self.model.classes):

            p = precision[i,0]
            rec = recall[i,0]
            f1 = 2*p*rec/(p+rec)

            desc += f' {"{}".format(c): <9} {"{:.2f}".format(p): <14} {"{:.2f}".format(rec): <11} {"{:.2f}".format(f1)}'
            log.info(desc)

        precision = np.mean(precision[:, -1])
        recall = np.mean(recall[:, -1])
        f1 = 2*precision*recall/(precision+recall)

        log.info("")
        log.info("Overall_precision: {:.2f}".format(precision))
        log.info("Overall_recall: {:.2f}".format(recall))
        log.info("Overall_F1: {:.2f}".format(f1))

        precision = float(precision)
        recall = float(recall)
        f1 = float(f1)
        
        test_protocol = {
            '0_model': self.cfg.get('model_name', None),
            '1_model_version':self.cfg.get('resume_from', None), 
            '2_dataset': self.cfg.get('dataset_name', None), 
            '3_date': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), 
            '4_precision': precision, 
            '5_recall': recall, 
            '6_f1': f1,
                    }

        with open(test_folder + 'test_protocol.yaml', 'w') as outfile:
            yaml.dump(test_protocol, outfile)

    def run_validation(self):
        """
        Validation session with validation data split, computes
        precision, recall and the loss of the prediction results.
        """

        # Model in evaluation mode -> no gradient = parameters are not optimized
        self.model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        batcher = ConcatBatcher(self.device)

        valid_dataset = self.dataset.get_split('validation')

        valid_split = TorchDataloader(dataset=valid_dataset,
                                      preprocess=self.model.preprocess,
                                      transform=self.model.transform
                                     )

        validation_loader = DataLoader(
            valid_split,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        log.info("Started validation")

        self.valid_losses = {}

        prediction = []
        target = []

        with torch.no_grad():
            for data in tqdm(validation_loader, desc='validation'):

                data.to(self.device)

                results = self.model(data)
                loss = self.model.loss(results, data)

                for l, v in loss.items():
                    if l not in self.valid_losses:
                        self.valid_losses[l] = []
                        self.valid_losses[l].append(v.cpu().numpy())

                
                boxes_batch = self.model.inference_end(results)

                 # convert Input and Output for metrics evaluation    
                target.extend([self.transform_for_metric(self.transform_input_batch(boxes, labels=labels)) 
                               for boxes, labels in zip(data.boxes, data.labels)])

                prediction.extend([self.transform_for_metric(boxes) for boxes in boxes_batch])

        # Process bar data feed
        sum_loss = 0
        desc = "validation - "
        for l, v in self.valid_losses.items():
            desc += " %s: %.03f" % (l, np.mean(v))
            sum_loss += np.mean(v)

        desc += " > loss: %.03f" % sum_loss
        log.info(desc)

        # mAP metric evaluation for epoch over all validation data
        precision, recall = self.ME.evaluate(prediction,
                                             target,
                                             self.model.classes_ids,
                                             self.cfg.get("overlaps", [0.5]))

        log.info("")
        log.info(f' {" ": <3} "==== Precision ==== Recall ==== F1 ====')
        desc = ''

        for i, c in enumerate(self.model.classes):

            p = precision[i,0]
            rec = recall[i,0]
            f1 = 2*p*rec/(p+rec)

            desc += f' {"{}".format(c): <9} {"{:.2f}".format(p): <14} {"{:.2f}".format(rec): <11} {"{:.2f}".format(f1)}'
            log.info(desc)

        precision = np.mean(precision[:, -1])
        recall = np.mean(recall[:, -1])
        f1 = 2*precision*recall/(precision+recall)

        log.info("")
        log.info("Overall_precision: {:.2f}".format(precision))
        log.info("Overall_recall: {:.2f}".format(recall))
        log.info("Overall_F1: {:.2f}".format(f1))
        
        self.valid_losses["precision"] = precision
        self.valid_losses["recall"] = recall
        self.valid_losses["f1"] = f1

        return self.valid_losses


    def run_training(self):

        if not os.path.exists(self.cfg.log_dir + 'process_config.json'):
            with open(self.cfg.log_dir + 'process_config.json', "w") as outfile:
                json.dump(dict(self.cfg_dict), outfile)

        torch.manual_seed(self.rng.integers(np.iinfo(np.int32).max))  # Random reproducible seed for torch

        log.info("DEVICE : {}".format(self.device))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(self.cfg.log_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(self.device)
        
        """Run training with train data split."""
        train_dataset = self.dataset.get_split('training')

        train_split = TorchDataloader(dataset=train_dataset,
                                      preprocess=self.model.preprocess,
                                      transform=self.model.transform
                                     )

        self.optimizer, self.scheduler = self.model.get_optimizer(self.cfg.optimizer)

        start_ep, _ = self.load_ckpt()

        if os.path.exists(self.cfg.log_dir + '/training_record.csv'):
            training_record = pd.read_csv(self.cfg.log_dir + '/training_record.csv', index_col=False)
        else:
            training_record = pd.DataFrame([],columns=['epoch', 'precision', 'recall', 'f1', 'loss_sum'])

        log.info("Started training")
        
        for epoch in range(start_ep+1, self.cfg.max_epoch + 1):

            log.info(f'================================ EPOCH {epoch:d}/{self.cfg.max_epoch:d} ================================')
            self.model.train()
            self.losses = {}

            train_loader = DataLoader(
            train_split,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
                                    )
        
            process_bar = tqdm(train_loader, desc='training')

            for data in process_bar:

                data.to(self.device)
                results = self.model(data)
                loss = self.model.loss(results, data)

                loss_sum = sum([self.cfg.loss.cls_weight * loss['loss_cls'],
                                self.cfg.loss.loc_weight * loss['loss_box']
                               ]) 
 
                self.optimizer.zero_grad()
                loss_sum.backward()
                     
                self.optimizer.step()

                desc = "training - "
                for l, v in loss.items():
                    if l not in self.losses:
                        self.losses[l] = []
                    self.losses[l].append(v.cpu().detach().numpy())
                    desc += " %s: %.03f" % (l, v.cpu().detach().numpy())

                desc += " > loss: %.03f" % loss_sum.cpu().detach().numpy()
                process_bar.set_description(desc)
                process_bar.refresh()


            if os.path.exists(self.cfg.log_dir + '/metrics.npy'):
                metrics = np.load(self.cfg.log_dir + '/metrics.npy')
                best_f1 = metrics[2]
            else:
                best_f1 = 0

            # Execute validation session inside of the training session with given frequency
            if (epoch % self.cfg.get("validation_freq", 1)) == 0:

                metrics = self.run_validation()
                loss_total = metrics['loss_cls'][0] + metrics['loss_box'][0]

                training_record.loc[epoch] = [epoch, metrics['precision'], metrics['recall'], metrics['f1'],
                                              loss_total]

                actual_f1 = metrics['f1']
                
                if actual_f1 > best_f1:

                    best_f1 = actual_f1
                    self.save_ckpt(epoch, save_best=True)
                    np.save(self.cfg.log_dir + '/metrics.npy', 
                            np.array([metrics['precision'], metrics['recall'], metrics['f1'], loss_total]))
                
            if epoch % self.cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch,save_best=False)

            training_record.to_csv(self.cfg.log_dir + '/training_record.csv', index=False)
<h1>2D Object Detection framework</h1>

<h2>Introduction</h2>
The code in this project/repository is designed for training and experimenting with RetinaNet object detection model. Framework expects two-dimensional RGB image as a source data. Framework is configured to train model detecting license plate objects. Dataset prepared for use by this framework is located in <b>smb://oelnas.riegl-gmbh/scandaten/AI_DATA/Object_Detection/2D/License_plates/data/</b>.

<h2>Installation</h2>

<h3>Requirements</h3>

- Installed Anaconda/Miniconda

- Essential python packages:
  - [pytorch==2.0.1](https://pytorch.org/get-started/)
  - [torchvision==0.15.2](https://pytorch.org/vision/stable/index.html)
  - [opencv==4.8.0.74 ](https://pypi.org/project/opencv-python/)
  - [scikit_learn==1.3.0](https://pypi.org/project/scikit-learn/)
  - [numpy==0.57.0](https://numpy.org/)

<h3>Local setup</h3>

```
1. open Anaconda Prompt

2. Download projet to your local machine
 ```git clone git@gitlab.riegl-gmbh:xp/ObjectDetection_2D.git```

3. Go to the project folder:
cd /Path/To/ObjectDetection_2D/

4. Create and activate virtual environment
conda create --name "env_name" python=3.10
conda activate "env_name"

5. Install dependecies with requirements. txt file
pip3 install -r requirements.txt
```

<h2>Execution of framework sessions</h2>

```
conda activate "env_name"
```

1. Run training session
```
python run_training.py
```

2. Run single sample inference session with visualization
```
python show_inference.py
```

3. Run model performance evaluation on testing dataset
```
python run_testing.py
```

<h2>Framework structure</h2>

-	<b>dataset</b> – contains modules for data preprocessing and importing data to Pytorch framework for training/validation/testing session
    -	<b>base_dataset.py</b> - Base dataset class, all datasets must inherit from this class to be compatible with pipeline. 
    -	<b>ImageDataset.py</b> - Imports data for given data split (training/validation/testing) and defines how single sample (image+annotation) should be imported.
    -	<b>dataloaders.py</b> – Custom dataloader which forms images and annotations into batches as an input for model.
    -	<b>dataset_transformers.py</b> – Collection of transformers for public datasets mentioned above, to unified their format and merge them into one dataset.
-	<b>augment</b> – module is responsible for augmentation applied on single image during training process.
    -	<b>augmentation.py</b> – Class implementing set of custom image augmentation techniques, which (are/can be) applied during training process.
    -	<b>ops.py</b> -  Collection of operations used for image augmentation.
-	model – Contains modules defining RetinaNet structure and all methods for network training and inference.
    -	<b>base_model.py</b> – Base model class, all models must inherit from this class to be compatible with pipeline.
    -	<b>MobileNet.py</b> – Architecture of MobileNetV1 classifier (used as backbone of RetinaNet).
    -	<b>RetinaNet.py</b> – Architecture of RetinaNet detector with methods for training and inference.
    -	<b>utils.py</b> – Collection of function for target encoding/decoding, anchor generator and NMS postprocessing.
-	<b>pipeline</b> – Object detection pipeline defines process of training, testing and model prediction visualization.
    -	<b>base_pipeline.py</b> -  Base pipeline class, all pipelines must inherit from this class to be compatible with pipeline. Determines log directory of training/inference session.
    -	<b>pipeline.py</b> – Object Detection pipeline defines logic of training, validation, testing, inference session. It uses dataset  module to introduce data to model, performs required actions and formats model output.
    -	<b>metrics.py</b> – Module evaluates precision and recall based on model prediction and ground truth for given data batch.
    -	<b>utils.py</b> - Collection of function used in Object Detection pipeline.
-	<b>weights</b> – Contains mobilenet.tar file of pretrained weights for MobileNet backbone.
-	<b>config.py</b> – Config class in this module is responsible for correct parsing of config.yaml which contains dataset, model, pipeline parameters
-	<b>run_training.py</b> – Starts training session (new one or resumed one).
-	<b>run_testing.py</b> – Evaluates performance of trained model on testing dataset.
-	<b>show_inference.py</b> – Performs prediction with trained model on single image and visualize it.

<h2>Framework configuration file <b>config.yaml</b></h2>

-	global_args – parameters shared globally through the whole framework
    -	device - Device used by framework (cpu or cuda)
    -	seed - The seed is used to initialize the random number generator, to control reproducibility of model training.
    -	output_path - Directory, where pipeline output will be stored.
-	dataset – parameters used in dataset  module
    -	dataset_path - Directory, where training/validation/testing split folders are located.
-	model - parameters used in model module
    -	model_name - Name of used model.
    -	backbone - Defines which backbone architecture should be used (mobilenet or resnet)
    -	pretrained - Defines if model training should start with already pretrained weights of backbone or it will be randomly initialized.
    -	classes - list of object names (strings), defining which categories model detects
    -	mobilenet - Parameters used for training and validation for model with MobileNet backbone architecture.
        -	batch_size - Batch size used by dataloader.
        -	image_size – Image size used in preprocessing/augmentation stage.
        -	return_layers – Dictionary defining which items of backbone dictionary are returned as feature maps to Retina network. There are 3 feature maps.
        -	in_channel - Input channel number of backbone.
        -	out_channel - Output channel number of backbone.
    - resnet - Parameters used for training and validation for model with Resnet50 backbone architecture.
        -	batch_size - Batch size used by dataloader.
        -	image_size – Image size used in preprocessing/augmentation stage.
        -	return_layers – Dictionary defining which items of backbone dictionary are returned as feature maps to Retina network. There are 3 feature maps.
        -	in_channel - Input channel number of backbone.
        -	out_channel - Output channel number of backbone.
    - head – Parameters defining shape of anchors used for detection, thresholds for NMS and inference procedure.  
        -	steps - spacing/stride (in pixels) of anchors for corresponding feature maps used for anchors generation
        -	aspects – list of anchor aspect ratios used for anchors generation, it is shared for all feature maps  
        -	sizes – list of sizes (in pixels) of anchors for corresponding feature maps used for anchors generation
        -	variances – normalizing factors used for encoding/decoding between anchors and ground truth targets
        -	negpos_ratio - Defines how many negative samples is selected for 1 positive sample and introduced to loss function during training procedure.
        -	nms_top_k -  Defines K anchors with confidence score, which are considered as model prediction.
        -	nms_thresh – Maximal allowed IOU between predicted bounding boxes. If IOU is higher, bounding box with higher confidence score is kept for prediction. 
        - score_thresh – Minimal confidence score of anchor to be considered for prediction. Anchors with lower score can’t be predicted by model.
        - Iou_thr – Defines IOU threshold between predicted bounding box and ground truth bounding box, which distinguish between positive and negative matches. If anchor has IOU at least on this level with any ground truth bounding box, than it is considered as positive sample.
-	augment – Dictionary which defines parameters used in augmentation module.
-	pipeline – parameters used in pipeline module
    -	inference_mode (True/False) – Determines how is the model initialized for given session. Should be False for training session and True for any inference session.
    -	is_resume – Determines if training session should initialize new training session (False) or to continue from previously initialized training session (True)
    -	resume_from – Defines which training session is used for inference or resumed training. It is name of folder in form of timestamp, which was generated by pipeline during training session. Checkpoint having best performance (f1) score on validation dataset is used for inference or resumed training.
    -	max_epoch – defines maximal number of training epochs performed during given training session
    -	validation_freq – Defines frequency of validation session execution during training session. Validation is executed after given number of training epochs.
    -	save_ckpt_freq - Defines frequency of exporting model state. Export is executed after given number of training epochs.
    -	num_workers – number of workers used for parallel data batch import in Dataloader
    -	overlaps – IOU threshold (can be specific for each class for multiclass detection) used for Precision and Recall evaluation. Determines if predicted bounding box is True or False positive.
        -	loss – Parameters used for loss computation.
            -	cls_weight – normalizing factor for classification loss
            -	loc_weight – normalizing factor for bounding box regression loss
        -	optimizer – Parametrs of SGD optimizer.  
            -	lr – learning rate
            -	momentum
            -	weight_decay

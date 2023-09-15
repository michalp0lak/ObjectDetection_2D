import logging
from dataset.ImageDataset import Dataset
from model.RetinaNet import RetinaDetector
from pipeline.pipeline import ObjectDetection
from config import Config

def main():

    cfg = Config.load_from_file('./config.yaml')
    cfg_dict = cfg.dump()

    if (cfg.global_args and cfg.pipeline and cfg.model and cfg.dataset) is None:
            raise ValueError("Please specify global arguments, pipeline, model, and dataset in config file")

    cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = Config.initialize_cfg_file(cfg)
    dataset = Dataset(**cfg_dict_dataset)
    model = RetinaDetector(**cfg_dict_model)
    pipeline = ObjectDetection(model, dataset, cfg_dict, **cfg_dict_pipeline)
    
    if not cfg_dict_pipeline.inference_mode:
        pipeline.run_training()
    else:
         raise ValueError("Can't run training session with configuration of inference_mode: True")
    
if __name__ == '__main__':

    logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',)

    main()
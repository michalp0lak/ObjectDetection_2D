import logging
from dataset.ImageDataset import Dataset
from model.RetinaNet import RetinaDetector
from pipeline.pipeline import ObjectDetection
from config import Config

def main():

    cfg = Config.load_from_file('./config.yaml')
    global_cfg = cfg.dump()
    
    if (cfg.global_args and cfg.pipeline and cfg.model and cfg.dataset) is None:
        raise ValueError("Please specify global arguments, pipeline, model, and dataset in config file")
    
    cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = Config.initialize_cfg_file(cfg)

    dataset = Dataset(**cfg_dict_dataset)
    model = RetinaDetector(**cfg_dict_model)
    pipeline = ObjectDetection(model, dataset, global_cfg, **cfg_dict_pipeline)

    if cfg_dict_pipeline.get('inference_mode'):
        pipeline.run_testing()

if __name__ == '__main__':

    logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',)

    main()
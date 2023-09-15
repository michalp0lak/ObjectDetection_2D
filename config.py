import os
from addict import Dict
import yaml

class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

class Config(object):

    def __init__(self, cfg_dict=None):

        if cfg_dict is None:
            cfg_dict = dict()

        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict should be a dict, but'
                            f'got {type(cfg_dict)}')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))

        self.cfg_dict = cfg_dict

    def dump(self, *args, **kwargs):
        """Dump to a string."""

        def convert_to_dict(cfg_node, key_list):
            if not isinstance(cfg_node, ConfigDict):
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict

        self_as_dict = convert_to_dict(self._cfg_dict, [])
        
        return self_as_dict

    @staticmethod
    def initialize_cfg_file(cfg):

        """
            Merge args and extra_dict from the input arguments.
        """
    
        if (cfg.global_args is None) | ('device' not in cfg.global_args):
            cfg.global_args.device = 'cpu'

        cfg.dataset.seed = cfg.global_args.seed
        cfg.pipeline.seed = cfg.global_args.seed
        cfg.model.seed = cfg.global_args.seed
        
        cfg.model.device = cfg.global_args.device

        cfg.pipeline.device = cfg.global_args.device
        cfg.pipeline.log_dir = cfg.global_args.output_path
        cfg.pipeline.model_name = cfg.model.model_name

        cfg.model.augment['image_size'] = cfg.model[cfg.model['backbone']]['image_size']
        cfg.model.head['image_size'] = cfg.model[cfg.model['backbone']]['image_size']
        cfg.pipeline['batch_size'] = cfg.model[cfg.model['backbone']]['batch_size']   
 
        return cfg.dataset, cfg.pipeline, cfg.model

    @staticmethod
    def load_from_file(filename):
        
        if filename is None:
            raise FileExistsError("Config file is not defined")
        
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} not found')

        if not (filename.endswith('.yaml') or filename.endswith('.yml') or filename.endswith('.json')):
            raise ImportError('Config file has to yaml, yml or json file')

        else: 
            with open(filename) as f: cfg_dict = yaml.safe_load(f)

        return Config(cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)
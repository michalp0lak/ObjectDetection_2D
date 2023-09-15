from os.path import exists
from os import makedirs

def convert_device_name(framework):
    """Convert device to either cpu or cuda."""
    gpu_names = ["gpu", "cuda"]
    cpu_names = ["cpu"]
    if framework not in cpu_names + gpu_names:
        raise KeyError("the device should either "
                       "be cuda or cpu but got {}".format(framework))
    if framework in gpu_names:
        return "cuda"
    else:
        return "cpu"


def make_dir(folder_name):
    """Create a directory.
    If already exists, do nothing
    """
    if not exists(folder_name):
        makedirs(folder_name)
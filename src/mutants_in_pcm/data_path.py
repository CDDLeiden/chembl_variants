import os

import pystow


data_dir = pystow.module('data-mutants-in-pcm').base.as_posix()


def set_data_path(path: str) -> None:
    """Set the default data directory for the mutant_in_pcm's analysis

    :param path: path to the default data directory; if None, defaults to PyStow's home
    """
    if path is not None and not os.path.exists(path):
        raise ValueError('The data directory does not exist.')
    elif path is None:
        path = pystow.module('data-mutants-in-pcm').base.as_posix()
    global data_dir
    data_dir = os.path.abspath(path)


def get_data_path() -> str:
    global data_dir
    if data_dir is None or not os.path.exists(data_dir):
        raise ValueError('The data directory does not exist. Please check the path in data_path.py')
    else:
        return os.path.abspath(data_dir)

# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike

class DType:
    pass

class Device:
    def __init__(device_id):
        pass

    def compile_cu(self, cu_code, func_name):
        """ """
        pass

    def compile_ptx(self, ptx_code, func_name):
        """ """
        pass

    def slice(self, vs):
        """ """
        pass

    def synchronize(self):
        """ """
        pass

    def zeros(self, len):
        """ """
        pass

class Func:
    def launch3(self, s1, s2, s3, block_dim=1, grid_dim=1, shared_mem_bytes=0):
        """ """
        pass

class Slice:
    def len(self):
        """ """
        pass

    def to_vec(self):
        """ """
        pass

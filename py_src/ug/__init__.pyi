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
    def launch3(self, s1, s2, s3, grid_size=1):
        """ """
        pass

class Instr:
    @staticmethod
    def binary(op, lhs, rhs, dtype):
        """ """
        pass

    @staticmethod
    def const_f32(v):
        """ """
        pass

    @staticmethod
    def const_i32(v):
        """ """
        pass

    @staticmethod
    def define_acc_f32(v):
        """ """
        pass

    @staticmethod
    def define_acc_i32(v):
        """ """
        pass

    @staticmethod
    def define_global(index, dtype):
        """ """
        pass

    @staticmethod
    def end_range(start_idx):
        """ """
        pass

    @staticmethod
    def load(src, offset, dtype):
        """ """
        pass

    @staticmethod
    def range(lo, up, end_idx):
        """ """
        pass

    @staticmethod
    def special_g():
        """ """
        pass

    @staticmethod
    def special_l():
        """ """
        pass

    @staticmethod
    def store(dst, offset, value, dtype):
        """ """
        pass

    @staticmethod
    def unary(op, arg, dtype):
        """ """
        pass

class Kernel:
    def __init__(instrs):
        pass

    def cuda_code(self, name):
        """ """
        pass

    def to_list(self):
        """ """
        pass

class Slice:
    def len(self):
        """ """
        pass

    def to_vec(self):
        """ """
        pass

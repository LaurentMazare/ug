# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike

class Arg:
    @staticmethod
    def i32():
        """ """
        pass

    @staticmethod
    def ptr_f32():
        """ """
        pass

    @staticmethod
    def ptr_i32():
        """ """
        pass

class Expr:
    @staticmethod
    def load(ptr, offset, len, stride):
        """ """
        pass

class IndexExpr:
    @staticmethod
    def cst(v):
        """ """
        pass

    @staticmethod
    def program_id():
        """ """
        pass

class Kernel:
    def __init__(name, args, ops):
        pass

    def lower(self):
        """ """
        pass

class Ops:
    @staticmethod
    def store(dst, offset, len, stride, value):
        """ """
        pass

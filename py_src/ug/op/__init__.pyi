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

class Ast:
    def exp(self):
        """ """
        pass

    @staticmethod
    def f32(v):
        """ """
        pass

    @staticmethod
    def i32(v):
        """ """
        pass

    def max(self, axis):
        """ """
        pass

    def min(self, axis):
        """ """
        pass

    def prod(self, axis):
        """ """
        pass

    def shape(self):
        """ """
        pass

    def sum(self, axis):
        """ """
        pass

class Kernel:
    def __init__(name, args, ops):
        pass

    def lower(self):
        """ """
        pass

class Store:
    pass

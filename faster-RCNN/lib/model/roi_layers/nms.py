# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
import torch
from model import _C

nms = _C.nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""

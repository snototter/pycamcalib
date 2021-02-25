#!/usr/bin/env python
# coding=utf-8
"""
Python utility package for intrinsic camera calibration using OpenCV and Eddie patterns.
"""

__all__ = ['patterns']
__author__ = 'snototter'

# Load version
import os
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.py')) as vf:
    exec(vf.read())

#!/usr/bin/env python
# coding=utf-8
"""
Python utility package for intrinsic camera calibration.
"""

__all__ = ['patterns', 'preproc']
__author__ = 'snototter'

# Load version
import os
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.py')) as vf:
    exec(vf.read())


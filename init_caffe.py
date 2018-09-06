#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

import sys
from config import *
sys.path.insert(0, CAFFEDIR + 'python')
import caffe
caffe.set_mode_gpu()

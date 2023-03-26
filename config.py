
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
from Utils.attr_dict import AttrDict


__C = AttrDict()
cfg = __C


__C.SYSROOT = os.getcwd() #current dir
__C.STARTTIME = None #start time

#dataset
__C.DATASET = AttrDict()
__C.DATASET.IMGEXT = "png"
__C.DATASET.ROOT = os.path.join(__C.SYSROOT, 'workdir6/selects_by_pro_randomcrop')
__C.DATASET.IGNORE_LABEL = 255

#net
__C.NET = AttrDict()
__C.NET.SAVESELFPATH = '/NetSave/'#Save path of the trained network


__C.RUN = AttrDict()
__C.RUN.SHOWINFO = False


__C.DATA = AttrDict()
__C.DATA.FeaturesDataPath = "/ExtractFeaturesByNet"









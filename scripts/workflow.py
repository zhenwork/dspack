import os,sys
import subprocess 
import numpy as np
from numba import jit

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH) 


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname_text","--fname_text", help="workflow file", default=None, type=str) 
parser.add_argument("-fname_json","--fname_json", help="workflow file", default=None, type=str) 
args = parser.parse_args()

"""
- run_import
- run_image
- run_scaling
- run_pca
- run_merging
- run_volume
- run_statistics
"""

args.run_import = scripts.utils.get_process_params(fname_text=args.fname_text,fname_json=args.fname_json,"run_import")
args.run_image = scripts.utils.get_process_params(fname_text=args.fname_text,fname_json=args.fname_json,"run_image")
args.run_scaling = scripts.utils.get_process_params(fname_text=args.fname_text,fname_json=args.fname_json,"run_scaling")
args.run_pca = scripts.utils.get_process_params(fname_text=args.fname_text,fname_json=args.fname_json,"run_pca")
args.run_merging = scripts.utils.get_process_params(fname_text=args.fname_text,fname_json=args.fname_json,"run_merging")
args.run_volume = scripts.utils.get_process_params(fname_text=args.fname_text,fname_json=args.fname_json,"run_volume")
args.run_statistics = scripts.utils.get_process_params(fname_text=args.fname_text,fname_json=args.fname_json,"run_statistics")


if args.run_import.status:
    command = 
if args.run_image.status:
    command = 
if args.run_scaling.status:
    command = 
if args.run_pca.status:
    command = 
if args.run_merging.status:
    command = 
if args.run_volume.status:
    command = 
if args.run_statistics.status:
    command = 
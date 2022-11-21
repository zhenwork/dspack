
###########################################################################
# Import Necessary files into one big file for further processing
###########################################################################
import os,sys
import numpy as np
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-image_file","--image_file", help="crystal diffraction files",default=None,nargs="*")
parser.add_argument("-backg_file","--backg_file", help="no-crystal backg files",default=None,nargs="*")
parser.add_argument("-pdb_file","--pdb_file", help="pdb file",default=None,type=str)
parser.add_argument("-gxparms_file","--gxparms_file", help="xds gxparms file",default=None,type=str)
parser.add_argument("-dials_report_file","--dials_report_file",help="dials report html file",default=None,type=str)
parser.add_argument("-detector_mask_file","--detector_mask_file", help="numpy array file",default=None,type=str)
parser.add_argument("-extra_params","--extra_params", help="extra params json file",default=None,type=str)
parser.add_argument("-fname","--fname",help="save file", default="./data.dsdata",type=str)
args, unknown = parser.parse_known_args()

import scripts.fsystem

#### create folder
folder = os.path.dirname(os.path.realpath(args.fname))
if not os.path.isdir(folder):
    os.makedirs(folder)


#### save the diffraction image
data_save = {}
if args.image_file:
    data_save["image_file"] = args.image_file 
if args.backg_file:
    data_save["backg_file"] = args.backg_file 
if args.gxparms_file:
    data_save["gxparms_file"] = os.path.realpath(args.gxparms_file)
if args.extra_params:
    data_save["extra_params"] = os.path.realpath(args.extra_params)
if args.dials_report_file:
    data_save["dials_report_file"] = os.path.realpath(args.dials_report_file)
if args.detector_mask_file:
    data_save["detector_mask_file"] = os.path.realpath(args.detector_mask_file)
if args.pdb_file:
    data_save["pdb_file"] = os.path.realpath(args.pdb_file)

    
if len(unknown)>0:
    # unknown is a list, in format: x=10 y=20 ...
    for each in unknown:
        key,val = each.split("=")
        key = key.strip()
        val = val.strip()
        data_save[key] = val

#### Save history
scripts.fsystem.PVmanager.modify(data_save, args.fname)
print ("#### Import Done")
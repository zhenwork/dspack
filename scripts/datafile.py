"""
Angstrom (A): wavelength, crystal lattice
"""
import json
import os,sys
import numpy as np 
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.crystal
import scripts.fsystem
import scripts.image


# "../DIALS/G150T-2/dials.report.html"
def dials_report_to_psvm(fname):
    import codecs
    f=codecs.open(fname, 'r')
    content = f.readlines()

    counter = 0
    for line in content:
        if "smoothly-varying scaling term" in line:
            counter += 1
            if counter == 2:
                break
    content = None
    startx = line.index('"x": [')
    stopx = line.index('], "xaxis"')

    xvalue = []
    for data in line[startx+6:stopx].split(","):
        xvalue.append(float(data))

    starty = line.index('"y": [')
    stopy = line.index('], "yaxis"')

    yvalue = []
    for data in line[starty+6:stopy].split(","):
        yvalue.append(float(data))    
    
    return {"dials_scaling_angle_deg":xvalue,"dials_scaling_value":yvalue}


def get_scale_from_dials_expt(dials_scaling_angle_deg,dials_scaling_value,theta_deg):
    x = dials_scaling_angle_deg
    y = dials_scaling_value
    theta = theta_deg
    if x[0] > theta:
        return y[0]
    if x[-1] < theta:
        return y[-1]
    a = np.where(np.array(x)-theta>0)[0][0] - 1
    b = a+1 
    return (y[b] - y[a]) / (x[b] - x[a]) * (theta - x[a]) + y[a]


def gxparms_to_psvm(fileGXPARMS):
    psvmParms = {}

    f = open(fileGXPARMS)
    content = f.readlines()
    f.close()

    psvmParms['pixel_size_mm'] = float(content[7].split()[3])
    psvmParms['pixel_size_um'] = float(content[7].split()[3])*1.0e3
    psvmParms['polarization_fr'] = -1.0
    psvmParms['wavelength_A'] = float(content[2].split()[0])
    psvmParms['angle_step_deg'] = float(content[1].split()[2])
    psvmParms['detector_distance_mm'] = float(content[8].split()[2])
    psvmParms['detector_center_px'] = (float(content[8].split()[0]), float(content[8].split()[1]))

    ## calculate the Amat matrix
    ## GXPARMS saves real space vecA, vecB, vecC, which is invAmat
    invAmat = np.zeros((3,3))
    for i in range(4,7):
        for j in range(3):
            invAmat[i-4,j] = float(content[i].split()[j])
    Amat = np.linalg.inv(invAmat)

    ## calculate B matrix
    ## Bmat is the user's defined coordinates
    (a,b,c,alpha,beta,gamma) = [float(each) for each in content[3].split()[1:]]
    latticeConstant = np.array([a,b,c,alpha,beta,gamma])
    Bmat = scripts.crystal.standardBmat(latticceConstant=(a,b,c,alpha,beta,gamma))
    
    psvmParms["lattice_constant_A_deg"] = latticeConstant
    psvmParms["Amat_invA"] = Amat
    psvmParms["Bmat_invA"] = Bmat
    return psvmParms


def cbf_to_psvm(fileName):
    image, header = scripts.fsystem.CBFmanager.reader(fileName)

    psvmParms = {}
    psvmParms["start_angle_deg"] = header['phi']
    psvmParms["current_angle_deg"] = header['start_angle'] 
    psvmParms["angle_step_deg"] = header['angle_increment'] 
    psvmParms["exposure_time_s"] = header['exposure_time'] 
    psvmParms["wavelength_A"] = header['wavelength'] 
    
    psvmParms["pixel_size_m"] = header['x_pixel_size']
    psvmParms["pixel_size_mm"] = header['x_pixel_size'] * 1.0e3
    psvmParms["pixel_size_um"] = header['x_pixel_size'] * 1.0e6 
    
    psvmParms["thickness_m"] = header['sensor_thickness']
    psvmParms["thickness_mm"] = header['sensor_thickness'] * 1.0e3
    psvmParms["thickness_um"] = header['sensor_thickness'] * 1.0e6
    
    nx = header['pixels_in_x']
    ny = header['pixels_in_y']

    ## Detector is flipped
    if not image.shape == (nx, ny):
        #print "## flip image x/y"
        image=image.T
        if not image.shape == (nx, ny):
            raise Exception("!! Image shape doesn't fit")

    psvmParms["image"] = image 
    return psvmParms


def dials_expt_to_psvm(fexpt): 
    def length(arr):
        return np.linalg.norm(arr)
    def angle(x,y):
        return np.arccos(x.dot(y)/length(x)/length(y)) * 180. / np.pi

    psvmParms = {}
    data = json.load(open(fexpt,"r"))
    psvmParms['thickness_mm'] = data["detector"][0]["panels"][0]["thickness"]
    psvmParms['thickness_um'] = data["detector"][0]["panels"][0]["thickness"]*1.0e3
    psvmParms['absorption_coefficient_invmm'] = data["detector"][0]["panels"][0]["mu"]
    psvmParms['absorption_coefficient_invum'] = data["detector"][0]["panels"][0]["mu"]/1.0e3
    return psvmParms


def h5py_to_psvm(fileName):
    return scripts.fsystem.PVmanager.reader(fileName)

def json_to_psvm(fileName): 
    return scripts.fsystem.YMmanager.reader(fileName)

def numpy_to_psvm(fname):
    return {"data":np.load(fname)}

def reader(filename, fileType=None):
    if filename is None:
        return {}
    if fileType is None:
        fileType = filename.split(".")[-1]
        
    if fileType.lower() in ["gxparms","xds"]:
        return gxparms_to_psvm(filename)
    elif fileType.lower() == "cbf":
        return cbf_to_psvm(filename)
    elif fileType.lower() in ["h5", "h5py"]:
        return h5py_to_psvm(filename)
    elif fileType.lower() in ["json", "js"]:
        return json_to_psvm(filename)
    elif fileType.lower() in ["numpy", "npy"]:
        return numpy_to_psvm(filename)
    elif fileType.lower() in ["dials_expt","expt"]:
        return dials_expt_to_psvm(filename)
    elif fileType.lower() in ["dials_report","html"]:
        return dials_report_to_psvm(filename)
    return {}


def special_params(notation="wtich"):
    #mT = imageTools.MaskTools()
    #mask = mT.valueLimitMask(image, vmin=0.001, vmax=100000)
    #mask = mT.circleMask(size, rmin=40, rmax=None, center=None)
    if notation == "wtich":
        psvmParms = {}
        mask = np.ones((2527,2463)).astype(int)
        mask[1255:1300+2,1235:2463] = 0
        mask[1255:1305+2,1735:2000] = 0
        mask[1255:1310+2,2000:2463] = 0
        psvmParms["mask"] = mask.T
        psvmParms["firMask"] = mask.T
        return psvmParms
    elif notation == "old":
        psvmParms = {}
        mask = np.ones((2527,2463)).astype(int)
        mask[1255:1300,1235:2463] = 0
        mask[1255:1305,1735:2000] = 0
        mask[1255:1310,2000:2463] = 0
        psvmParms["mask"] = mask.T
        psvmParms["firMask"] = mask.T
        return psvmParms
    else:
        print "!! No Special Params"
        return {}
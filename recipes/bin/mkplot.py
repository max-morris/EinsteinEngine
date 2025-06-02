#!/usr/bin/python3
from typing import Tuple, Any, Optional
import os, sys
import matplotlib.pyplot as plt
from kuibit.simdir import SimDir
import kuibit.visualize_matplotlib as viz
import numpy as np
import matplotlib.pyplot as plt

num = np.typing.NDArray[np.float64]

def get_xyz_data(name:str, frame_number:int)->Tuple[Any,int]:
    xyz_data = sim.gridfunctions.xyz[name]
    times = xyz_data.times

    iterations = xyz_data.iterations
    print("number of frames:",len(iterations))
    iteration = iterations[frame_number]
    print("iteration:",iteration)
    print("time:",times[frame_number])

    return xyz_data[iteration], times[frame_number]

def get_slice_z(xyz_data:Any, zindex:Optional[int]=None, resolution:Optional[Tuple[int,int,int]]=None)->Tuple[Any,num,num]:

    if resolution is None:
        for datum in xyz_data.iter_from_finest():
            resolution = datum[2].data.shape
        print(f"Defaulting to resolution: {resolution}")
    assert resolution is not None
    if zindex is None:
        zindex = resolution[0]//2
        print(f"Defaulting to zindex: {zindex}")

    x0 = xyz_data.x0
    x1 = xyz_data.x1
    xyz_data_unif = xyz_data.to_UniformGridData(shape=resolution,
                                        x0=x0,
                                        x1=x1,
                                        resample=True)
    # Coordintes are z, y, x
    slice_z = xyz_data_unif.data[zindex,:,:]
    print(f"Range of slice at {zindex}: {slice_z.min()}, {slice_z.max()}")
    return slice_z, x0, x1


# Create x and y coord objects
def get_xy_coords(xxx:Any, x0:num, x1:num)->Tuple[num, num]:
    xvals = np.linspace(x0[2], x1[2], np.shape(xxx)[1])
    yvals = np.linspace(x0[1], x1[1], np.shape(xxx)[0])
    xcoord, ycoord = np.meshgrid(xvals, yvals)
    return xcoord, ycoord

def usage():
    print("./mkplot.py sim_dir variable frame")
    exit(2)

try:
    datadir = sys.argv[1]
except IndexError as ie:
    usage()

try:
    frame_number = int(sys.argv[3])
except IndexError as ie:
    print("Frame number defaulting to 1")
    frame_number = 1

sim = SimDir(datadir)
print(sim)

try:
    variable = sys.argv[2]
except IndexError as ie:
    usage()

sim_data, sim_time = get_xyz_data(variable, frame_number)
slice_z, x0, x1 = get_slice_z(sim_data)
xcoord, ycoord = get_xy_coords(slice_z, x0, x1)

plt.pcolormesh(xcoord, ycoord, slice_z)
plt.title(f"{variable} at time %.5g" % sim_time)
fname = variable + ("%04d.png" % frame_number)
print("Output filename:", fname)
plt.savefig(fname)

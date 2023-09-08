import numpy as np
import time
import argparse
from bigfile import File
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import MAS_library as MASL

parser = argparse.ArgumentParser()

parser.add_argument('FastPMPath', type=str, help='The path to load in FastPM particles.')
parser.add_argument('--Nmesh', type=int, help='Number of particles per side.')
parser.add_argument('--BoxSize', type=float, default=205., help='Size of simulation box.')
parser.add_argument('--Npixels', type=int, help='Size of output image.')

args = parser.parse_args()

# 2D overdensity params
plane = 'XY'
slicew = 20. # Thickness of slice for 2D projection wrt plane
offset = 0. # Offset for slice (who cares)

try:
    Nmesh = int(args.FastPMPath.split('Nmesh')[1].split('_')[0])
except:
    if args.Nmesh:
        Nmesh = args.Nmesh
    else:
        raise Exception("Something wrong with filename and Nmesh argument missing.")

# Image params
if args.Npixels:
    Npixels = args.Npixels
    fout = f'fastpmDMoverdensity{Npixels}_upsample{Nmesh}.png'
    print("Upsampling true.")
else:
    Npixels = Nmesh
    fout = f'fastpmDMoverdensity_Nmesh{Nmesh}.png'

min_overdensity = 0.5     # Minimum overdensity to plot
max_overdensity = 50.0    # Maximum overdensity to plot
scale           = 'log'   # 'linear' or 'log'
cmap            = 'viridis'

print("Importing positions...")
X = File(args.FastPMPath)['Position']
X = np.array(X[0:X.size]).astype(np.float32)
X = X % args.BoxSize

print("Positions imported. Generating overdensity...")

# ADAPTED FROM https://github.com/franciscovillaescusa/Pylians/blob/master/library/plotting_library.py
xmin = 0.
xmax = args.BoxSize
ymin = 0.
ymax = args.BoxSize
zmin = offset
zmax = slicew
BoxSize_slice = args.BoxSize

# Overdensity definition
overdensity = np.zeros((Npixels, Npixels), dtype=np.float32)

# Total mass (unit masses for each particle)
total_mass = len(X)

# Keep only with the particles in the slice
indexes = np.where((X[:, 0]>xmin) & (X[:, 0]<xmax) &
                    (X[:, 1]>ymin) & (X[:, 1]<ymax) &
                    (X[:, 2]>zmin) & (X[:, 2]<zmax) )
X = X[indexes]

# Renormalize positions
X[:, 0] -= xmin;  X[:, 1] -= ymin;  X[:, 2] -= zmin

# Project particle positions into a 2D plane
plane_dict = {'XY':[0, 1], 'XZ':[0, 2], 'YZ':[1, 2]}
X = X[:, plane_dict[plane]]

slice_mass = len(X)

# Compute overdensity somehow
MASL.MA(X, overdensity, BoxSize_slice, MAS='PCS', W=None, renormalize_2D=True)

print(f"Expected mass = {slice_mass}, computed mass = {np.sum(overdensity)}.")

# Mean density in the whole box
mass_density = total_mass/args.BoxSize**3
# Volume of each cell in the density field slice
V_cell = BoxSize_slice**2*slicew/Npixels**2
# Mean mass in each cell of the slice
mean_mass = mass_density*V_cell

# Normalize overdensity
overdensity /= mean_mass

# in our convention overdensity(x,y), while for matplotlib is
# overdensity(y,x), so we need to transpose the field (alr)
overdensity = np.transpose(overdensity)

print("Overdensity generated. Creating figure...")

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_xlim([0, args.BoxSize])
ax1.set_ylim([0, args.BoxSize])

ax1.set_xlabel(r'$h^{-1}{\rm Mpc}$',fontsize=10)
ax1.set_ylabel(r'$h^{-1}{\rm Mpc}$',fontsize=10)

if min_overdensity==None:  min_overdensity = np.min(overdensity)
if max_overdensity==None:  max_overdensity = np.max(overdensity)

overdensity[np.where(overdensity<min_overdensity)] = min_overdensity

if scale=='linear':
      cax = ax1.imshow(overdensity, cmap=plt.get_cmap(cmap), origin='lower',
		       extent=[0, args.BoxSize, 0, args.BoxSize], interpolation='bicubic',
                       vmin=min_overdensity, vmax=max_overdensity)
else:
      cax = ax1.imshow(overdensity, cmap=plt.get_cmap(cmap), origin='lower',
                       extent=[0, args.BoxSize, 0, args.BoxSize], interpolation='bicubic',
                       norm = LogNorm(vmin=min_overdensity, vmax=max_overdensity))

cbar = fig.colorbar(cax)
cbar.set_label(r"$\rho/\bar{\rho}$",fontsize=10)
plt.savefig(fout, bbox_inches='tight')
plt.close(fig)

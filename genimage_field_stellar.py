from nbodykit.lab import BigFileMesh
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path of the field.')
parser.add_argument('--BoxSize', type=float, default=205., help='Size of simulation box.')
# NO UPSAMPLING SUPPORTED FOR THE TIME BEING
args = parser.parse_args()

try:
    Nmesh = int(args.path.split('Nmesh')[1].split('_')[0])
except:
    raise Exception("Something wrong with path.")

fout = f'LDLstellar{Nmesh}.png'
plane = 'XY'
offset = 0.
slicew = 20.
scale = 'log'
cmap = 'viridis'
min_overdensity = 0.1
max_overdensity = 100.

print("Importing data...")

bfm = BigFileMesh(args.path, 'Field')

print("Done. Generating overdensity...")

overdensity = bfm.preview()
start = int(offset / args.BoxSize * Nmesh)
end = start + int(slicew / args.BoxSize * Nmesh) + 1
indices = [slice(Nmesh)]*3
axisdict = {'XY': 2, 'XZ': 1, 'YZ': 0}
axnum = axisdict[plane]
indices[axnum] = slice(start, end)
indices = tuple(indices)
avg_mass = np.sum(overdensity) / Nmesh**3 
overdensity = overdensity[indices].sum(axis=axnum)/(end-start) # 2D overdensity (IDK WHY NORMLIZATION LIKE THAT)
overdensity /= avg_mass
overdensity = np.transpose(overdensity)
print(np.max(overdensity), np.min(overdensity), np.mean(overdensity))

overdensity *= 1e9
min_overdensity *= 1e9
max_overdensity *= 1e9
print("Overdensity generated, creating figure...")

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_xlim([0, args.BoxSize])  #set the range for the x-axis
ax1.set_ylim([0, args.BoxSize])  #set the range for the y-axis

ax1.set_xlabel(r'$h^{-1}{\rm Mpc}$',fontsize=10)  #x-axis label
ax1.set_ylabel(r'$h^{-1}{\rm Mpc}$',fontsize=10)  #y-axis label

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

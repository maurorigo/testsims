import numpy as np
import plotting_library as PL
from matplotlib import pylab
from pylab import *
from matplotlib.colors import LogNorm

#snapshot name
snapshot = '/scratch/mrigo/SIMULATIONS/TNG/TNG300-3/snapdir_099/snap_099'

# density field parameters
x_min, x_max = 0.0, 205
y_min, y_max = 0.0, 205
z_min, z_max = 0.0, 20.
grid         = 256
ptypes       = [4]   # 0-Gas, 1-CDM, 2-NU, 4-Stars; can deal with several species
plane        = 'XY'  #'XY','YZ' or 'XZ'
MAS          = 'CIC' #'NGP', 'CIC', 'TSC', 'PCS'
save_df      = False #whether save the density field into a file

# image parameters
fout            = f'TNG300-3stellar{grid}.png'
min_overdensity = 0.1      #minimum overdensity to plot
max_overdensity = 100.0    #maximum overdensity to plot
scale           = 'log' #'linear' or 'log'
cmap            = 'viridis'


# compute 2D overdensity field
dx, x, dy, y, overdensity = PL.density_field_2D(snapshot, x_min, x_max, y_min, y_max,
                                                z_min, z_max, grid, ptypes, plane, MAS, save_df)

print(np.max(overdensity), np.min(overdensity), np.mean(overdensity))
overdensity *= 1e10
min_overdensity *= 1e10
max_overdensity *= 1e10
# plot density field
print('\nCreating the figure...')
fig = figure()    #create the figure
ax1 = fig.add_subplot(111)

ax1.set_xlim([x, x+dx])  #set the range for the x-axis
ax1.set_ylim([y, y+dy])  #set the range for the y-axis

ax1.set_xlabel(r'$h^{-1}{\rm Mpc}$',fontsize=10)  #x-axis label
ax1.set_ylabel(r'$h^{-1}{\rm Mpc}$',fontsize=10)  #y-axis label

if min_overdensity==None:  min_overdensity = np.min(overdensity)
if max_overdensity==None:  max_overdensity = np.max(overdensity)

overdensity[np.where(overdensity<min_overdensity)] = min_overdensity

if scale=='linear':
      cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                       extent=[x, x+dx, y, y+dy], interpolation='bicubic',
                       vmin=min_overdensity,vmax=max_overdensity)
else:
      cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                       extent=[x, x+dx, y, y+dy], interpolation='bicubic',
                       norm = LogNorm(vmin=min_overdensity,vmax=max_overdensity))

cbar = fig.colorbar(cax)
cbar.set_label(r"$\rho/\bar{\rho}TNG$",fontsize=10)
savefig(fout, bbox_inches='tight')
close(fig)

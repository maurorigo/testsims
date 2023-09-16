import numpy as np
import h5py
from mpi4py import MPI
import os
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileMesh, FieldMesh
import argparse
import sys
import logging

""" This program imports data from TNG snapshot and paints density fields with pmesh utils.
    kind is what to paint ('dm': DM density, 'Mstar' for stellar mass, 'MstarVz', 'ne', 'nT', 'nHI', 'neVz' idk for the time being)
    If float32 is True, load any float64 datatype arrays directly as float32 (save memory).
"""

parser = argparse.ArgumentParser()

parser.add_argument('TNGPath', type=str, help='The path to load in TNG particles.')
parser.add_argument('snapNum', type=int, help='Snapshot number.')
parser.add_argument('nChunks', type=int, help='Number of chunks for snapshot.')
parser.add_argument('kind', type=str, help='what to paint (''dm'': DM density, ''Mstar'' for stellar mass, ''MstarVz'', ''ne'', ''nT'', ''nHI'', ''neVz'' idk for the time being)')
parser.add_argument('Nmesh', type=int, help='Number of particles per side.')
parser.add_argument('--float64', dest='float32', action='store_false')
parser.set_defaults(float32=True)
parser.add_argument('--BoxSize', type=float, default=205., help='Size of simulation box.')

args = parser.parse_args()

pm = ParticleMesh(Nmesh=[args.Nmesh]*3, BoxSize=args.BoxSize, resampler='cic')
comm = pm.comm

if comm.rank==0:
    logger = logging.getLogger()
    hdlr = logging.FileHandler(f'mapgeneration.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    ch = logging.StreamHandler()
    logger.addHandler(hdlr) 
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)


# 0: gas, 1: dm, 3: tracers, 4: stars/wind, 5: bhs
if args.kind == 'dm':
    gName = "PartType1" # To be consistent with original TNG import lib
    ptNum = 1
elif args.kind in ['Mstar', 'MstarVz']:
    gName = "PartType4"
    ptNum = 4
elif args.kind in ['ne', 'nT', 'nHI', 'neVz']:
    gName = "PartType0"
    ptNum = 0
else:
    raise Exception("Unkown required map type " + args.kind)


def snapPath(basepath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basepath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath

# Make the offset table (by type) for the snapshot files, to be able to quickly determine within which file(s) a given offset+length will exist.
# I guess this contains, for each ptype and in each chunk, the starting index of the particles of that type 
snapOffsets = np.zeros((6, args.nChunks), dtype='int64') # This does it for all particle types, a bit useless but who cares tbh
for i in range(1, args.nChunks):
    f = h5py.File(snapPath(args.TNGPath, args.snapNum, i-1), 'r')
    for j in range(6):
        snapOffsets[j, i] = snapOffsets[j, i-1] + f['Header'].attrs['NumPart_ThisFile'][j] # Index in prev chunk + num parts in prev chunk
    f.close()


def scalefactor():
    with h5py.File(snapPath(args.TNGPath, args.snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())
        z = header['Redshift']
        a = 1. / (1. + z)
    return a


def paintTNGmap():

    # Require fields to load (then I'll add more for the different maps)
    fields = ['Coordinates'] # Should load all coordinates at once (IF NOT, SAD)
    mdi = [None]

    if args.kind == 'Mstar':
        fields.append('Masses')
        mdi.append(None)
    elif args.kind == 'ne':
        fields.append('Masses')
        mdi.append(None)
        fields.append('ElectronAbundance')
        mdi.append(None)
    elif args.kind == 'nT':
        fields.append('ElectronAbundance')
        mdi.append(None)
        fields.append('InternalEnergy')
        mdi.append(None)
        fields.append('Masses')
        mdi.append(None)
    elif args.kind == 'nHI':
        fields.append('Masses')
        mdi.append(None)
        fields.append('NeutralHydrogenAbundance')
        mdi.append(None)
    elif args.kind == 'neVz':
        fields.append('Masses')
        mdi.append(None)
        fields.append('ElectronAbundance')
        mdi.append(None)
        fields.append('Velocities')
        mdi.append(2) # Vz
    elif args.kind == 'MstarVz':
        fields.append('Masses')
        mdi.append(None)
        fields.append('Velocities')
        mdi.append(2)

    if comm.rank == 0:
        print("Importing required data...")
        logger.info("Start import")
    comm.Barrier()
    
    # LOAD SUBSET DIRECTLY HERE TO MAKE IT MORE MEMORY EFFICIENT
    with h5py.File(snapPath(args.TNGPath, args.snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        # Calculate number of particles of required type for snapshot header.
        nPart = header['NumPart_Total'][ptNum] | (header['NumPart_Total_HighWord'][ptNum] << 32)
        if nPart==0:
            raise Exception("No particles of type " + args.kind)
        if comm.rank==0:
            print(f"To load {nPart} particles")
            logger.info(f"{nPart} particles")
        #sys.exit()

        start = comm.rank * nPart // comm.size
        end = (comm.rank+1) * nPart // comm.size

        offsetsThisType = start - snapOffsets[ptNum, :]
        fileNum = np.max(np.where(offsetsThisType >= 0))
        fileOff = offsetsThisType[fileNum]
        numToRead = nPart // comm.size
        print(f"Rank {comm.rank} reading {numToRead} particles starting from index {start}.")

        i = 1
        while gName not in f: # Remember gName is str of ptype
            f = h5py.File(snapPath(args.TNGPath, args.snapNum, i), 'r')
            i += 1

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and args.float32: dtype = np.float32
            if field=='Coordinates': # Initialize directly the fields with correct name so that we don't use additional memory for copying and stuff
                pos = np.zeros(shape, dtype=dtype)
                nMB = round(pos.nbytes / 1024 / 1024, 2)
                if comm.rank==0:
                    logger.info(f"Allocating {nMB}MB for positions, for a total of {comm.allreduce(nMB, op=MPI.SUM)}MB requested.")
            elif field=='Masses':
                mass = np.zeros(shape, dtype=dtype)
            elif field=='ElectronAbundance':
                Xe = np.zeros(shape, dtype=dtype)
            elif field=='InternalEnergy':
                u = np.zeros(shape, dtype=dtype)
            elif field=='NeutralHydrogenAbundance':
                XHI = np.zeros(shape, dtype=dtype)
            elif field=='Velocities':
                vz = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        f = h5py.File(snapPath(args.TNGPath, args.snapNum, fileNum), 'r')

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if field=='Coordinates': # Use fields aready defined
                pos[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
                if comm.rank==0:
                    logger.info("Adding to the positions")
            elif field=='Masses':
                mass[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            elif field=='ElectronAbundance':
                Xe[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            elif field=='InternalEnergy':
                u[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            elif field=='NeutralHydrogenAbundance':
                XHI[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            elif field=='Velocities':
                vz[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    pos /= 1000.
    pos %= args.BoxSize
    # FINISHED LOCAL LOADSUBSET

    comm.Barrier()
    if comm.rank == 0:
        print("Done. Computing quantities...")
        logger.info("Things imported, computing quantities")

    h = 0.6774
    XH = 0.76
    mp = 1.6726219e-27
    Msun10 = 1.989e40
    BoxSize = args.BoxSize
    Mpc_cm = 3.085678e24

    if args.kind == 'Mstar':
        foo = 1
        # Relic
    elif args.kind == 'ne':
        a = scalefactor()
        mass = Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass*XH/mp*Xe # From n_cm3
        del Xe
    elif args.kind == 'nT':
        kb = 1.38064852e-23 # From temperature
        mp = 1.6726219e-27
        ufac = 1e6 #u: (km/s)^2 -> (m/s)^2
        u = 2./3. * ufac * u / kb * 4./(1.+3.*XH+4.*XH*Xe) * mp # Actually this is temperature, but to save mem
        a = scalefactor() 
        mass = Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass*XH/mp*Xe * u # This is n_cm3(mss, Xe) * T = ne*T
        del Xe, u
    elif args.kind == 'nHI':
        a = scalefactor()
        mass = Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass*XH/mp*XHI # From n_cm3
        del XHI
    elif args.kind == 'neVz':
        a = scalefactor() 
        mass = Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass*XH/mp*Xe * vz * a**0.5 # This is n_cm3(mass, Xe)*vz = ne*vz
        del vz
    elif args.kind == 'MstarVz':
        mass *= vz * a**0.5 # Vz factor as above

    comm.Barrier()
    if comm.rank == 0:
        print("Done. Creating layout...")
        logger.info("Creating map")
    lenpos = len(pos)
    layout = pm.decompose(pos)
    pos = layout.exchange(pos)
    if args.kind == 'dm':
        mass = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(lenpos, op=MPI.SUM)
    else:
        mass = layout.exchange(mass)

    comm.Barrier()
    if comm.rank==0:
        print("Painting map")
        logger.info("Painting map")
    TNGmap = pm.paint(pos, mass=mass)
    del pos, mass
    if comm.rank == 0:
        print("Done. Saving map...")
        logger.info("Saving map")

    address = args.TNGPath + '/snapdir_' + str(args.snapNum).zfill(3) + '/' + args.kind + 'map_Nmesh' + str(pm.Nmesh[0])
    FieldMesh(TNGmap).save(address)

paintTNGmap()


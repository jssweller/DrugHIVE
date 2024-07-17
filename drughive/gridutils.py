import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import torch
import warnings


def gauss_3d(x,y,z, r0, var=3):
    return np.exp(-((x-r0[0])**2 + (y-r0[1])**2 + (z-r0[2])**2)/(2*var))

def get_mask(kw, trunc, m):
    '''Creates truncation mask for a 3D kernel.'''
    xx,yy,zz = np.meshgrid(np.arange(kw), np.arange(kw), np.arange(kw))
    mask = np.sqrt((xx-kw/2+0.5)**2 + (yy-kw/2+0.5)**2 + (zz-kw/2+0.5)**2)
    mask = mask < (float(trunc)*m)
    return mask

def get_gaussian_kernel(var, trunc, resolution, m=7, use_old_version=False):
    """Returns a 3D truncated gaussian kernel.

    Args:
        var (float): variance of gaussian.
        trunc (float): truncation radius.
        resolution (float): voxel resolution.
        m (int, optional): resolution multiplier for smoother kernel function. Higher values result in more accurate kernel. Defaults to 7.
        use_old_version (bool, optional): Use old version that had a bug that makes the kernel asymmetric. May still need to be used for old models.. Defaults to False.

    Returns:
        numpy array: 3d kernel with side length (2*trunc) // resolution
    """    

    if use_old_version:
        kw = 2 * trunc / resolution # this produces incorrect results. retained for backward compatibility with old models.
    else:
        kw = (2*trunc) // resolution
    kw += (kw%2 ==0)

    trunc = trunc/resolution
    var = var/resolution**2

    kw = int(kw*m)
    kernel = np.fromfunction(gauss_3d, var=var*m**2, r0=(kw//2,kw//2,kw//2), shape=(kw,kw,kw), dtype=float)

    mask = get_mask(kw, trunc, m)

    kernel = kernel*mask

    kw = int(kw/m)
    newkern = np.zeros((kw,kw,kw))
    for i in range(kw):
        for j in range(kw):
            for k in range(kw):
                newkern[i,j,k] = kernel[i*m:m*(i+1), j*m:m*(j+1), k*m:m*(k+1)].mean()

    return newkern/newkern.max()


def rot3d_random_safe(coords, grid_encoder, pad=0, n_attempts=20, voxels=True, warn_fail=False):
    '''Rotates molecule coordinates within grid, keeping molecule within grid bounds.'''
    shape0 = coords.shape
    if coords.ndim == 2:
        coords = coords.reshape(1, *shape0)

    success_bool = False
    for i in range(n_attempts):
        coords_new, rot_angles = rot3d_random(coords)
        if grid_encoder.is_in_bounds(coords_new, pad=pad, voxels=voxels).all():
            success_bool = True
            continue
    
    if not success_bool:
        if grid_encoder.is_in_bounds(coords, pad=pad, voxels=voxels).all(): 
            # return original coords
            if warn_fail:
                warnings.warn('Rotation in bounds failed. Returning unrotated (in bounds) coordinates.')
            coords_new = coords
            rot_angles = np.zeros_like(rot_angles)
            success_bool = 1
        else:
            # return last rotation attempt
            if warn_fail:
                warnings.warn('Rotation in bounds failed. Returning last attempt (out of bounds).')

    coords_new = coords_new.reshape(shape0)    
    return coords_new, rot_angles, success_bool 



def trans3d_random_safe(coords, grid_encoder, max_dist=None, pad=0, n_attempts=20, voxels=True, warn_fail=False):
    '''Translates molecule coordinates within grid, keeping molecule within grid bounds.'''
    shape0 = coords.shape
    if coords.ndim == 2:
        coords = coords.reshape(1, *shape0)

    if max_dist is None:
        max_dist = grid_encoder.shape[-1]//4
    i = 0
    success_bool = False
    for i in range(n_attempts):
        i += 1
        coords_new, trans_vec = trans3d_random(coords, max_dist=max_dist)
        if grid_encoder.is_in_bounds(coords_new, voxels=voxels, pad=pad).all():
            success_bool = True
            continue

    if not success_bool:
        if grid_encoder.is_in_bounds(coords, pad=pad, voxels=voxels).all():
            # return original coords
            coords_new = coords
            trans_vec = np.zeros_like(trans_vec)
            if warn_fail:
                warnings.warn('Translation in bounds failed. Returning untranslated (in bounds) coordinates.')
        else:
            # return last translation attempt
            if warn_fail:
                warnings.warn('Translation in bounds failed. Returning last attempt (out of bounds).')
    
    coords_new = coords_new.reshape(shape0)
    
    return coords_new, trans_vec, success_bool 


def rot3d_random(coordinates):
    '''Rotates 3D coordinates randomly.'''
    shape0 = coordinates.shape
    if coordinates.ndim == 2:
        coordinates = coordinates.reshape(1, *shape0)

    angles = np.zeros((len(coordinates), 3))
    newcoords = np.zeros(coordinates.shape)

    for i, coords in enumerate(coordinates):
        angles[i] = np.random.random(3)*2*np.pi
        newcoords[i] = rot3d(coords, angles[i])

    return newcoords.reshape(*shape0), angles


def rot3d(coordinates, angles, inverse=False, degrees=False):
    '''Rotates 3D coordinates using euler angles.'''
    R = Rotation.from_euler('zyx', angles, degrees=degrees)
    if inverse:
        R = R.inv()
    if coordinates.ndim == 3:
        newcoords = np.stack([R.apply(x) for x in coordinates], axis=0)
    else:
        newcoords = R.apply(coordinates)
    return newcoords



def trans3d(coordinates, vec, inverse=False):
    '''Translates 3D coordinates.'''
    if inverse:
        vec = -vec
    return coordinates + vec


def trans3d_random(coordinates, max_dist):
    '''Randomly translates 3D coordinates up to a maximum distance.'''
    shape0 = coordinates.shape
    if coordinates.ndim == 2:
        coordinates = coordinates.reshape(1, *shape0)

    tvecs = np.zeros((len(coordinates), 3))
    newcoords = np.zeros(coordinates.shape)

    for i, coords in enumerate(coordinates):
        tvecs[i] = (np.random.random(3)-0.5)*max_dist
        newcoords[i] = trans3d(coords, tvecs[i])
    return newcoords.reshape(*shape0), tvecs

def get_distm(c1,c2):
    '''return distance matrix for coordinates in sets c1 and c2'''
    distm = np.zeros((len(c1),len(c2)))
    for i in range(len(c1)):
        dist = ((c2 - c1[i])**2).sum(axis=1)
        distm[i,:] = dist
    return distm


def sort_coords(coords):
    coords = np.asarray(coords)
    for d in reversed(range(coords.shape[1])):
        coords = coords[np.argsort(coords[:,d])]
    return coords


def align_coords(coords_a, coords_b, dist_thresh=3., plot=False):
    '''Aligns two sets of coordinates and assigns pairs based on distance threshold.'''
    if not isinstance(coords_a, list):
        if isinstance(coords_a, torch.Tensor):
            coords_a = [x.numpy() for x in coords_a]
        if isinstance(coords_a, np.ndarray):
            coords_a = [x for x in coords_a]
    if not isinstance(coords_b, list):
        if isinstance(coords_a, torch.Tensor):
            coords_b = [x.numpy() for x in coords_b]
        if isinstance(coords_b, np.ndarray):
            coords_b = [x for x in coords_b]
        
    alen = len(coords_a)
    blen = len(coords_b)

    aidxs = list(range(len(coords_a)))
    bidxs = list(range(len(coords_b)))

    atom_pairs = []
    
    if plot:
        nrows = int(min(len(coords_a),len(coords_b))//4 + min(len(coords_a),len(coords_b))%4)
        fig,axs = plt.subplots(nrows,4, figsize=(20,nrows*5))
        vmax = cdist(coords_a, coords_b).max()
        
    p = 0
    while min(len(coords_a),len(coords_b)) > 0:
        d = cdist(coords_a, coords_b)
        if d.min() > dist_thresh:
            break
        bcol = np.argmin(d.min(axis=0))
        arow = np.argmin(d[:,bcol])

        aidx = aidxs.pop(arow)
        bidx = bidxs.pop(bcol)

        # if d.min(axis=0) < dist_thresh:
        atom_pairs.append((aidx,bidx))

        coords_a.pop(arow)
        coords_b.pop(bcol)
        
        if plot:
            ax = axs.flatten()[p]
            ax.imshow(d, vmin=1, vmax=vmax)
            ax.scatter(bcol,arow, c='r')
            ax.set_title('step %d   val=%.2f'%(p,d[arow,bcol]))
            p += 1
    
    
    if len(atom_pairs) > 0:
        atom_pairs = np.asarray(atom_pairs, dtype=int)
        atom_pairs = atom_pairs[np.argsort(atom_pairs[:,1])] # sort to keep second input (target) in order

        unpaired_0 = np.asarray([x for x in range(alen) if x not in atom_pairs[:,0]], dtype=int)
        unpaired_1 = np.asarray([x for x in range(blen) if x not in atom_pairs[:,1]], dtype=int)
    else:
        unpaired_0 = np.arange(alen)
        unpaired_1 = np.arange(blen)

    return atom_pairs, unpaired_0, unpaired_1

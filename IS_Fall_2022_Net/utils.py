"""for opening files and stuff from cochlea"""
import torch
import numpy as np
import functools
import struct

def read_mesh_verts(mesh_file): #in millis
    with open(mesh_file, 'rb') as mesh_in:
        return __read_mesh_verts_bytes(mesh_in)

def __read_mesh_verts_bytes(mesh_in): #in millis
    _fread = create_fread(mesh_in)
    _, num_verts, num_triangles, n = _fread('i', 4)

    orientation = None
    dimension = None
    voxel_size = None
    color = np.zeros(3)

    if n == -1:
        orientation = np.array(_fread('i', 3), dtype=np.int32)
        dimension = np.array(_fread('i', 3), dtype=np.int32)
        voxel_size = np.array(_fread('f', 3))
        color = np.array(_fread('i', 3))
    else:
        color[0] = n
        color[1:3] = np.array(_fread('i', 2))

    verts = np.reshape(np.array(_fread('f', 3 * num_verts)), (-1, 3))
    triangles = np.reshape(np.array(_fread('i', 3 * num_triangles),
                                    dtype=np.int32), (-1, 3))
    return verts

def create_fread(f):
    return functools.partial(fread, f)

def fread(f, fmt, count=1, endian='<'):
    fmt = endian + str(count) + fmt
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))

def read_raw_volume_file(filename, dim, data_type, order="F"):
    #cuz reubens kept giving me errors
    try:
        dim = dim.numpy().astype(np.int32)
    except:
        pass
    if data_type == "h":
        dtype = np.int16
    elif data_type == "B":
        dtype = np.uint8
    elif data_type == 'f':
        dtype = np.single
    else:
        dtype = data_type
    with open(filename, "rb") as file:
        I = np.fromfile(file, dtype = dtype)
    I = np.reshape(I, dim, order=order)
    return I

def write_raw_volume_file(filename, I, data_type, order = "F"):
    if data_type == "h":
        dtype = np.int16
    elif data_type == "B":
        dtype = np.uint8
    elif data_type == 'f':
        dtype = np.single
    else:
        dtype = data_type
    with open(filename,"wb") as file:
        towrite = I.astype(dtype)
        tobyte = towrite.tobytes(order="F")
        file.write(tobyte)
    return

#image = read_raw_volume_file(im_filename, dim, data_type='h') #int16
#mask = read_raw_volume_file(msk_filename, dim, data_type='B') #uint8
#dist = read_raw_volume_file(dist_filename, dim, data_type='f') #float
def reuben_read_raw_volume_file(filename, dimension, voxel_size=None, orientation=None, data_type='h'):
    try:
        dimension = dimension.numpy()
    except:
        dimension = np.array(dimension)
    image = None
    with open(filename, 'rb') as im_resource:
        fread = create_fread(im_resource)
        image = np.reshape(
            np.array(fread(data_type, np.prod(dimension))),
            dimension,
            order='F'
        )
    return image

def sample(volume, points, pad_value=None):
    floored = torch.floor(points)
    remainder = points - floored
    floored = floored.long()

    x = floored[:, 0]
    y = floored[:, 1]
    z = floored[:, 2]

    rmx = remainder[:, 0]
    rmy = remainder[:, 1]
    rmz = remainder[:, 2]

    #pixel_values = torch.zeros(len(points)).cuda()
    pixel_values = torch.zeros(len(points)).to(volume.device)
    for x_inc in range(2):
        for y_inc in range(2):
            for z_inc in range(2):
                x_indexes = (x + x_inc)
                y_indexes = (y + y_inc)
                z_indexes = (z + z_inc)

                valids_x = torch.clamp(x_indexes, 0, volume.shape[0] - 1)
                valids_y = torch.clamp(y_indexes, 0, volume.shape[1] - 1)
                valids_z = torch.clamp(z_indexes, 0, volume.shape[2] - 1)

                values = volume[valids_x, valids_y, valids_z]
                if pad_value is not None:
                    values[valids_x != x_indexes] = pad_value
                    values[valids_y != y_indexes] = pad_value
                    values[valids_z != z_indexes] = pad_value
                multiplier = (
                    ((1. - 2. * x_inc) * ((1. - x_inc) - rmx)) *
                    ((1. - 2. * y_inc) * ((1. - y_inc) - rmy)) *
                    ((1. - 2. * z_inc) * ((1. - z_inc) - rmz))
                )

                pixel_values = pixel_values + (multiplier * values)

    return pixel_values

def my_erf(A):
    return torch.erf(A)/2 + .5

def get_slice_i(annotations):
    A = annotations.cpu().detach().numpy()
    maxi = 0
    maxsum = 0
    for i in range(A.shape[4]):
        smallA = A[0,0,:,:,i]
        thissum = smallA.sum()
        if thissum > maxsum:
            maxi = i
            maxsum = thissum
    return maxi

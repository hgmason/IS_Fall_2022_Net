import numpy as np
import inspect
import torch
from dataclasses import dataclass
from typing import Union
import functools
import struct
import random

def fread(f, fmt, count=1, endian='<'):
    fmt = endian + str(count) + fmt
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))

def create_fread(f):
    return functools.partial(fread, f)

def check_numpy(x):
    return inspect.getmodule(type(x)).__name__ == 'numpy'

@dataclass
class ImageAttributes:
    dimension: Union[np.array, torch.Tensor]
    voxel_size: Union[np.array, torch.Tensor]
    orientation: Union[np.array, torch.Tensor] = None

    def __post_init__(self):
        if check_numpy(self.dimension):
            self.dimension = torch.Tensor(self.dimension)
        if check_numpy(self.voxel_size):
            self.voxel_size = torch.Tensor(self.voxel_size)

        self.dimension = self.dimension.long()
        self.voxel_size = self.voxel_size.double()

        if self.orientation is not None:
            if check_numpy(self.orientation):
                self.orientation = torch.Tensor(self.orientation)
            self.orientation = self.orientation.long()


@dataclass
class Mesh:
    verts: Union[np.array, torch.Tensor]
    triangles: Union[np.array, torch.Tensor]
    color: Union[np.array, torch.Tensor] = None

    def __post_init__(self):
        if check_numpy(self.verts):
            self.verts = torch.Tensor(self.verts)
        if check_numpy(self.triangles):
            self.triangles = torch.Tensor(self.triangles)
        self.verts = self.verts.double()
        self.triangles = self.triangles.long()

def read_mesh_file(mesh_file):
    with open(mesh_file, 'rb') as mesh_in:
        return __read_mesh_bytes(mesh_in)
    
def __read_mesh_bytes(mesh_in):
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
    mesh = Mesh(verts, triangles)
    image_attributes = ImageAttributes(dimension, voxel_size, orientation)
    return mesh, image_attributes

def write_mesh_file(filename, mesh, image_attrs, mask=None):
    with open(filename, 'wb') as outfile:
        __write_mesh_bytes(outfile, mesh, image_attrs, mask)
        
        
def __write_mesh_bytes(outfile, mesh, image_attrs, mask=None):
    outbytes = struct.pack(
        '4i',
        random.randint(0, 10000),
        len(mesh.verts),
        len(mesh.triangles),
        -1
    )
    outfile.write(outbytes)
    outfile.write(image_attrs.orientation.numpy().astype(np.int32).tobytes())
    outfile.write(image_attrs.dimension.numpy().astype(np.int32).tobytes())
    outfile.write(image_attrs.voxel_size.numpy().astype(np.float32).tobytes())

    color = np.random.randint(0, 255, size=(3,), dtype=np.int32)
    outfile.write(color.astype(np.int32).tobytes())
    outfile.write(mesh.verts.numpy().astype(np.float32).flatten().tobytes())
    outfile.write(mesh.triangles.numpy().astype(np.int32).flatten().tobytes())
    if mask is not None:
        outfile.write(struct.pack('1f', 1.0))
        outfile.write(struct.pack('2i', 2, len(mask)))
        outfile.write(struct.pack('6d', 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        mask_to_int = []
        for i in mask:
            if i is True:
                mask_to_int.append(1)
            else:
                mask_to_int.append(0)

        mask_to_int = np.array(mask_to_int)
        outfile.write(mask_to_int.astype(np.int32).flatten().tobytes())
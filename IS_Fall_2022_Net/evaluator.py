from copy import deepcopy
import utils
import nrrd
import os
import parser
import transforms
from parser import Mesh, ImageAttributes

from scipy.interpolate import RegularGridInterpolator as rgi
import numpy as np
import torch
from scipy import ndimage

def get_mask(target_mesh, atlas_mesh, atlas_mask, dim):
    #set up general and convert to millimeter torch tensors
    device = "cuda";
    try:
        target_mesh_torch = torch.Tensor(target_mesh).to(device)
    except:
        target_mesh_torch = target_mesh.to(device)
    try:
        atlas_mesh_torch = torch.Tensor(atlas_mesh).to(device)
    except:
        atlas_mesh_torch = atlas_mesh.to(device)
    #get the tps transform function
    tps_transform = transforms.thin_plate_spline(atlas_mesh_torch, target_mesh_torch)
    #set up target image points
    xx = np.arange(0, dim[0])
    yy = np.arange(0, dim[1])
    zz = np.arange(0, dim[2])
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij') #it's definitely this one, cool
    target_points = np.zeros((len(X.flatten()), 3))
    target_points[:,0] = X.flatten()
    target_points[:,1] = Y.flatten()
    target_points[:,2] = Z.flatten()
    target_points = torch.Tensor(target_points)
    #run through tps, break up into sections and do it on each part
    sections = 50
    width = len(target_points)//sections
    atlas_points = np.zeros(target_points.shape)
    #--send to gpu
    target_points = target_points.to(device)
    atlas_points = torch.Tensor(atlas_points).to(device)
    with torch.no_grad():
        for s in range(sections-1):
            atlas_points[width*s:width*(s+1)] = tps_transform(target_points[width*s:width*(s+1)])
        atlas_points[width*(sections-1):len(atlas_points)-1] = tps_transform(target_points[width*(sections-1):len(target_points)-1])
    #--get back from gpu
    target_points = target_points.cpu()
    atlas_points = atlas_points.cpu().numpy()
    
    #print("inside get mask: atlas mask dtype:", atlas_mask.dtype)
    atlas_interp_func = rgi((xx, yy, zz), atlas_mask, bounds_error = False, fill_value = 0)
    pixel_vals = atlas_interp_func(atlas_points)
    def_mask = np.reshape(pixel_vals, dim)
    #print("inside get mask | atlas min: %0.2f, atlas max: %0.2f, target min %0.2f, target max %0.2f | atlas dtype: %s, target dtype: %s" % (atlas_mask.min(), atlas_mask.max(), def_mask.min(), def_mask.max(), atlas_mask.dtype, def_mask.dtype))
    return def_mask

def resize(mask, old_dim, new_dim):
    xx = np.arange(0,old_dim[0]); yy = np.arange(0,old_dim[1]); zz = np.arange(0,old_dim[2])
    mask_func = rgi((xx,yy,zz), mask, bounds_error=False, fill_value=0)
    new_xx = np.linspace(0, old_dim[0], new_dim[0]); new_yy = np.linspace(0, old_dim[1], new_dim[1]); new_zz = np.linspace(0, old_dim[2], new_dim[2]);
    X, Y, Z = np.meshgrid(new_xx, new_yy, new_zz, indexing='ij')
    X = X.flatten(); Y = Y.flatten(); Z = Z.flatten()
    new_points = np.zeros((len(X), 3))
    new_points[:,0] = X; new_points[:,1] = Y; new_points[:,2] = Z
    new_mask = mask_func(new_points).reshape(new_dim)
    return new_mask

def dice(im1, im2, oob_mask = None): #doesnt make a difference but yay consistency
    im1 = im1.flatten()
    im2 = im2.flatten()
    if oob_mask is not None:
        oob = np.invert(oob_mask.flatten())
        im1 = im1[oob]
        im2 = im2[oob]
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def comp_mesh_to_im(verts, mask, sz, dim, mask_sz, mask_dim, oob_mask = None):
    #print("inside comp mesh to im", type(oob_mask))
    interp_method = "linear"
    #find inverse mask
    n_mask = np.invert(mask)
    #get distance to closest pixel
    inner = ndimage.distance_transform_edt(mask)
    outer = ndimage.distance_transform_edt(n_mask)
    D = (outer + inner)*mask_sz #absolute value distance in millimeters
    #create a interpolater for finding the distance at points
    ii = np.arange(0,mask_dim[0])*mask_sz
    dist_ifunc = rgi((ii,ii,ii), D, bounds_error = False, fill_value = None, method = interp_method)
    
    #take out the verts that are in oob_mask
    if oob_mask is not None:
        oob_ifunc = rgi((ii,ii,ii), oob_mask, bounds_error = False, fill_value = None, method = interp_method)
        oob_verts_val = oob_ifunc(verts*sz) <= .5
        #print(D.shape, oob_mask.shape, oob_mask.min(), oob_mask.max())
        #print(oob_verts_val.min(), oob_verts_val.max())
        verts = verts[oob_verts_val]
    
    #run the verts from the mesh through the interp
    dists = dist_ifunc(verts*sz)
    
    dist_max = max(dists)
    dist_95 = np.percentile(dists, 95)
    dist_85 = np.percentile(dists, 85)
    dist_mean = np.mean(dists)
    return dist_max, dist_95, dist_85, dist_mean

def evaluate_fnames(test_filenames, test_dim, test_sz, gt_filenames, gt_dim, gt_sz, atlas_mesh_fname, atlas_mask_fname, atlas_mesh_sz, oob_mask_names = [], return_masks = False, return_meshes = False ): #the names of the deformation fields, the names of the ground truth mask, and the name of the atlas mesh
    #print("USING ATLAS MESH", atlas_mesh_fname)
    N = len(test_filenames)
    atlas_mesh, atlas_attr = parser.read_mesh_file(atlas_mesh_fname) #load as voxels
    atlas_mesh.verts = atlas_mesh.verts*atlas_mesh_sz/test_sz
    atlas_mask, _ = nrrd.read(atlas_mask_fname)
    results_95 = []
    results_85 = []
    results_dice = []
    masks = []
    meshes = []
    full_meshes = []
    #print(oob_mask_names)
    for i in range(N):
        #get filenames
        test_fn = test_filenames[i]
        #print("TEST FN[%d]: %s"%(i, test_fn))
        gt_fn = gt_filenames[i]
        #print("TEST FN[%d]: %s | GT FN[%d]: %s"%(i, test_fn, i, gt_fn))
        if len(oob_mask_names) > 0:
            oob_mask_fname = oob_mask_names[i]
        #get deformation field
        net_def = utils.read_raw_volume_file(test_fn, (3,test_dim[0],test_dim[1],test_dim[2]), np.single)
        X_def = net_def[0,:,:,:].squeeze()
        Y_def = net_def[1,:,:,:].squeeze()
        Z_def = net_def[2,:,:,:].squeeze()
        #get ground truth image
        gt = utils.read_raw_volume_file(gt_fn, gt_dim, np.uint8)
        if len(oob_mask_names) > 0:
            oob_mask = utils.read_raw_volume_file(oob_mask_fname, gt_dim, np.uint8) >= 127
        else:
            oob_mask = None
        #print(oob_mask)
        #print(mesh_fname)
        mask, mesh, dist_95, dist_85, dice_score = evaluate_kw(X_def, Y_def, Z_def, test_sz, gt >= 127, gt_sz, atlas_mesh.verts, atlas_mask, oob_mask, return_mask = True)
        results_95.append(dist_95)
        results_85.append(dist_85)
        results_dice.append(dice_score)
        masks.append(mask)
        meshes.append(mesh)
        mesh_obj = deepcopy(atlas_mesh); mesh_obj.verts = mesh
        full_meshes.append((mesh_obj, atlas_attr))
    if return_meshes:
        return results_95, results_85, results_dice, full_meshes
    elif return_masks:
        return masks, meshes, results_95, results_85, results_dice
    else:
        return results_95, results_85, results_dice
        
        
def evaluate_kw(X_def, Y_def, Z_def, kw_sz, gt_mask, gt_sz, atlas_verts, atlas_mask, oob_mask = None, return_mask = False): #atlas mask expected to be 0 to 255
    '''create target mesh'''
    #set up rgi in network voxels
    xx = np.arange(0, X_def.shape[0]); yy = np.arange(0, X_def.shape[1]); zz = np.arange(0, X_def.shape[2])
    x_interp_func = rgi((xx, yy, zz), X_def, bounds_error = False, fill_value = 0)
    y_interp_func = rgi((xx, yy, zz), Y_def, bounds_error = False, fill_value = 0)
    z_interp_func = rgi((xx, yy, zz), Z_def, bounds_error = False, fill_value = 0)
        
    #interpolate to find new verts
    verts = deepcopy(atlas_verts)
    try:
        verts = torch.Tensor(verts)
    except:
        pass
    new_verts = torch.Tensor(np.zeros_like(verts))
    new_verts[:,0] = verts[:,0] + torch.Tensor(x_interp_func(verts))
    new_verts[:,1] = verts[:,1] + torch.Tensor(y_interp_func(verts))
    new_verts[:,2] = verts[:,2] + torch.Tensor(z_interp_func(verts))
        
    '''evaluate distance metric (95% only)'''
    #print("inside evaluate kw", type(oob_mask))
    _, dist_95, dist_85, _ = comp_mesh_to_im(new_verts, gt_mask, kw_sz, atlas_mask.shape, gt_sz, gt_mask.shape, oob_mask)
    
    '''create target mask'''
    mask = get_mask(new_verts, atlas_verts, atlas_mask, atlas_mask.shape)
    '''resize'''
    if atlas_mask.shape != gt_mask.shape:
        big_mask = resize(mask, atlas_mask.shape, gt_mask.shape) >= 127
    else:
        big_mask = mask >= 127
    
    '''evaluate mask metric'''
    dice_score = dice(big_mask, gt_mask, oob_mask)   
    
    if return_mask:
        return big_mask, new_verts, dist_95, dist_85, dice_score
    else:
        return dist_95, dist_85, dice_score
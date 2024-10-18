import torch
import utils

def loss_func(lrnr, small_combo_A, trans_pred, def_field, target_meshes): #all loss_funcs must have these inputs
        #atlas_mesh (N, 3), def_field (5, 3, 64, 64, 64), target_meshes (5, N, 3)
        atlas_mesh = lrnr.p2p_atlas_mesh
        b = target_meshes.shape[0]
        T = target_meshes[0,:,:]
        dx = utils.sample(def_field[0,0,:,:,:], atlas_mesh)
        dy = utils.sample(def_field[0,1,:,:,:], atlas_mesh)
        dz = utils.sample(def_field[0,2,:,:,:], atlas_mesh)
        newx = atlas_mesh[:,0] + dx
        newy = atlas_mesh[:,1] + dy
        newz = atlas_mesh[:,2] + dz
        new_verts = torch.cat((newx.unsqueeze(1),newy.unsqueeze(1),newz.unsqueeze(1)),dim=1)
        diff1 = torch.linalg.norm(new_verts - T, dim = 1)
        ret = diff1

        if b >= 2:
            T = target_meshes[1,:,:]
            dx = utils.sample(def_field[1,0,:,:,:], atlas_mesh)
            dy = utils.sample(def_field[1,1,:,:,:], atlas_mesh)
            dz = utils.sample(def_field[1,2,:,:,:], atlas_mesh)
            newx = atlas_mesh[:,0] + dx
            newy = atlas_mesh[:,1] + dy
            newz = atlas_mesh[:,2] + dz
            new_verts = torch.cat((newx.unsqueeze(1),newy.unsqueeze(1),newz.unsqueeze(1)),dim=1)
            diff2 = torch.linalg.norm(new_verts - T, dim = 1)
            ret = ret + diff2

        if b>=3:
            T = target_meshes[2,:,:]
            dx = utils.sample(def_field[2,0,:,:,:], atlas_mesh)
            dy = utils.sample(def_field[2,1,:,:,:], atlas_mesh)
            dz = utils.sample(def_field[2,2,:,:,:], atlas_mesh)
            newx = atlas_mesh[:,0] + dx
            newy = atlas_mesh[:,1] + dy
            newz = atlas_mesh[:,2] + dz
            new_verts = torch.cat((newx.unsqueeze(1),newy.unsqueeze(1),newz.unsqueeze(1)),dim=1)
            diff3 = torch.linalg.norm(new_verts - T, dim = 1)
            ret = ret + diff3

        if b >=4:
            T = target_meshes[3,:,:]
            dx = utils.sample(def_field[3,0,:,:,:], atlas_mesh)
            dy = utils.sample(def_field[3,1,:,:,:], atlas_mesh)
            dz = utils.sample(def_field[3,2,:,:,:], atlas_mesh)
            newx = atlas_mesh[:,0] + dx
            newy = atlas_mesh[:,1] + dy
            newz = atlas_mesh[:,2] + dz
            new_verts = torch.cat((newx.unsqueeze(1),newy.unsqueeze(1),newz.unsqueeze(1)),dim=1)
            diff4 = torch.linalg.norm(new_verts - T, dim = 1)
            ret = ret + diff4

        if b >= 5:
            T = target_meshes[4,:,:]
            dx = utils.sample(def_field[4,0,:,:,:], atlas_mesh)
            dy = utils.sample(def_field[4,1,:,:,:], atlas_mesh)
            dz = utils.sample(def_field[4,2,:,:,:], atlas_mesh)
            newx = atlas_mesh[:,0] + dx
            newy = atlas_mesh[:,1] + dy
            newz = atlas_mesh[:,2] + dz
            new_verts = torch.cat((newx.unsqueeze(1),newy.unsqueeze(1),newz.unsqueeze(1)),dim=1)
            diff5 = torch.linalg.norm(new_verts - T, dim = 1)
            ret = ret + diff5
        ret = ret / b
        thresh = 10
        ret = ret.mean()
        ret = 1 / (1 + torch.exp(-(ret -thresh)))
        return ret
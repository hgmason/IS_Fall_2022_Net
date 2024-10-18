import torch

def loss_func(lrnr, small_combo_A, trans_pred, def_field, target_meshes): #all loss_funcs must have these inputs
    
    batch_size = def_field.shape[0]
    #distances = lrnr.atlas_dist.unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1,1,1)
    #D = distances**2 < 1.5**2 #highlight where distances is less than 1.5 mm, cuz distances is in mm
    D = lrnr.atlas_dist_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1,1,1)
        
    Dy = D[:,:,1:,:,:]
    Dx = D[:,:,:,1:,:]
    Dz = D[:,:,:,:,1:]
    
    dy = torch.abs(def_field[:, :, 1:, :, :] - def_field[:, :, :-1, :, :]) #each pixel minus the pixel next to it
    dx = torch.abs(def_field[:, :, :, 1:, :] - def_field[:, :, :, :-1, :])
    dz = torch.abs(def_field[:, :, :, :, 1:] - def_field[:, :, :, :, :-1])
    
    #mask the gradient calculations
    dy = dy * Dy
    dx = dx * Dx
    dz = dz * Dz
    
    dy = dy * dy
    dx = dx * dx
    dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    grad = d / 3.0

    return grad
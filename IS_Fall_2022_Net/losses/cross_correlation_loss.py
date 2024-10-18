import torch

def loss_func(lrnr, small_combo_A, trans_pred, def_field, target_meshes): #all loss_funcs must have these inputs:
    #print(small_combo_A.shape, trans_pred.shape)
    #dfsdfsdfsd
    small_val = 1e-37
    target = trans_pred.flatten()
    atlas = small_combo_A[:,1,:,:,:].unsqueeze(1).flatten()
    mu_f = torch.mean(target)
    mu_t = torch.mean(atlas) 
    F = target - mu_f
    T = atlas - mu_t
    F = F / (torch.linalg.norm(F) + small_val)
    T = T / (torch.linalg.norm(T) + small_val)
    ret = torch.inner(F, T) #-1 when worst, 1 when best
    ret = -ret #-1 when best, 1 when worst
    ret = ret/2 + .5 #0 when best, 1 when worst
    #ret = ret * 500 * 64**3 #to kinda make it more similar to mse i guess     
    return ret
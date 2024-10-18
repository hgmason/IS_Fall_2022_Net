import torch

def loss_func(lrnr, small_combo_A, trans_pred, def_field, target_meshes): #all loss_funcs must have these inputs    
    #so we're comparing trans_pred to the atlas mask
    P = small_combo_A[:,2,:,:,:].unsqueeze(1) #modified atlas mask, regular atlas mask if modified filename not provided
    I = trans_pred
    batch_size = I.shape[0]
    distances = lrnr.atlas_dist.unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1,1,1)
    small_val = 1e-30
    threshold = 1.5
    
    #find means
    m0 = (I*P).sum() /  (P.sum() + small_val) #weighted mean inside prediction
    m1 = (I*(1-P)).sum() / ((1-P).sum() + small_val) #weighted mean outside prediction

    #find variance
    v0 = (P*(I - m0)**2).sum()/(P.sum() + small_val)  #weighted varience inside prediction
    v1 = ((1-P)*(I - m1)**2).sum() / ((1-P).sum() + small_val) #weighted varience outside prediction

    #normalize the variances
    max_poss = 1 #for when when range is 0 to 1 ish
    norm_scalar = 1 #make it less tiny
    v0 = v0 / max_poss * norm_scalar
    v1 = v1 / max_poss * norm_scalar

    return v0 + v1
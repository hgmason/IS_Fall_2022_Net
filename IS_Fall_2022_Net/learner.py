from loss_utils import LossManager
import json
import os
import utils
import torch
import nrrd
from scipy.io import loadmat
import model
import data_utils
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.ndimage
import evaluator
import pickle
import parser

"""the main object for training and testing a model"""

class Learner():
    def __init__(self, params, trialname = "DABS-MS-Example", verbose_learner = True):
        self.verbose_learner = verbose_learner #print during epochs, testing, etc
        self.populate_params(params)

        trialname = trialname+"_"+str(self.epochs)
        self.trialname = trialname
        self.trial_dir = os.path.join(self.progress_dir, "final_networks_and_losses")

        self.make_destination_dirs()
        self.load_globals()
        self.load_model()
        self.make_dataloaders()

        #initialize the loss manager
        self.loss_mngr = LossManager(self)
        return

    def populate_params(self, params):
        if self.verbose_learner:
            print("Loading parameters...")
        #get all the parameters defined
        training_params = params['training_params']
        path_params = params['path_params']
        debug_params = params['debug_params']

        '''training params'''
        self.batch_size = training_params['batch_size']
        self.epochs = training_params['epochs']
        self.device = training_params["device"]
        self.dim = torch.Tensor(training_params['dim'])
        self.sz = torch.Tensor(training_params['sz'])
        self.gt_sz = self.sz
        self.gt_dim = self.dim

        '''path params'''
        self.progress_dir = path_params['progress_dir']
        self.save_dir_def = os.path.join(self.progress_dir, "deformation_fields")
        self.save_dir_slices = os.path.join(self.progress_dir, "test_slices")
        self.valid_dir_slices = os.path.join(self.progress_dir, "validation_slices")
        self.save_mesh_dir = os.path.join(self.progress_dir, "deformed_meshes")
        self.img_dir  = path_params['img_dir']
        self.mesh_dir = path_params['mesh_dir']
        self.gt_dir   = path_params['gt_dir']
        self.img_template  = os.path.join(self.img_dir, path_params['img_template'])
        self.mesh_template = os.path.join(self.mesh_dir, path_params['mesh_template'])
        self.gt_template   = os.path.join(self.gt_dir, path_params['gt_template'])
        self.atlas_im_filename   = path_params['atlas_im_filename']
        self.atlas_mesh_filename = path_params['atlas_mesh_filename']
        self.p2p_atlas_mesh_filename = path_params['p2p_atlas_mesh_filename']
        self.atlas_mask_filename = path_params['atlas_mask_filename']
        self.atlas_dist_filename = path_params['atlas_dist_filename']

        '''get keywords (patient ids)'''
        with open(path_params["keyword_filename"]) as file:
            contents = file.read()
            train, valid, test = contents.split("\n\n")
        self.train_keywords = train.split('\n')
        self.valid_keywords = valid.split('\n')
        self.test_keywords  = test.split('\n')

        '''loss weights'''
        self.losses = params["losses"]

        '''debug params'''
        self.verbose = debug_params['verbose']
        return

    def make_destination_dirs(self):
        if self.verbose_learner:
            print("Making destination directories...")
        dirs = [self.progress_dir, self.save_dir_def, self.save_dir_slices, self.valid_dir_slices, self.trial_dir, self.save_mesh_dir]
        for d in dirs:
            os.makedirs(d, exist_ok = True)
        return

    def load_globals(self):
        if self.verbose_learner:
            print("Loading global values...")
        #load the atlas image
        self.atlas_image, _ = nrrd.read(self.atlas_im_filename)
        #load the atlas mask
        self.atlas_mask, _ = nrrd.read(self.atlas_mask_filename)
        #load the atlas distance
        self.atlas_dist, _ = nrrd.read(self.atlas_dist_filename)
                
        #create the distance mask used for the gradient term
        DM = np.abs(self.atlas_dist)
        high_thresh = 4
        low_thresh = 1
        DM[DM > high_thresh] = high_thresh
        DM[DM < low_thresh] = low_thresh
        DM = (high_thresh - DM) / (high_thresh - low_thresh) #flip and scale between 0 and 1
        percent = .5
        DM = ((1-percent) + percent*DM) #decide how much more important the area around the atlas is to the rest
        self.atlas_dist_mask = DM
        
        #load the atlas mesh
        self.atlas_mesh = utils.read_mesh_verts(self.atlas_mesh_filename)
        self.p2p_atlas_mesh = utils.read_mesh_verts(self.p2p_atlas_mesh_filename)
        #pass the mask and image through the data transforms
        temp_data3D = data_utils.Data3D([], self)
        self.atlas_mask  = temp_data3D.transform(self.atlas_mask, scaling = "A").squeeze(0)
        self.atlas_image = temp_data3D.transform(self.atlas_image, scaling = "I").squeeze(0)
        del temp_data3D
        #put everything on the appropriate device
        self.atlas_mask  = self.atlas_mask.to(self.device)
        self.atlas_image = self.atlas_image.to(self.device)
        self.atlas_mesh = torch.Tensor(self.atlas_mesh).to(self.device)
        self.p2p_atlas_mesh = torch.Tensor(self.p2p_atlas_mesh).to(self.device)
        self.atlas_dist  = torch.Tensor(self.atlas_dist).to(self.device)
        self.atlas_dist_mask = torch.Tensor(self.atlas_dist_mask).to(self.device)            
        return

    def load_model(self, use_res = True):
        if self.verbose_learner:
            print("Loading model...")
        unet = model.DefNet(self.dim, 1, 1, verbose = self.verbose)
        unet = unet.to(self.device)
        optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
        self.unet = unet
        self.optimizer = optimizer
        return

    def save_model(self,valid = False):
        if self.verbose_learner:
            print("Saving model...")
        if valid:
            filename =  os.path.join(self.trial_dir,'best_unet_'+self.trialname+'.pth')
        else:
             filename = os.path.join(self.trial_dir,'unet_'+self.trialname+'.pth')
        torch.save(self.unet.state_dict(),filename)
        while True:
            try:
                torch.save(self.unet.state_dict(), filename)
                return
            except:
                pass
        return

    def make_dataloaders(self):
        if self.verbose_learner:
            print("Making dataloaders...")
        
        trainset = data_utils.Data3D(self.train_keywords, self)
        validset = data_utils.Data3D(self.valid_keywords, self)
        testset  = data_utils.Data3D(self.test_keywords , self)

        self.valid_loader = torch.utils.data.DataLoader(validset, batch_size = self.batch_size)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size = 1)
        self.train_loader = torch.utils.data.DataLoader (trainset, batch_size = self.batch_size)
        
        self.train_size = len(self.train_keywords)
        self.valid_size = len(self.valid_keywords)
        self.test_size  = len(self.test_keywords)
        
        return

    def train(self):
        if self.verbose_learner:
            print("Starting training...")
        start = time.time()
        best_valid = float('inf')
        best_valid_epoch = 0
        for epoch in range(1,self.epochs+1):
            estart = time.time()
            if self.verbose_learner:
                print('Epoch {}/{}:'.format(epoch, self.epochs))
            '''train'''
            for batch_idx, batch in enumerate(self.train_loader):
                #print("\ttrain batch", batch_idx, "out of", len(self.train_loader))
                self.loop(batch, grad=True)
            '''validate'''
            for batch_idx, batch in enumerate(self.valid_loader):
                #print("\tvalid batch", batch_idx, "out of", len(self.valid_loader))
                if batch_idx == 1 and (epoch%10 == 1):
                    self.loop(batch, grad=False, save = epoch)
                else:
                    self.loop(batch, grad=False)
            _, valid_loss = self.loss_mngr.end_epoch(epoch)            
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_valid_epoch = epoch
                self.save_model(valid = True)
            eend = time.time()
            if self.verbose_learner:
                print("\tcompleted in", eend - estart, "seconds")
        time_elapsed = time.time() - start
        if self.verbose_learner:
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.save_model()
        self.loss_mngr.save_losses()
        self.loss_mngr.make_plots()
        return

    def loop(self, batch, grad = True, save = -1, return_outputs = False, test = False):
        if grad:
            images, meshes, filenames = batch
            images = images.to(self.device)
            meshes = meshes.to(self.device)
            self.optimizer.zero_grad()
            trans_pred, def_field = self.unet.forward(images)
            trans_pred = trans_pred[:,0,:,:,:].unsqueeze(1)
            loss = self.loss_mngr.total_loss(images, meshes, trans_pred, def_field, train = grad, record = (not test))
            loss.backward()            
            self.optimizer.step()
        else:
            with torch.no_grad():
                images, meshes, filenames = batch
                images = images.to(self.device)
                meshes = meshes.to(self.device)
                trans_pred, def_field = self.unet.forward(images)
                trans_pred = trans_pred[:,0,:,:,:].unsqueeze(1)
                loss = self.loss_mngr.total_loss(images, meshes, trans_pred, def_field, train = grad, record = (not test))

        if save >= 0 and not test:
            self.save_slices(images, trans_pred, def_field, filenames, valid = (not grad), epoch = save)
        if test:
            self.save_slices(images, trans_pred, def_field, filenames, valid = False, epoch = save)
        if return_outputs:
            return trans_pred, def_field, filenames, loss
        else:
            return

    def export(self, return_loss = False, include_valid = True):
        if return_loss:
            temp = []
        for batch_idx, batch in enumerate(self.test_loader):
            if self.verbose_learner:
                print("Testing batch", batch_idx+1, "of", self.test_size, end = " | ")
            trans_pred, def_field, filenames, loss = self.loop(batch, grad=False, save = self.epochs, return_outputs = True, test = True) #saves the slices
            if return_loss:
                temp.append(loss.item())
            if self.verbose_learner:
                print("Loss:", loss.item())
            
            TP_im = trans_pred[0,0,:,:,:]
            TP_im = TP_im.cpu().numpy()

            #to not break improvise
            TP_im[0,0,0] = 0
            TP_im[0,0,1] = 255

            #phrase = os.path.basename(filenames[0]).split("_")[0]
            phrase = os.path.basename(filenames[0])[0:os.path.basename(filenames[0]).find("_scaled")]
                
            with open(os.path.join(self.save_dir_def, phrase+"_def_im.im"),"wb") as file:
                towrite = TP_im.astype(np.uint16)
                tobyte = towrite.tobytes(order="F")
                file.write(tobyte)

            #save the def field
            D = def_field[0,:,:,:,:]
            with open(os.path.join(self.save_dir_def, phrase+"_def_field.im"),"wb") as file:
                towrite = D.cpu().numpy().astype(np.float32)
                tobyte = towrite.tobytes(order="F")
                file.write(tobyte)
                
        if include_valid:
            validset = data_utils.Data3D(self.valid_keywords, self)
            temp_valid_loader = torch.utils.data.DataLoader(validset, batch_size = 1)
            for batch_idx, batch in enumerate(temp_valid_loader):
                if self.verbose_learner:
                    print("Testing valid batch", batch_idx+1, "of", self.valid_size, end = " | ")
                trans_pred, def_field, filenames, loss = self.loop(batch, grad=False, save = self.epochs, return_outputs = True, test = True) #saves the slices
                if self.verbose_learner:
                    print("Loss:", loss.item())

                TP_im = trans_pred[0,0,:,:,:]
                TP_im = TP_im.cpu().numpy()

                #to not break improvise
                TP_im[0,0,0] = 0
                TP_im[0,0,1] = 255

                #phrase = os.path.basename(filenames[0]).split("_")[0]
                phrase = os.path.basename(filenames[0])[0:os.path.basename(filenames[0]).find("_scaled")]

                with open(os.path.join(self.save_dir_def, phrase+"_def_im.im"),"wb") as file:
                    towrite = TP_im.astype(np.uint16)
                    tobyte = towrite.tobytes(order="F")
                    file.write(tobyte)

                #save the def field
                D = def_field[0,:,:,:,:]
                with open(os.path.join(self.save_dir_def, phrase+"_def_field.im"),"wb") as file:
                    towrite = D.cpu().numpy().astype(np.float32)
                    tobyte = towrite.tobytes(order="F")
                    file.write(tobyte)
                
        if return_loss:
            return temp
        return
    
    def evaluate(self):
        dim = self.dim.cpu().numpy().astype(int).tolist()
        sz = self.sz.cpu().numpy().tolist()
        #evaluate validation data
        valid_defnames = [os.path.join(self.save_dir_def, keyword+"_def_field.im") for keyword in self.valid_keywords]
        gt_names = [self.gt_template.replace("%s", keyword) for keyword in self.valid_keywords]
        oob_names = []
        valid_95, _, valid_dice, valid_deformed_meshes = evaluator.evaluate_fnames(valid_defnames, dim, sz[0], gt_names, dim, sz[0], self.atlas_mesh_filename, self.atlas_mask_filename, sz[0], oob_names, return_meshes=True)
        valid_results_dict = {}
        valid_results_dict["dist_95s"] = valid_95
        valid_results_dict["dice_scores"] = valid_dice
        
        #evaluate test data
        test_defnames = [os.path.join(self.save_dir_def, keyword+"_def_field.im") for keyword in self.test_keywords]
        gt_names = [self.gt_template.replace("%s", keyword) for keyword in self.test_keywords]
        oob_names = []
        test_95, _, test_dice, test_deformed_meshes = evaluator.evaluate_fnames(test_defnames, dim, sz[0], gt_names, dim, sz[0], self.atlas_mesh_filename, self.atlas_mask_filename, sz[0], oob_names, return_meshes=True)
        test_results_dict = {}
        test_results_dict["dist_95s"] = test_95
        test_results_dict["dice_scores"] = test_dice
        
        #save results
        results_dict = {"valid":valid_results_dict, "test":test_results_dict}
        with open(os.path.join(self.progress_dir, "results.pkl"), "wb") as file:
            pickle.dump(results_dict, file)
            
        #save meshes
        for t in range(self.test_size):
            fname = os.path.join(self.save_mesh_dir, self.test_keywords[t] + "_result.mesh")
            parser.write_mesh_file(fname, test_deformed_meshes[t][0], test_deformed_meshes[t][1])
        for t in range(self.valid_size):
            fname = os.path.join(self.save_mesh_dir, self.valid_keywords[t] + "_result.mesh")
            parser.write_mesh_file(fname, valid_deformed_meshes[t][0], valid_deformed_meshes[t][1])
        return
        

    def save_slices(self, images, trans_pred, def_field, filenames, valid = False, epoch = 0):
        TP_im = trans_pred[0,0,:,:,:]
        TP_im = TP_im.cpu().numpy()
        D = np.linalg.norm(def_field.cpu().numpy()[0,:,:,:,:], axis = 0)
        phrase = os.path.basename(filenames[0])[0:os.path.basename(filenames[0]).find("_standardized")]
        phrase = phrase.split('.')[0]
        bestslice = 39 #central slice of atlas

        smallI = np.array(images[0,0,:,:,bestslice].cpu().detach())
        smallTP_im = TP_im[:,:,bestslice]
        smallD = D[:,:,bestslice]
        small_atlas_mask = self.atlas_mask[:,:,bestslice].cpu()
        small_atlas_im = self.atlas_image[:,:,bestslice].cpu()

        '''seg section'''
        scalar = 5
        fig= plt.figure(figsize=(int(4*scalar),int(3*scalar)))

        #input image #predicted seg #known seg
        #atlas mask #deformed mask #deformation magnituge
        #atlas im #deformed im #
        #deform x #deform y #deform z
        plt.subplot(431); plt.axis("off")
        p = plt.imshow(smallI); plt.gca().set_title("Image - slc: "+str(bestslice)); plt.colorbar(p, ax=plt.gca())
        plt.subplot(432); plt.axis("off")
        plt.subplot(433); plt.axis("off")


        '''def section'''
        #print(phrase, bestslice, end = " ")
        #bestslice = utils.get_slice_i(self.atlas_mask[0,:,:,:])
        #print(bestslice)
        smallI = np.array(images[0,0,:,:,bestslice].cpu().detach())
        smallTP_im = TP_im[:,:,bestslice]
        smallD = D[:,:,bestslice]
        small_atlas_mask = self.atlas_mask[:,:,bestslice].cpu()
        small_atlas_im = self.atlas_image[:,:,bestslice].cpu()

        #atlas mask #deformed mask #deformation magnituge
        plt.subplot(434); plt.axis("off")
        p = plt.imshow(small_atlas_mask); plt.gca().set_title("Atlas Mask - slc: "+str(bestslice)); plt.colorbar(p, ax=plt.gca())
        plt.subplot(435); plt.axis("off")
        plt.subplot(436); plt.axis("off")
        p = plt.imshow(smallD); plt.gca().set_title("Deform Mag"); plt.colorbar(p, ax=plt.gca())

        #atlas im #deformed im #
        plt.subplot(437); plt.axis("off")
        p = plt.imshow(small_atlas_im); plt.gca().set_title("Atlas Im"); plt.colorbar(p, ax=plt.gca())
        plt.subplot(438); plt.axis("off")
        p = plt.imshow(smallTP_im); plt.gca().set_title("Deformed Im"); plt.colorbar(p, ax=plt.gca())
        plt.subplot(439); plt.axis("off")
        #plt.imshow(smallD); plt.gca().set_title("Deform Mag")
        p = plt.imshow(smallI); plt.gca().set_title("Undeformed Im - slc: "+str(bestslice)); plt.colorbar(p, ax=plt.gca())

        #deform x #deform y #deform z
        plt.subplot(4,3,10); plt.axis("off")
        p = plt.imshow(def_field.cpu().numpy()[0,0,:,:,bestslice]); plt.gca().set_title("Deform X"); plt.colorbar(p, ax=plt.gca())
        plt.subplot(4,3,11); plt.axis("off")
        p = plt.imshow(def_field.cpu().numpy()[0,1,:,:,bestslice]); plt.gca().set_title("Deform Y"); plt.colorbar(p, ax=plt.gca())
        plt.subplot(4,3,12); plt.axis("off")
        p = plt.imshow(def_field.cpu().numpy()[0,2,:,:,bestslice]); plt.gca().set_title("Deform Z"); plt.colorbar(p, ax=plt.gca())

        if valid:
            plt.savefig(os.path.join(self.valid_dir_slices, phrase+"_epoch_"+str(epoch)+"_big.png"), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir_slices, phrase+"_big.png"), bbox_inches='tight')
        plt.close()

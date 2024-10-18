"""loss & accuracy functions"""
"""should be an object so it can store the global info"""
import utils
import math
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import importlib
import time

class LossManager():
    def __init__(self, learner):
        #weights, atlas
        #create specific loss lists
        self.learner = learner
        A_i = self.learner.atlas_image.repeat(learner.batch_size,1,1,1,1)
        A_m = self.learner.atlas_mask.repeat(learner.batch_size,1,1,1,1)
        self.combo_A = torch.cat([A_m, A_i, A_m], dim=1)

        #all of these losses are per epoch
        self.train_losses = []
        self.specific_names = [key for key in self.learner.losses.keys() if self.learner.losses[key]["weight"] > 0]
        self.specific_modules = [importlib.import_module(self.learner.losses[name]["filename"]) for name in self.specific_names]
        self.specific_weights = [self.learner.losses[name]["weight"] for name in self.specific_names]
        self.specific_running = [0 for i in self.specific_names]
        self.specific_losses = [[] for i in self.specific_names]

        self.valid_losses = []

        self.running_train = 0
        self.running_valid = 0
        return

    def save_losses(self):
        tfile = open(os.path.join(self.learner.trial_dir,'train_'+self.learner.trialname+'.csv'),'w')
        vfile = open(os.path.join(self.learner.trial_dir,'valid_'+self.learner.trialname+'.csv'),'w')
        for i in range(self.learner.epochs):
            tfile.write(str(self.train_losses[i])+"\n")
            vfile.write(str(self.valid_losses[i])+"\n")
        tfile.close()
        vfile.close()
        return

    def make_plots(self):
        if not self.learner.verbose_learner:
            return
        ee = [i+1 for i in range(self.learner.epochs)]

        tt = self.train_losses
        vv = self.valid_losses

        plt.figure()
        plt.plot(ee,np.log(tt))
        plt.plot(ee, np.log(vv))
        plt.legend(["Train", "Valid"])
        plt.title("Log loss per epoch")

        plt.figure()
        plt.plot(ee,np.log(tt))
        plt.title("Train log loss per epoch")

        plt.figure()
        plt.plot(ee,np.log(vv))
        plt.title("Valid log loss per epoch")
        
        plt.figure()
        for i in range(len(self.specific_losses)):
            plt.plot(ee, np.log(self.specific_losses[i]))
        plt.legend(self.specific_names)
        plt.title("Log Loss")
        
        for i in range(len(self.specific_losses)):
            plt.figure()
            plt.plot(ee, np.log(self.specific_losses[i]))
            plt.title(self.specific_names[i] + " log loss")

        plt.show()

        return

    def total_loss(self, images, meshes, trans_pred, def_field, train = True, record = True):
        batch_size = images.shape[0]
        small_combo_A = self.combo_A[0:batch_size, :, :, :, :]
        
        combined_loss = 0
        for i in range(len(self.specific_modules)):
            mod = self.specific_modules[i]
            weight = self.specific_weights[i]
            val = weight*mod.loss_func(self.learner, small_combo_A, trans_pred, def_field, meshes)
            if self.learner.verbose:
                print("\t",self.specific_names[i],":",val.item())
            if record and train:
                self.specific_running[i] = self.specific_running[i] + val.item()*batch_size
            combined_loss = combined_loss + val
        if record:
            if train:
                self.running_train = self.running_train + combined_loss.item()*batch_size
            else:
                self.running_valid = self.running_valid + combined_loss.item()*batch_size
        if torch.isnan(combined_loss):
            raise ValueError("Combined loss is NaN, investigation required.")
        return combined_loss

    def end_epoch(self, epoch = None):
        epoch_train = self.running_train/self.learner.train_size
        epoch_valid = self.running_valid/self.learner.valid_size
        for i in range(len(self.specific_losses)):
            self.specific_losses[i].append(self.specific_running[i]/self.learner.train_size)
            self.specific_running[i] = 0
        self.train_losses.append(epoch_train)
        self.valid_losses.append(epoch_valid)
        self.running_train = 0
        self.running_valid = 0
        if self.learner.verbose_learner:
            if epoch is not None:
                print("Epoch: ", epoch, end = " , ")
            print("train:", epoch_train,",","valid:",epoch_valid)
        return epoch_train, epoch_valid
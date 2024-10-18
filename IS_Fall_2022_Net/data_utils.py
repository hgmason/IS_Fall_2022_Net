"""dataset, dataloader, save slices, visualizer"""
from torch.utils.data import Dataset
import torch
import utils
import numpy as np

#define data loaders
class Data3D(Dataset):
    def __init__(self, keywords, learner, threshold = .5):
        self.keywords = keywords
        self.threshold = threshold
        self.dim = learner.dim
        self.image_filenames = [learner.img_template.replace("%s", keyword) for keyword in keywords]
        self.mesh_filenames  = [learner.mesh_template.replace("%s", keyword) for keyword in keywords]
        return

    def __len__(self):
        return len(self.image_filenames)

    def transform(self, image, scaling = None):
        image = torch.Tensor(image.astype(np.float32))
        if scaling == "A" or scaling == "M": #for "annotation" or "mask"
            image = image - image.min()
            if image.max() > 0:
                image = image / image.max()
            image = ((image > self.threshold) + 1.0) - 1.0
            image = torch.Tensor(image).float()
        image = image.unsqueeze(0)
        return image

    def __getitem__(self, index):
        im_filename = self.image_filenames[index]
        image = utils.read_raw_volume_file(im_filename, self.dim, np.float32)
        image = self.transform(image, scaling = "I")
        image = torch.Tensor(image)
        
        mesh_filename = self.mesh_filenames[index]
        mesh_verts = torch.Tensor(utils.read_mesh_verts(mesh_filename)).float()
        
        return image, mesh_verts, self.image_filenames[index]

# IS_Fall_2022_Net
An atlas-based registration network created for my Fall 2022 Independent Study

## Jupyter Notebook Example
See Run-Net.ipynb for an example.

# Data Preprocessing and Structure
### Preprocessing
This network expects that
- All the training/validation/testing images are affinely registered to the atlas image.
  - And each have a corresponding "rough guess" mesh (Class from the IS_Fall_2022_Net/parser.py)
- The dimension and voxel size is constant in all 3 dimensions, though the actual value is configurable.
- The training/validation/testing/gt images are raw image files written with a byte order of 'F' (Fortran-style order), with a float32/single datatype
- The atlas image has a corresponding mask, mesh, point correspondence mesh, and distance map
  - The image, mask, and distance map should be .nrrd files
  - The mesh is a Mesh class, defined in IS_Fall_2022_Net/parser.py
- All the masks (atlas/gt) are binary masks with black = 0 and white = 255
- All the images are scaled such that the min is around 0 and the max is around 1 (doesn't have to be exact)
  - If there is any "out of bounds" areas that happen due to the affine registration and cropping issues, fill them in with the mean value of the rest of the image
  
### File Structure
This network expects that each training/validation/testing case has a unique patient ID, also called a "keyword". It expects that there's a file where the keywords are listed, in three sections separated by double newlines, for training/validation/testing respectively. For example:

train_keyword1<br>
train_keyword2<br>
train_keyword3<br>

valid_keyword1<br>
valid_keyword2<br>
valid_keyword3<br>

test_keyword1<br>
test_keyword2<br>
test_keyword3<br>

It also expects that there's an image directory, mesh directory, and ground truth directory, with files following a consistent structure. These structures can be defined in the img_template, mesh_template, and gt_template sections of the parameter dictionary.

## Learner Object
The "learner" is an object which handles all of the network's training, testing, etc. The goal is to separate the "input" and "processing" as much as possible, for maximal versatility. Its main functions are as follows.
- train(): train the model using the given parameter dictionary
- export(): create and save the deformation fields for all the valid/test cases
- evaluate(): create and save the deformed meshes for all the valid/test cases, as well as evaluate their accuracy using the Dice Score and 95th Percentile Hausdorff Distance metrics. The resulting values are saved.

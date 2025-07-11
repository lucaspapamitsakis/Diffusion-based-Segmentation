import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from torchvision.transforms import Resize
# from monai.data import Dataset, DataLoader
# from monai.transforms import (
#     Compose,
#     LoadImaged,
#     Resized,
#     ScaleIntensityd,
#     SqueezeDimd,
#     RandRotated,
#     RandZoomd,
#     RandFlipd,
#     EnsureChannelFirstd,
#     EnsureTyped,
#     RandRotate90d,
#     ResizeWithPadOrCropd
# )

# root = "../data"
# split = "testing"
# directory = os.path.join(root_dir, split)

directory = os.path.expanduser('../data')
database = []

for root, dirs, files in os.walk(directory):
    # print(root, dirs, files)
    # print("-"*20)
    # if there are no subdirs, we have data
    if not dirs:
        mr_files = sorted([f for f in files if f.startswith('mr')])
        seg_files = sorted([f for f in files if f.startswith('seg')])

        for mr_file in mr_files:
            # Construct the corresponding segmentation file name by replacing 'mr' with 'seg'
            seg_file = 'seg' + mr_file[2:]
            if seg_file in seg_files:
                database.append({'mr': os.path.join(root, mr_file), 'seg': os.path.join(root, seg_file)})

print(database)



datapoint = database[0]
mr_path = datapoint["mr"]
bone_path = datapoint["seg"]

# Load the 3D images
mr_img_nii = nibabel.load(mr_path)
bone_img_nii = nibabel.load(bone_path)

mr_img_data = torch.tensor(mr_img_nii.get_fdata(), dtype=torch.float32)
bone_img_data = torch.tensor(bone_img_nii.get_fdata(), dtype=torch.float32)

processed_slices = []
for i in range(mr_img_data.shape[2]): 
    mr_slice = mr_img_data[:, :, i].unsqueeze(0)
    bone_slice = bone_img_data[:, :, i].unsqueeze(0)

    resize_transform = Resize((219, 252), antialias=True)
    mr_slice_resized = resize_transform(mr_slice)
    bone_slice_resized = resize_transform(bone_slice)
    
    mr_slice_resized = (mr_slice_resized - mr_slice_resized.min()) / \
                        (mr_slice_resized.max() - mr_slice_resized.min())
    
    bone_slice_resized = torch.where(bone_slice_resized > 0, 1.0, 0.0)
    
    processed_slices.append((mr_slice_resized, bone_slice_resized))
    
print(len(processed_slices))
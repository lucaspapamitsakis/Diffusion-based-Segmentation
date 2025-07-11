import torch
import torch.nn
import numpy as np
import os
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




class MRBoneDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        """
        Initializes the dataset. This constructor will now iterate through all
        3D volumes to create a flat list of all available 2D slices.
        """
        super().__init__()
        self.transform = transform
        self.slice_map = [] # This will store a reference to every single slice

        # First, find all the paired 3D volumes
        volume_paths = []
        for root, _, files in os.walk(os.path.expanduser(directory)):
            if not os.listdir(root): # Check if the directory is empty
                continue
            mr_files = sorted([f for f in files if f.startswith('mr')])
            seg_files = sorted([f for f in files if f.startswith('seg')])

            for mr_file in mr_files:
                seg_file = 'seg' + mr_file[2:]
                if seg_file in seg_files:
                    volume_paths.append({
                        'mr': os.path.join(root, mr_file),
                        'seg': os.path.join(root, seg_file)
                    })

        # ## --- KEY CHANGE: Create a map of every slice --- ##
        # Now, iterate through the volumes to map out each individual slice
        for volume in volume_paths:
            # We load the image here just to get its dimensions
            mr_img_nii = nibabel.load(volume['mr'])
            num_slices = mr_img_nii.shape[2] # Get the number of axial slices

            for i in range(num_slices):
                self.slice_map.append({
                    "mr_path": volume['mr'],
                    "seg_path": volume['seg'],
                    "slice_index": i
                })

    def __len__(self):
        # ## --- KEY CHANGE: The length is the total number of 2D slices --- ##
        return len(self.slice_map)

    def __getitem__(self, idx):
        # ## --- KEY CHANGE: Load and process only ONE slice --- ##

        # Get the information for the requested slice
        slice_info = self.slice_map[idx]
        mr_path = slice_info["mr_path"]
        seg_path = slice_info["seg_path"]
        slice_index = slice_info["slice_index"]

        # Load the 3D volumes
        mr_img_nii = nibabel.load(mr_path)
        bone_img_nii = nibabel.load(seg_path)

        mr_img_data = torch.tensor(mr_img_nii.get_fdata(), dtype=torch.float32)
        bone_img_data = torch.tensor(bone_img_nii.get_fdata(), dtype=torch.float32)

        # Extract the specific slice
        mr_slice = mr_img_data[:, :, slice_index].unsqueeze(0)   # Add channel dim
        bone_slice = bone_img_data[:, :, slice_index].unsqueeze(0) # Add channel dim

        # Apply transformations
        resize_transform = Resize((219, 252), antialias=True)
        mr_slice_resized = resize_transform(mr_slice)
        bone_slice_resized = resize_transform(bone_slice)
        
        # Normalize the MR slice
        mr_max = mr_slice_resized.max()
        mr_min = mr_slice_resized.min()
        if mr_max > mr_min:
            mr_slice_resized = (mr_slice_resized - mr_min) / (mr_max - mr_min)
        
        # Binarize the bone mask
        bone_slice_resized = torch.where(bone_slice_resized > 0, 1.0, 0.0)
        
        # Note: If you passed in transforms (like from MONAI or torchvision),
        # you would apply them here. For example:
        # sample = {'mr': mr_slice_resized, 'seg': bone_slice_resized}
        # if self.transform:
        #     sample = self.transform(sample)
        # return sample['mr'], sample['seg']
        
        return mr_slice_resized, bone_slice_resized

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

class MRBoneDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        '''
        directory is expected to be the path to your training or testing folder.
        This loader will find all files starting with "mr" and pair them
        with their corresponding "seg" file.
        e.g., ./data/training/mr1BA001.nii.gz and ./data/training/seg1BA001.nii.gz
        '''

        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.database = []
        self.transform = transform

        # self.test_flag=test_flag
        # if test_flag:
        #     self.seqtypes = ['mr']
        # else:
        #     self.seqtypes = ['mr', 'seg']

        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                mr_files = sorted([f for f in files if f.startswith('mr')])
                seg_files = sorted([f for f in files if f.startswith('seg')])

                for mr_file in mr_files:
                    # Construct the corresponding segmentation file name by replacing 'mr' with 'seg'
                    seg_file = 'seg' + mr_file[2:]
                    if seg_file in seg_files:
                        self.database.append({'mr': os.path.join(root, mr_file), 'seg': os.path.join(root, seg_file)})        


    def __getitem__(self, x):
        
        datapoint = self.database[x]
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
            
        return processed_slices

        # This returns a list of all 2D slices for a given 3D volume.
        # Likely flatten this list in training script.

    def __len__(self):
        return len(self.database)


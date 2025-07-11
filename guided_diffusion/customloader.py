import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    SqueezeDimd,
    RandRotated,
    RandZoomd,
    RandFlipd,
    EnsureChannelFirstd,
    EnsureTyped,
    RandRotate90d,
    ResizeWithPadOrCropd
)

class MRBoneDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, test_flag=False):
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


    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            return (image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            return (image, label)

    def __len__(self):
        return len(self.database)




# def view_3d_slice_comparison(original_pair, transformed_pair, target_hw):
#     """
#     Displays a central 2D slice from a pair of original and resized 3D volumes.
#     """
#     fig, axes = plt.subplots(2, 2, figsize=(12, 12))

#     # --- Original Images (as NumPy arrays) ---
#     original_image_np = original_pair["img"]
#     original_label_np = original_pair["seg"]

#     # Get the middle slice index for the Z-axis (depth)
#     slice_idx_img = original_image_np.shape[2] // 2
#     slice_idx_lbl = original_label_np.shape[2] // 2

#     axes[0, 0].imshow(np.rot90(original_image_np[:, :, slice_idx_img]), cmap="gray")
#     axes[0, 0].set_title(f"Original Image Slice\nShape: {original_image_np.shape}")
#     axes[0, 0].axis('off')

#     axes[0, 1].imshow(np.rot90(original_label_np[:, :, slice_idx_lbl]), cmap="gray")
#     axes[0, 1].set_title(f"Original Label Slice\nShape: {original_label_np.shape}")
#     axes[0, 1].axis('off')

#     # --- Transformed Images (as PyTorch Tensors) ---
#     transformed_image_tensor = transformed_pair["img"].squeeze()
#     transformed_label_tensor = transformed_pair["seg"].squeeze()

#     # Get the middle slice for the resized volumes
#     resized_slice_idx = transformed_image_tensor.shape[2] // 2

#     axes[1, 0].imshow(np.rot90(transformed_image_tensor[:, :, resized_slice_idx]), cmap="gray")
#     axes[1, 0].set_title(f"Resized Image Slice\nTarget HxW: {target_hw} (Depth Unchanged)")
#     axes[1, 0].axis('off')

#     axes[1, 1].imshow(np.rot90(transformed_label_tensor[:, :, resized_slice_idx]), cmap="gray")
#     axes[1, 1].set_title(f"Resized Label Slice\nFinal Shape: {transformed_label_tensor.shape}")
#     axes[1, 1].axis('off')

#     plt.tight_layout()
#     plt.show()

# def main2(data_dicts, target_hw_size=(219, 252)):
#     """
#     Main function to load and transform paired 3D images, resizing only Height and Width.
#     """

#     keys = ["img", "seg"]

#     # KEY CHANGE HERE: Define the spatial size for resizing.
#     # Use -1 for the third dimension (axial/depth) to keep its original size.
#     resize_shape = (target_hw_size[0], target_hw_size[1], -1)

#     # Define transformations for resizing 3D volumes
#     resize_transforms = Compose([
#         LoadImaged(keys=keys, image_only=True),
#         EnsureChannelFirstd(keys=keys),
#         # Orientationd(keys=keys, axcodes="RAS"), # Reorient to a standard orientation
#         Resized(keys=keys, spatial_size=resize_shape, mode='trilinear'),
#         EnsureTyped(keys=keys, dtype=torch.float32),
#     ])

#     # Create MONAI Dataset and DataLoader for transformed data
#     resize_ds = Dataset(data=data_dicts, transform=resize_transforms)
#     resize_loader = DataLoader(resize_ds, batch_size=1)

#     # Create a simple loader for the original data for comparison
#     original_loader = DataLoader(
#         Dataset(data=data_dicts, transform=LoadImaged(keys=keys, image_only=True)),
#         batch_size=1
#     )

#     # Iterate and view the image slices
#     for original_batch, transformed_batch in zip(original_loader, resize_loader):
#         original_pair = {key: original_batch[key][0] for key in keys}
#         transformed_pair = {key: transformed_batch[key][0] for key in keys}

#         view_3d_slice_comparison(original_pair, transformed_pair, target_hw_size)
        
#         break

    
# # Define your desired output size
# new_size = (219, 252)

# main2(data_dicts=train_files, target_hw_size=new_size)
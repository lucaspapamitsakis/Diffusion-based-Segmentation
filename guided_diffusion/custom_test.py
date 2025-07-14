import sys
sys.path.append("..")
sys.path.append(".")
import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from torchvision.transforms import Resize
import matplotlib.pyplot as plt 
from guided_diffusion.customloader import MRBoneDataset
from guided_diffusion.customloader_norm import MRBoneDatasetNorm
from guided_diffusion.bratsloader import BRATSDataset


##################################


# def view_3d_slice_comparison(original_pair, transformed_pair):
#     """
#     Displays a central 2D slice from a pair of original and resized 3D volumes.
#     """
#     fig, axes = plt.subplots(2, 2, figsize=(12, 12))

#     # --- Original Images (as NumPy arrays) ---
#     original_image_np = original_pair[0].squeeze()
#     original_label_np = original_pair[1].squeeze()

#     # --- Transformed Images (as PyTorch Tensors) ---
#     transformed_image_tensor = transformed_pair[0].squeeze()
#     transformed_label_tensor = transformed_pair[1].squeeze()


#     axes[0, 0].imshow(np.rot90(original_image_np), cmap="gray")
#     axes[0, 0].set_title(f"Original MR Slice\nShape: {original_image_np.shape}")
#     axes[0, 0].axis('off')

#     axes[0, 1].imshow(np.rot90(transformed_image_tensor), cmap="gray")
#     axes[0, 1].set_title(f"Resized MR Slice\nShape: {transformed_image_tensor.shape}")
#     axes[0, 1].axis('off')


#     axes[1, 0].imshow(np.rot90(original_label_np), cmap="gray")
#     axes[1, 0].set_title(f"Original Seg Slice\Shape: {original_label_np.shape}")
#     axes[1, 0].axis('off')

#     axes[1, 1].imshow(np.rot90(transformed_label_tensor), cmap="gray")
#     axes[1, 1].set_title(f"Resized Seg Slice\nFinal Shape: {transformed_label_tensor.shape}")
#     axes[1, 1].axis('off')

#     plt.tight_layout()
#     plt.show()

########################################

dat_path = "../data/training/"
# dat_path = "../data/"
dat_path2 = "../data_eg/training/000001"
ds_resize = MRBoneDataset(dat_path)
# ds = MRBoneDatasetNorm(dat_path)
ds = BRATSDataset(dat_path2)
data_loader= torch.utils.data.DataLoader(
    ds,
    batch_size=1,
    shuffle=False)

data_resize_loader = torch.utils.data.DataLoader(
    ds_resize,
    batch_size=1,
    shuffle=False)

print(len(data_loader))
data = iter(data_loader)

for batch in data:
    x = batch[0]
    y = batch[1]
    print(x.shape)
    print(y.shape)
    break





################################################
# logger.log("training...")
# TrainLoop(
#     model=model,
#     diffusion=diffusion,
#     classifier=None,
#     data=data,
#     dataloader=datal,
#     batch_size=args.batch_size,
#     microbatch=args.microbatch,
#     lr=args.lr,
#     ema_rate=args.ema_rate,
#     log_interval=args.log_interval,
#     save_interval=args.save_interval,
#     resume_checkpoint=args.resume_checkpoint,
#     use_fp16=args.use_fp16,
#     fp16_scale_growth=args.fp16_scale_growth,
#     schedule_sampler=schedule_sampler,
#     weight_decay=args.weight_decay,
#     lr_anneal_steps=args.lr_anneal_steps,
# ).run_loop()
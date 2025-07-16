"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
from visdom import Visdom
viz = Visdom(port=8097)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.customloader import MRBoneDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")

    # # --- Start of Data Loading ---
    # # Get all file paths from the data directory
    # all_files = [f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))]
    
    # # Extract unique volume IDs from filenames (e.g., "volume_01" from "volume_01.nii.gz")
    # # volume_ids = sorted(list(set([os.path.splitext(os.path.basename(f))[0] for f in all_files])))
    # volume_ids = sorted([f.split(".")[0] for f in all_files])
    
    # logger.log(f"Found volume IDs: {volume_ids}")

    # Instantiate your custom MRBoneDataset
    ds = MRBoneDataset(args.data_dir, test_flag=True)
    
    data = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False
    )
    # --- End of Modified Data Loading ---
    
    logger.log("creating samples...")

    # --- Start of Modified Sampling Loop ---
    # unpacks the output from MRBoneDataset
    for j, (img, _, slice_info) in enumerate(data):
        
        # Extract volume_id and slice_index for a unique filename
        # Assumes a batch size of 1
        volume_id = slice_info["mr_path"][0].split("1B")[1][:4]
        slice_index = slice_info["slice_index"].item()
        
        # Create a descriptive and unique ID, e.g., "volume_01_slice_025"
        slice_ID = f"{volume_id}_slice_{slice_index:03d}"
        
        c = th.randn_like(img[:, :1, ...])
        img = th.cat((img, c), dim=1)     #add a noise channel


        viz.image(visualize(img[0,0,...]), opts=dict(caption="img input0"))
        viz.image(visualize(img[0, 1, ...]), opts=dict(caption="img input1"))

        logger.log(f"sampling {slice_ID}...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)


        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            shape = (args.batch_size, 2, args.image_size, args.image_size)

            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                shape,
                img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            s = th.tensor(sample)
            viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            sample_path = f'./results/{slice_ID}_output{i}.pt'
            th.save(s, sample_path) #save the generated mask

        
        # The rest of the saving logic remains the same
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        # sample = sample.contiguous()

        # out_path = os.path.join(logger.get_dir(), f"sample_{slice_ID}.npy")
        # np.save(out_path, sample.cpu().numpy())

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
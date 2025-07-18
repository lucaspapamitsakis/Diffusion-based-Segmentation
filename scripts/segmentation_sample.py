"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
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

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_coefficient(pred, target, smooth=1e-5):

    target = target.to(pred.device)
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return dice.mean()

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(format_strs=['stdout', 'log', 'tensorboard'])

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
    
    ds = MRBoneDataset(args.data_dir, test_flag=True)
    
    data = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    logger.log("creating samples...")

    all_dice_scores = []

    for j, (img, seg, slice_info) in enumerate(data):
        
        volume_id = slice_info["mr_path"][0].split("1B")[1][:4]
        slice_index = slice_info["slice_index"].item()
        
        slice_ID = f"{volume_id}_slice_{slice_index:03d}"
        
        c = th.randn_like(img[:, :1, ...])
        img = th.cat((img, c), dim=1)     #add a noise channel

        # Log input MR and ground truth to TensorBoard
        # NOTE that the above code concatenates the noise to the (mr) img var, so we must do the double-0 idx & unsqueeze for the MR
        logger.log_image(f"sample_{slice_ID}/1_input_MR", visualize(img[0,0, ...]).unsqueeze(0), j)
        # logger.log_image(f"sample_{slice_ID}/input_Noise", visualize(img[0, 1, ...]).unsqueeze(0), j) # THIS IS THE NOISE
        logger.log_image(f"sample_{slice_ID}/2_ground_Truth", visualize(seg[0, ...]), j)

        logger.log(f"sampling {slice_ID}...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        # Make a dummy var to hold the ensemble sample info
        # sample_ensemble = th.zeros_like(img[:, :1, ...])
        sample_ensemble = []

        for i in range(args.num_ensemble):
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
            print(f'time for 1 sample ({slice_ID}_output{i}):', start.elapsed_time(end))

            s = th.tensor(sample)
            sample_ensemble.append(s)
            # sample_ensemble = torch.cat((sample_ensemble, s), dim=1)

            # Log the generated sample to TensorBoard
            # logger.log_image(f"sample_{slice_ID}/output_{i}", visualize(s[0, 0, ...]).unsqueeze(0), j)
            
            # sample_path = f'./results/{slice_ID}_output{i}.pt'
            # th.save(s, sample_path)

        sample_ensemble = th.cat(sample_ensemble, dim=1)
        mean_ensemble = th.mean(sample_ensemble, dim=1, keepdim=True)
        var_ensemble = th.var(sample_ensemble, dim=1, keepdim=True)
        # Binarize the ensemble prediction segment
        pred_ensemble = th.where(mean_ensemble > 0.5, 1.0, 0.0)

        logger.log_image(f"sample_{slice_ID}/3_ensemble_threshold", visualize(pred_ensemble[0, ...]), j)
        logger.log_image(f"sample_{slice_ID}/4_ensemble_mean", visualize(mean_ensemble[0, ...]), j)
        logger.log_image(f"sample_{slice_ID}/5_ensemble_var", visualize(var_ensemble[0, ...]), j)

        # Calculate and store the dice score
        dice_score = dice_coefficient(pred_ensemble, seg)
        all_dice_scores.append(dice_score.item())

        # # Calculate moving average Dice
        running_avg_dice = np.mean(all_dice_scores)

        # #  Log the running average to TensorBoard for this step (slice j)
        logger.logkv("eval/running_average_dice", running_avg_dice)
        
        # # Log the individual dice score for this slice as well
        logger.logkv("eval/slice_dice_score", dice_score.item())

        # # Critical (or maybe not??) to Log the slice number 'j' as the 'step'
        # logger.logkv("step", j)

        # # Dump the key-values. TensorBoard will now use 'j' as the step.
        logger.dumpkvs()


        # logger.log(f"Dice score for {slice_ID}: {dice_score.item()}")
        logger.log(
            f"Slice {slice_ID} | "
            f"Dice: {dice_score.item():.4f} | "
            f"Running Avg Dice: {running_avg_dice:.4f}"
        )

    # After the loop, calculate and log the average dice score
    avg_dice = np.mean(all_dice_scores)
    std_dice = np.std(all_dice_scores)
    logger.log(f"Final Average Dice Score: {avg_dice:.4f} (± {std_dice:.4f})")

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
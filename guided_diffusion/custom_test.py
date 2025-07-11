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
from guided_diffusion.customloader import MRBoneDataset
from guided_diffusion.bratsloader import BRATSDataset


dat_path = "../data/training"
# dat_path = "../data/"
# dat_path2 = "../data_eg"
ds = MRBoneDataset(dat_path)
# ds = BRATSDataset(dat_path2)
datal= torch.utils.data.DataLoader(
    ds,
    batch_size=1,
    shuffle=True)

print(len(datal))
data = iter(datal)

for batch in data:
    x = batch[0]
    y = batch[1]
    print(x.shape)
    print(y.shape)
    break


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
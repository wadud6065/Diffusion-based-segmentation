"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from LIDCLoader import load_LIDC
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def show_tensor_images(image, mask, tensor_array, figsize=(10, 2), title=None, cmap='viridis', columns=6, num = 1): 
    num_tensors = len(tensor_array)+2
    rows = (num_tensors + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    to_pil = transforms.ToPILImage()
    
    if title:
        fig.suptitle(title, fontsize=12)
    
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i == 0:
            tensor = image.squeeze().cpu()
            tensor = to_pil(tensor)
            ax.imshow(tensor, cmap=cmap)
            ax.set_title('Image', fontsize=8)
            continue
        if i == 1:
            tensor = mask.squeeze().cpu()
            tensor = to_pil(tensor)
            ax.imshow(tensor, cmap=cmap)
            ax.set_title('Ground Truth', fontsize=8)
            continue
        if i < num_tensors:
            i = i - 2
            tensor = tensor_array[i].squeeze().cpu()
            tensor = to_pil(tensor)
            ax.imshow(tensor, cmap=cmap)
            ax.set_title('Output'+str(i), fontsize=8)
    
    # Hide any empty subplots
    for i in range(num_tensors, rows * columns):
        fig.delaxes(axes.flatten()[i])
    
    plt.savefig('./output_images/output_figure'+str(num)+'.png')
    plt.show()

def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # ds = BRATSDataset(args.data_dir, test_flag=True)
    # datal = th.utils.data.DataLoader(
    #     ds,
    #     batch_size=1,
    #     shuffle=False)
    # data = iter(datal)

    output_dir = './output'
    output_img_dir = './output_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    ds = load_LIDC(image_size=224, combine_train_val=True, mode='Test')
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)

    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    title = ''
    cnt = 0
    while len(all_images) * args.batch_size < args.num_samples:
        # should return an image from the dataloader "data"
        b, mask, image_path, mask_path = next(data)
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)  # add a noise channel$
        print(image_path)
        # slice_ID = path[0].split("/", -1)[3]
        slice_ID = os.path.basename(image_path[0]).split(".")[0]
        title = os.path.basename(mask_path[0]).split(".")[0]

        # viz.image(visualize(img[0,0,...]), opts=dict(caption="img input0"))
        # viz.image(visualize(img[0, 1, ...]), opts=dict(caption="img input1"))
        # viz.image(visualize(img[0, 2, ...]), opts=dict(caption="img input2"))
        # viz.image(visualize(img[0, 3, ...]), opts=dict(caption="img input3"))
        # viz.image(visualize(img[0, 4, ...]), opts=dict(caption="img input4"))

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        
        tensor_list = []
        # this is for the generation of an ensemble of 5 masks.
        for i in range(args.num_ensemble):
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            # time measurement for the generation of 1 sample
            print('time for 1 sample', start.elapsed_time(end))

            s = sample.clone().detach()
            tensor_list.append(s)
            # viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            th.save(s, './output/'+str(slice_ID)+'_output' +str(i))  # save the generated mask
        
        show_tensor_images(image=b, mask=mask, tensor_array=tensor_list, title=title, num=cnt)
        tensor_list.clear()
        cnt = cnt + 1


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5  # number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()

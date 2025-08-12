from __future__ import annotations

import os
import os.path as osp
import sys
import warnings
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

# Add local dataloaders path
sys.path.append("dataloaders")

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio
from skimage import img_as_ubyte

from lib import VideoModel_pvtv2 as Network
from dataloaders.video_list import test_dataset as test_dataloader  # noqa
from utils.utils import post_process  # noqa


def to_device(x, device):
    if isinstance(x, list):
        return [t.to(device, non_blocking=True) for t in x]
    return x.to(device, non_blocking=True)


def ensure_str(name) -> str:
    # Some loaders return a list with a single str; normalize to str
    if isinstance(name, (list, tuple)) and len(name) == 1:
        return name[0]
    return name


@torch.no_grad()
def validate(
    test_loader,
    model: torch.nn.Module,
    output_dir: str,
    device: torch.device,
    sequence: Optional[str] = None,
    binarize_threshold: float = 0.0,
) -> None:
    """
    Runs inference over the test_loader and writes out masks for selected frames.

    Args:
        test_loader: iterable with a `.size` and `.load_data()` API.
        model: torch model.
        output_dir: directory where predictions are written.
        device: torch device.
        sequence: optional sequence override for output subfolder.
        binarize_threshold: threshold for binarizing logits/probabilities to {0,1}.
    """
    model.eval()
    model.to(device)

    for _ in tqdm(range(getattr(test_loader, "size", len(test_loader)))):
        (
            images,           # list[Tensor]
            images_ycbcr,     # list[Tensor]
            images_sam,       # list[Tensor] or Tensor list-like
            points,           # unused here
            gt,               # np / tensor-ish mask
            name,             # str or [str]
            scene,            # str
        ) = test_loader.load_data()

        name = ensure_str(name)
        scene = ensure_str(scene)

        # Normalize GT to [0,1] float32 np array (spatial dims only)
        gt_np = np.asarray(np.squeeze(gt), dtype=np.float32)
        gt_np /= (gt_np.max() + 1e-8)

        # Move inputs to device
        images = to_device(images, device)
        images_ycbcr = to_device(images_ycbcr, device)

        # In the original code, only the last element of images_sam is used
        if isinstance(images_sam, list) and len(images_sam) > 0:
            images_sam = images_sam[-1]
        images_sam = to_device(images_sam, device)

        # Forward
        pred = model(images, images_ycbcr, images_sam)

        # Save only on certain frames (mimic original behavior)
        # name[-5] assumes filenames like 'xxxx0.jpg'/'xxxx5.jpg', keep as-is but safer:
        if isinstance(name, str) and len(name) >= 5 and name[-5] in ["0", "5"]:
            # Decide subfolder
            subfolder = sequence if sequence is not None else scene
            sam_path = osp.join(output_dir, subfolder)
            os.makedirs(sam_path, exist_ok=True)

            # Use last prediction (commonly multi-scale list); resize to GT HxW
            pred_last = pred[-1] if isinstance(pred, (list, tuple)) else pred
            out = F.interpolate(
                pred_last, size=gt_np.shape, mode="bilinear", align_corners=False
            )

            # Binarize logits/probabilities -> {0,1}
            out = torch.where(out > binarize_threshold, torch.ones_like(out), torch.zeros_like(out))

            # to numpy HxW
            out_np = out.detach().cpu().numpy().squeeze()
            out_np = post_process(out_np)  # keep your existing post-processing

            # Save as PNG
            out_name = name
            if out_name.lower().endswith(".jpg"):
                out_name = out_name[:-4] + ".png"
            elif not out_name.lower().endswith(".png"):
                out_name = out_name + ".png"

            imageio.imwrite(osp.join(sam_path, out_name), img_as_ubyte(out_np))


def build_model_and_load(opt) -> torch.nn.Module:
    """
    Creates the model, puts it on the primary device, and loads weights.
    """
    device = torch.device(f"cuda:{opt.gpu_ids[0]}") if torch.cuda.is_available() else torch.device("cpu")
    model = Network(opt)

    if torch.cuda.is_available():
        if len(opt.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        model = model.to(device)

    if opt.resume and osp.exists(opt.resume):
        # When using DataParallel, state dict keys may be prefixed with 'module.'
        state = torch.load(opt.resume, map_location="cpu")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            # Try to fix mismatch between DP/non-DP checkpoints
            from collections import OrderedDict
            new_state = OrderedDict()
            if any(k.startswith("module.") for k in state.keys()):
                # Strip 'module.'
                for k, v in state.items():
                    new_state[k.replace("module.", "", 1)] = v
                model.load_state_dict(new_state, strict=False)
            else:
                # Add 'module.' if model is DataParallel
                if isinstance(model, torch.nn.DataParallel):
                    for k, v in state.items():
                        new_state["module." + k] = v
                    model.load_state_dict(new_state, strict=False)
                else:
                    raise
        print(f"[OK] Loaded checkpoint: {opt.resume}")
    else:
        print(f"[WARN] Checkpoint not found at: {opt.resume}")

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainsize", type=int, default=352, help="training dataset size")
    parser.add_argument("--testsize", type=int, default=352, help="testing dataset size")
    parser.add_argument("--grid", type=int, default=8, help="grid")
    parser.add_argument("--gpu_ids", type=int, default=[0, 1], nargs="+", help="gpu ids (e.g., 0 1)")
    parser.add_argument("--resume", type=str, default="./snapshot/best_checkpoint.pth", help="path to checkpoint")
    parser.add_argument("--dataset", type=str, default="MoCA", help="dataset name (e.g., MoCA, DAVIS)")
    parser.add_argument("--test_path", type=str, default="../../dataset/MoCA-Mask/TestDataset_per_sq", help="test dataset path")
    parser.add_argument("--sequence", type=str, default=None, help='specific sequence to process (e.g., "hike")')
    parser.add_argument("--output_dir", type=str, default="./pred", help="output directory")
    parser.add_argument("--threshold", type=float, default=0.0, help="binarization threshold (logits/prob > thr -> 1)")
    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu_ids[0]}") if torch.cuda.is_available() else torch.device("cpu")

    # Build model and load weights
    model = build_model_and_load(opt)

    # Load datasets
    print("===> Loading datasets")
    val_loader = test_dataloader(
        dataset=opt.dataset,
        testsize=opt.testsize,
        test_path=opt.test_path,
        sequence=opt.sequence,
        grid=opt.grid
    )
    print('Test with %d image pairs' % len(val_loader))

    # Validate
    validate(
        test_loader=val_loader,
        model=model,
        output_dir=opt.output_dir,
        device=device,
        sequence=opt.sequence,
        binarize_threshold=opt.threshold,
    )


if __name__ == "__main__":
    main()
    
    
    
#     cd tspsam
# python main.py --dataset DAVIS --test_path ..\input\davis2017\JPEGImages\480p --output_dir ..\output\tsp_sam\davis --resume model_checkpoint\best_checkpoint.pth --sequence hike --gpu_ids 0
# cd ..

# cd samurai
# python samurai/scripts/demo.py --video_path "input/davis2017/JPEGImages/480p/dog" --txt_path "input/davis2017/bboxes/bbox_dog.txt" --model_path "samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt"    


# python evaluation\davis_baseline_eval.py --method tsp-sam --sequences hike
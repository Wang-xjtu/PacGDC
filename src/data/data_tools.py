import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.v2 as trans
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from .data_utils import (
    rgb_read,
    depth_read,
    DataList,
    RandomResizedCropRGBD,
    RandomSimulator,
    depth_interpolation,
    depth_relocation,
)

RELATIVE_MODELS = ["DA", "DAV2", "DistllAD"]


class Transforms_train:
    def __init__(self, size, hole_ls):
        self._together_transform = trans.Compose(
            [
                trans.RandomCrop(size),
                RandomResizedCropRGBD(size, (0.49, 1.0), antialias=True),
                trans.RandomHorizontalFlip(0.5),
            ]
        )
        self._rgb_transform = trans.ColorJitter(0.4, 0.4, 0.4)
        self._raw_transform = RandomSimulator(size, hole_ls)

    def trans_rgbgt(self, rgb, gt):
        # gt must in the final channel
        data = torch.cat([rgb, gt], dim=0)
        data = self._together_transform(data)
        rgb = data[:3, :, :]
        rgb = self._rgb_transform(rgb)
        gt = data[3:4, :, :]
        raw = self._raw_transform.random_aug(gt, rgb)

        gt = torch.nan_to_num(gt)
        raw = torch.nan_to_num(raw)
        return rgb, gt, raw


class RGBDDataset(Dataset):
    def __init__(self, data_dir, size, foundation_models):
        super(RGBDDataset, self).__init__()
        foundation_models = foundation_models.split("_")
        self.data_dicts, hole_ls = DataList(data_dir, foundation_models).get_data()
        self.transforms = Transforms_train(size, hole_ls)

    def __len__(self):
        return len(self.data_dicts["rgb"])

    def __getitem__(self, item):
        rgb = rgb_read(self.data_dicts["rgb"][item])
        # start to synthesize pseduo depth

        depth_dict = {"relative": [], "absolute": [], "gt": []}
        for key in self.data_dicts.keys():
            if key not in ["rgb"]:
                data_path = self.data_dicts[key][item]
                if data_path is None:
                    # unlabeled data without gt
                    mask_gt = 1.0
                    continue
                else:
                    pred = depth_read(data_path)
                    if key in RELATIVE_MODELS:
                        # simply convert disp to pseudo depth
                        pred = torch.exp(-5 * pred)
                        depth_dict["relative"].append(pred)
                    elif key == "depth":
                        depth_dict["gt"].append(pred)
                        mask_gt = (pred > 0).float()
                    else:
                        depth_dict["absolute"].append(pred)

        depth_list = []
        if len(depth_dict["relative"]) > 0:
            relative_depth = depth_interpolation(depth_dict["relative"])
            depth_list.append(relative_depth)

        if len(depth_dict["absolute"]) > 0:
            absolute_depth = depth_interpolation(depth_dict["absolute"])
            depth_list.append(absolute_depth)

        if len(depth_dict["gt"]) > 0:
            # only have a single gt
            depth_list.append(depth_dict["gt"][0])

        gt = depth_interpolation(depth_list, mask_gt)
        gt = depth_relocation(gt)

        rgb, gt, raw = self.transforms.trans_rgbgt(rgb, gt)

        return rgb, gt, raw


def get_dataloader(
    data_dirs: Path,
    batch_size: int,
    foundation_models: str,
    sizes: int,
    rank: int,
    num_workers: int,
):
    dataset = RGBDDataset(data_dirs, sizes, foundation_models)
    if rank == 0:
        print(f"Loaded the dataset with: {len(dataset)} images...\n")

    # initialize dataloaders
    sampler = DistributedSampler(dataset)
    data = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return data, sampler

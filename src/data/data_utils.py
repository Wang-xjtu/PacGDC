import os
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torchvision.transforms.v2 as trans
import torchvision.transforms.v2.functional as TF
import cv2
import random


def rgb_read(filename: Path) -> Tensor:
    rgb = np.array(Image.open(filename), dtype="float32") / 255
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
    return rgb


def depth_read(filename: Path) -> Tensor:
    # make sure we have a proper 16bit depth map here.. not 8bit!
    depth = np.array(Image.open(filename), dtype="float32") / 65535
    depth = torch.from_numpy(depth).unsqueeze(0)
    return depth


def hole_read(filename: Path) -> Tensor:
    hole = np.array(Image.open(filename), dtype="float32") / 255
    hole = torch.from_numpy(hole).unsqueeze(0)
    return hole


def depth_interpolation(depth_list, mask_gt=1.0):
    if len(depth_list) == 0:
        return []
    elif len(depth_list) == 1:
        return depth_list[0] * mask_gt
    else:
        rest_pro = 1.0
        depth = torch.tensor(0.0)
        for i in range(len(depth_list) - 1):
            theta = np.random.uniform(0.0, rest_pro)
            depth = depth + theta * depth_list[i]
            rest_pro = rest_pro - theta
        return (depth + rest_pro * depth_list[-1]) * mask_gt


def depth_relocation(depth, factor=0.4):
    max_depth = torch.max(depth).item()
    if max_depth != 0.0:
        scale_factor = np.random.uniform(
            1 - factor, np.clip(1 + factor, a_min=1, a_max=1 / max_depth)
        )
        depth = depth * scale_factor
    return depth


class DataList:
    def __init__(
        self,
        data_dir: Path,
        foundation_models: list = ["DA", "DepthPro", "DAV2", "DistllAD"],
    ) -> None:
        self.data_dir = data_dir
        self.foundation_models = foundation_models

    def __getdata__(self):
        data_dicts = {"rgb": [], "depth": []}
        for model in self.foundation_models:
            data_dicts[model] = []

        # assuming labeled image is ".png" format (referring G2-MonoDepth's datasets)
        for file in (self.data_dir / "Data_Train" / "Labeled").rglob("*.png"):
            str_file = str(file)
            if "/rgb/" in str_file:
                for key in data_dicts.keys():
                    if key == "rgb":
                        data_dicts[key].append(file)
                    else:
                        model_file = str_file.replace("/rgb/", f"/{key}/", 1)
                        data_dicts[key].append(Path(model_file))

        # assuming unlabeled image is ".jpg" format (referring SA1B dataset)
        for file in (self.data_dir / "Data_Train" / "Unlabeled").rglob("*.jpg"):
            str_file = str(file)
            if "/rgb/" in str_file:
                for key in data_dicts.keys():
                    if key == "rgb":
                        data_dicts[key].append(file)
                    # depth is not available
                    elif key == "depth":
                        data_dicts[key].append(None)
                    else:
                        file_path = (
                            str_file.replace("/rgb/", f"/{key}/", 1)[:-4] + ".png"
                        )
                        data_dicts[key].append(Path(file_path))

        return data_dicts

    def __gethole__(self):
        hole_ls = []
        for file in (self.data_dir / "Data_Hole").rglob("*.png"):
            hole_ls.append(file)
        return hole_ls

    def get_data(self):
        data_dicts = self.__getdata__()
        hole_ls = self.__gethole__()
        return data_dicts, hole_ls


class RandomResizedCropRGBD(trans.RandomResizedCrop):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        # remove hole artifacts of depth maps in interpolation
        # depth must in the final channel
        dim = img.shape[0]
        hole = (img[dim - 1 : dim, :, :] > 0).float()
        img = torch.cat([img, hole], dim=0)
        img = TF.resized_crop(
            img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )
        hole = (img[dim, :, :] == 1).float()
        img = torch.cat(
            [img[: dim - 1, :, :], (img[dim - 1 : dim, :, :] * hole)], dim=0
        )
        return img


class RandomSimulator:
    def __init__(self, size, hole_ls):
        super().__init__()
        self.hole_ls = hole_ls
        self.hole_transform = trans.Compose(
            [
                trans.RandomCrop(size),
                trans.RandomAffine(
                    degrees=180,
                    translate=(0.5, 0.5),
                    scale=(0.5, 4.0),
                    shear=60,
                    fill=1.0,
                ),
            ]
        )

    def random_noise(self, raw):
        random_point = (torch.rand_like(raw) > np.random.uniform(0.0, 1.0)).float()
        hole = (raw > 0).float()
        gaussian_noise = torch.ones_like(raw).normal_(0, np.random.uniform(0.01, 0.1))
        raw = raw + gaussian_noise * hole * random_point

        return torch.clamp(raw, 0.0, 1.0)

    def random_blur(self, raw):
        raw_shape = raw.shape
        sample_factor = np.random.uniform(1.0, 16.0)
        raw = TF.resize(
            raw,
            (int(raw_shape[1] / sample_factor), int(raw_shape[2] / sample_factor)),
            TF.InterpolationMode.NEAREST,
        )
        raw = TF.resize(raw, (raw_shape[1], raw_shape[2]), TF.InterpolationMode.NEAREST)
        return raw

    def random_hole(self, raw):
        hole = hole_read(self.hole_ls[np.random.randint(0, len(self.hole_ls))])
        raw = raw * self.hole_transform(hole)
        return raw

    def random_aug(
        self,
        gt,
        rgb,
        p_hole=0.75,
        p_noise=0.25,
        p_blur=0.25,
        p_uniform=0.5,
        p_lidar=0.35,
    ):
        raw = gt.clone()
        r_hole, r_noise, r_blur, r_sample = np.random.uniform(0.0, 1.0, size=4)

        # holes
        if r_hole < p_hole:
            raw = self.random_hole(raw)
        # noise
        if r_noise < p_noise:
            raw = self.random_noise(raw)
        # blur
        if r_blur < p_blur:
            raw = self.random_blur(raw)

        # sparsity
        if r_sample <= p_uniform:
            raw = sample_uniform(raw)
        elif r_sample < (p_uniform + p_lidar):
            raw = sample_lidar(raw)
        else:
            raw = sample_sfm(raw, rgb)

        return raw


# LiDAR sampling from OMNI-DC implementation
def sample_lidar(dep):
    channel, height, width = dep.shape

    baseline_horizontal = np.random.choice([1.0, -1.0]) * np.random.uniform(0.03, 0.06)
    baseline_vertical = np.random.uniform(-0.02, 0.02)

    # the target view canvas need to be slightly bigger
    target_view_expand_factor = 1.5
    height_expanded = int(target_view_expand_factor * height)
    width_expanded = int(target_view_expand_factor * width)

    # sample a virtual intrinsics
    w_c = np.random.uniform(-0.5 * width, 1.5 * width)
    h_c = np.random.uniform(0.5 * height, 0.7 * height)
    focal = np.random.uniform(1.5 * height, 2.0 * height)
    Km = np.eye(3)
    Km[0, 0] = focal
    Km[1, 1] = focal
    Km[0, 2] = w_c
    Km[1, 2] = h_c

    Km_target = np.copy(Km)
    Km_target[0, 2] += (target_view_expand_factor - 1.0) / 2.0 * width
    Km_target[1, 2] += (target_view_expand_factor - 1.0) / 2.0 * height

    dep_np = dep.numpy()

    # unproject every depth to a virtual neighboring view
    _, v, u = np.nonzero(dep_np)
    z = dep_np[0, v, u]
    points3D_source = np.linalg.inv(Km) @ (
        np.vstack([u, v, np.ones_like(u)]) * z
    )  # 3 x N
    points3D_target = np.copy(points3D_source)
    points3D_target[0] -= baseline_horizontal  # move in the x direction
    points3D_target[1] -= baseline_vertical  # move in the y direction

    points2D_target = Km_target @ points3D_target
    depth_target = points2D_target[2]
    points2D_target = points2D_target[0:2] / (points2D_target[2:3] + 1e-8)  # 2 x N

    # 2 x N_valid
    points2D_target = np.round(points2D_target).astype(int)
    valid_mask = (
        (points2D_target[0] >= 0)
        & (points2D_target[0] < width_expanded)
        & (points2D_target[1] >= 0)
        & (points2D_target[1] < height_expanded)
    )

    points2D_target_valid = points2D_target[:, valid_mask]

    # N_valid
    depth_target_valid = depth_target[valid_mask]

    # take the min of all values
    dep_map_target = np.full((height_expanded, width_expanded), np.inf)
    np.minimum.at(
        dep_map_target,
        (points2D_target_valid[1], points2D_target_valid[0]),
        depth_target_valid,
    )
    dep_map_target[dep_map_target == np.inf] = 0.0

    # dep_map_target = fill_in_fast(dep_map_target, max_depth=np.max(dep_map_target))
    # dep_map_target = dep_map_target[None]  # 1 x H x W

    # sample the lidar patterns
    pitch_max = np.random.uniform(0.25, 0.30)
    pitch_min = np.random.uniform(-0.15, -0.20)
    num_lines = np.random.randint(8, 64)
    num_horizontal_points = np.random.randint(400, 1000)

    tgt_pitch = np.linspace(pitch_min, pitch_max, num_lines)
    tgt_yaw = np.linspace(-np.pi / 2.1, np.pi / 2.1, num_horizontal_points)

    pitch_grid, yaw_grid = np.meshgrid(tgt_pitch, tgt_yaw)
    y, x = np.sin(pitch_grid), np.cos(pitch_grid) * np.sin(
        yaw_grid
    )  # assume the distace is unit
    z = np.sqrt(1.0 - x**2 - y**2)
    points_3D = np.stack([x, y, z], axis=0).reshape(
        3, -1
    )  # 3 x (num_horizontal_points * num_lines)
    points_2D = Km @ points_3D
    points_2D = points_2D[0:2] / (
        points_2D[2:3] + 1e-8
    )  # 2 x (num_horizontal_points * num_lines)

    points_2D = np.round(points_2D).astype(int)
    points_2D_valid = points_2D[
        :,
        (
            (points_2D[0] >= 0)
            & (points_2D[0] < width_expanded)
            & (points_2D[1] >= 0)
            & (points_2D[1] < height_expanded)
        ),
    ]

    mask = np.zeros([channel, height_expanded, width_expanded])
    mask[:, points_2D_valid[1], points_2D_valid[0]] = 1.0

    dep_map_target_sampled = dep_map_target * mask

    # project it back to source
    _, v, u = np.nonzero(dep_map_target_sampled)
    if len(v) == 0:
        return sample_uniform(dep)

    z = dep_map_target_sampled[0, v, u]
    points3D_target = np.linalg.inv(Km_target) @ (
        np.vstack([u, v, np.ones_like(u)]) * z
    )  # 3 x N
    points3D_source = np.copy(points3D_target)
    points3D_source[0] += baseline_horizontal  # move in the x direction
    points3D_source[1] += baseline_vertical  # move in the y direction

    points2D_source = Km @ points3D_source
    depth_source = points2D_source[2]
    points2D_source = points2D_source[0:2] / (points2D_source[2:3] + 1e-8)  # 2 x N

    # 2 x N_valid
    points2D_source = np.round(points2D_source).astype(int)
    valid_mask = (
        (points2D_source[0] >= 0)
        & (points2D_source[0] < width)
        & (points2D_source[1] >= 0)
        & (points2D_source[1] < height)
    )
    points2D_source_valid = points2D_source[:, valid_mask]

    # N_valid
    depth_source_valid = depth_source[valid_mask]

    # take the min of all values
    dep_map_source = np.full((height, width), np.inf)
    np.minimum.at(
        dep_map_source,
        (points2D_source_valid[1], points2D_source_valid[0]),
        depth_source_valid,
    )
    dep_map_source[dep_map_source == np.inf] = 0.0

    # only keep the orginal valid regions
    dep_map_source = dep_map_source * ((dep_np > 0.0).astype(float))

    # only allow deeper value to appear in shallower region
    dep_map_source[dep_map_source < dep_np] = 0.0

    dep_sp = torch.tensor(dep_map_source).float()
    return dep_sp


# uniform sampling
def sample_uniform(dep) -> Tensor:
    sample_pattern = np.random.uniform(0, 1.0)

    def get_zero_rate(sample_pattern):
        if sample_pattern < 0.5:
            return np.random.uniform(0.4, 0.9)
        else:
            return np.random.uniform(0.9, 1.0)

    zero_rate = get_zero_rate(sample_pattern)
    random_point = (torch.rand_like(dep) > zero_rate).float()
    while torch.sum(random_point) < 2:
        zero_rate = get_zero_rate(sample_pattern)
        random_point = (torch.rand_like(dep) > zero_rate).float()

    return dep * random_point


# sfm sampling from OMNI-DC implementation
def sample_sfm(dep, rgb, pattern="sift") -> Tensor:
    channel, height, width = dep.shape
    rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

    if pattern == "sift":
        detector = cv2.SIFT.create()
    elif pattern == "orb":
        detector = cv2.ORB.create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
    else:
        raise NotImplementedError

    keypoints = detector.detect(gray)
    mask = torch.zeros([1, height, width])

    if len(keypoints) < 20:
        return sample_uniform(dep)

    for keypoint in keypoints:
        x = round(keypoint.pt[1])
        y = round(keypoint.pt[0])
        mask[:, x, y] = 1.0

    dep_sp = dep * mask.type_as(dep)
    return dep_sp

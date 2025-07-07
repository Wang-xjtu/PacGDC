import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpus

from src.body.train_body import PacGDCTrain
from src.configs import ConfigTrain
from src.utils import DDPutils

import torch
import argparse

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(
    "options for PacGDC",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model_type",
    type=str,
    default="L",
    help="Model Version: [T, S, B, L]",
)
parser.add_argument(
    "--foundation_models",
    type=str,
    default="DA_DepthPro",
    help="Foundation models splited with '_': [DA (DepthAnything), DepthPro, DAV2 (DepthAnythingV2), DistllAD (DistillAnyDepth)]",
)
args = parser.parse_args()
MODEL_TYPE = args.model_type
FOUNDATION_MODELS = args.foundation_models


def DDP_train_main(rank, world_size):
    cf = ConfigTrain(
        n_gpus=world_size,
        model_type=MODEL_TYPE,
        foundation_models=FOUNDATION_MODELS,
    )

    # DDP components
    DDPutils.setup(rank, world_size, 5998)
    if rank == 0:
        print(f"Selected arguments: {cf.__dict__}")
    trainer = PacGDCTrain(rank, cf)
    trainer.train(cf)
    DDPutils.cleanup()


if __name__ == "__main__":
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        DDPutils.run_demo(DDP_train_main, n_gpus)

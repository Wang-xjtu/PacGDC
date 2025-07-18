import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpus

from PIL import Image
from pathlib import Path
from src.SPNet.networks import CompletionNet
from test_utils import DataReader, metrics
import torch

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for PacGDC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=lambda x: Path(x),
        default="Datasets/Data_Test/Ibims",
        help="Path to test folder",
    )
    parser.add_argument(
        "--ckpt_path",
        type=lambda x: Path(x),
        default="Pretrained/L_DA_DepthPro.pth",
        help="Path to load models",
    )
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="save results",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=30.0,
        help="max_depth to normalize depth values [indoor: 30 m, outdoor: 150m]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    args = parser.parse_args()
    return args


@torch.no_grad()
def demo(args):
    data_reader = DataReader(args.device)
    print("-----------building model-------------")
    # hugging face loading
    # net = CompletionNet.from_pretrained("Haotian-sx/PacGDC_large").to(args.device).eval()
    # locally loading
    net = CompletionNet(str(args.ckpt_path.name)[0]).to(args.device).eval()
    net.load_state_dict(torch.load(args.ckpt_path)["network"])
    raw_dirs = ["10%", "1%", "0.1%"]
    avg_rmse, avg_mae = 0.0, 0.0
    print("-----------inferring---------------")
    for raw_dir in raw_dirs:
        # init
        count, rmse, mae = 0.0, 0.0, 0.0

        for file in (args.data_dir / "rgb").rglob("*.png"):
            rgb_path = str(file)
            raw_path = rgb_path.replace("/rgb/", "/raw_" + raw_dir + "/")
            gt_path = rgb_path.replace("/rgb/", "/gt/")
            data_reader = DataReader("cuda")

            rgb, raw, gt, hole_raw = data_reader.read_data(rgb_path, raw_path, gt_path)

            # forward
            pred = net(rgb, raw, hole_raw)

            # denormalize
            pred = pred * args.max_depth
            gt = gt * args.max_depth

            # metrics
            count += 1.0
            rmse_temp, mae_temp = metrics(pred, gt)
            rmse += rmse_temp
            mae += mae_temp

            # save img
            if args.save_results:
                save_path = rgb_path.replace("/rgb/", "/result_" + raw_dir + "/")
                pred = data_reader.toint32(pred)
                os.makedirs(str(Path(save_path).parent), exist_ok=True)
                Image.fromarray(pred).save(save_path)
                print(save_path)
        # metric each raw_dir
        rmse /= count
        mae /= count
        avg_rmse += rmse
        avg_mae += mae
    # average metric
    avg_rmse /= len(raw_dirs)
    avg_mae /= len(raw_dirs)
    print("Average: RMSE=", str(avg_rmse), " MAE=", str(avg_mae))


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)

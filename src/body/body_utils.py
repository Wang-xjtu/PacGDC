import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from ..utils import print_model_parm_nums
from ..data.data_tools import get_dataloader
from ..SPNet.networks import CompletionNet
from ..losses.losses import G2Loss


def cosinedecay_warmup(optimizer, iteration_num, warmup_epochs, epochs):
    # learning rate scheduler
    warmup_iters = iteration_num * warmup_epochs
    total_iters = iteration_num * epochs
    lr_lambda = lambda x: (
        x / warmup_iters
        if x < warmup_iters
        else 0.5
        * (1 + math.cos(math.pi * (x - warmup_iters) / (total_iters - warmup_iters)))
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler


def lineardecay_warmup(optimizer, iteration_num, warmup_epochs, epochs):
    # learning rate scheduler
    warmup_iters = iteration_num * warmup_epochs
    total_iters = iteration_num * epochs
    lr_lambda = lambda x: (
        x / warmup_iters
        if x < warmup_iters
        else 1 - (x - warmup_iters) / (total_iters - warmup_iters)
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler


def Polynomialdecay_warmup(optimizer, iteration_num, warmup_epochs, epochs, power=1.0):
    # learning rate scheduler
    warmup_iters = iteration_num * warmup_epochs
    total_iters = iteration_num * epochs
    lr_lambda = lambda x: (
        x / warmup_iters
        if x < warmup_iters
        else (1 - (x - warmup_iters) / (total_iters - warmup_iters)) ** power
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler


def save_everything(epoch, network, optimizer, scheduler, scaler, save_file):
    torch.save(
        {
            "epoch": epoch,
            "network": network.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        save_file,
    )


def load_everything(checkpoint, network, optimizer, scheduler, scaler, rank):
    if rank == 0:
        print("resume training using weights in " + str(checkpoint))
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    checkpoint = torch.load(checkpoint, map_location=map_location)
    network.module.load_state_dict(checkpoint["network"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint["epoch"]

    del checkpoint

    return start_epoch


def feedback_module(
    elapsed,
    iter_step,
    iteration_num,
    loss,
    summary,
    global_step,
):
    print(
        "Elapsed:[%s]|batch:[%d/%d]|loss:%.4f"
        % (elapsed, iter_step, iteration_num, float(loss))
    )
    # log loss
    summary.add_scalar("loss", loss, global_step=global_step)


def min_max_norm(depth):
    max_value = torch.max(depth)
    min_value = torch.min(depth)
    norm_depth = (depth - min_value) / (max_value - min_value)
    return norm_depth


def grid_depth(gt, raw, pred):
    list_depth_org = [gt[0], raw[0], pred[0]]
    list_depth_norm = []
    for i in range(len(list_depth_org)):
        temp = min_max_norm(list_depth_org[i])
        list_depth_norm.append(temp)
    vis_depth = make_grid(list_depth_org + list_depth_norm, nrow=len(list_depth_org))
    return vis_depth


class BaseBody:
    def __init__(self, rank, cf):
        comp_net = CompletionNet(cf.model_type)
        self.comp_net = comp_net.cuda().train()
        self.rank = rank
        self.loader, self.sampler = get_dataloader(
            cf.data_dirs,
            cf.batch_size,
            cf.foundation_models,
            cf.sizes,
            rank,
            cf.num_workers,
        )
        if rank == 0:
            print_model_parm_nums(self.comp_net)
        # Use DistributedDataParallel
        self.comp_net = DDP(
            self.comp_net,
            device_ids=[self.rank],
        )
        self.comp_net = torch.compile(self.comp_net)
        self.iteration_num = len(self.loader)  # total iteration number

        # create optimizers
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.comp_net.parameters(),
                    "initial_lr": cf.lr,
                },
            ],
            weight_decay=cf.wd,
        )

        # learning rate scheduler
        self.scheduler = cosinedecay_warmup(
            self.optimizer, self.iteration_num, cf.warmup_epochs, cf.epochs
        )

        # use amp
        self.amp = cf.amp
        if self.amp:
            self.scaler = torch.amp.GradScaler("cuda")

        self.start_epoch = 0
        # resume train
        if cf.checkpoint is not None:
            self.start_epoch = load_everything(
                cf.checkpoint,
                self.comp_net,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.rank,
            )

        self.loss_fun = G2Loss()

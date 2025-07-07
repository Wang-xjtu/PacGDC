import datetime
import time
import timeit
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from .body_utils import (
    save_everything,
    grid_depth,
    feedback_module,
    BaseBody,
)


class PacGDCTrain(BaseBody):
    def __init__(self, rank, cf):
        super().__init__(rank, cf)

    def optimize_one_iteration(self, rgb, gt, raw):
        mask_gt = (gt > 0).float()
        mask_raw = (raw > 0).float()
        # AMP type in pytorch 2.7
        with torch.autocast(device_type="cuda", enabled=self.amp):
            pred = self.comp_net(rgb, raw, mask_raw)
            loss = self.loss_fun(pred, gt, mask_raw, mask_gt)
        self.optimizer.zero_grad()
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss, pred

    @staticmethod
    def img_saving(rgb, gt, raw, pred, log_dir, epoch, iter_step):
        # make dir
        epoch_dir = log_dir / ("epoch_" + str(epoch))
        epoch_dir.mkdir(exist_ok=True)
        file_last = f"_{epoch}_{iter_step}.png"
        # save the images:
        vis_depth = grid_depth(gt, raw, pred)
        save_image(vis_depth, epoch_dir / ("depth" + file_last))
        save_image(rgb[0], epoch_dir / ("rgb" + file_last))

    def train(self, cf):
        if self.rank == 0:
            # model/log data_dirs
            model_dir, log_dir = cf.save_dir / "models", cf.save_dir / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            # tensorboard summarywriter
            summary = SummaryWriter(str(log_dir / "tensorboard"))
            # create a global time counter
            global_time = time.time()
            print("Starting the training process ... ")
        # epoch start
        global_step = self.start_epoch * self.iteration_num
        for epoch in range(self.start_epoch + 1, cf.epochs + 1):
            # set epoch in Distributed samplers
            self.sampler.set_epoch(epoch)
            # record time at the start of epoch
            if self.rank == 0:
                start = timeit.default_timer()
                print(f"\nEpoch: [{epoch}/{cf.epochs}]")

            for i, (rgb, gt, raw) in enumerate(self.loader, start=1):
                # get data
                rgb = rgb.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                raw = raw.cuda(non_blocking=True)
                # optimizing
                loss, pred = self.optimize_one_iteration(rgb, gt, raw)
                self.scheduler.step()  # learning rate decay
                # logging
                global_step += 1
                # log a loss feedback
                if self.rank == 0 and ((i % cf.feedback_iteration == 0) or (i == 1)):
                    with torch.no_grad():
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        # log and print
                        feedback_module(
                            elapsed,
                            i,
                            self.iteration_num,
                            loss,
                            summary,
                            global_step,
                        )
                        # save intermediate results
                        self.img_saving(rgb, gt, raw, pred, log_dir, epoch, i)
            # logging checkpoint
            if self.rank == 0:
                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))
                if epoch % cf.checkpoint_epoch == 0 or epoch == cf.epochs:
                    save_file = model_dir / f"epoch_{epoch}.pth"
                    save_everything(
                        epoch,
                        self.comp_net,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        save_file,
                    )

        if self.rank == 0:
            print("Training completed ...")

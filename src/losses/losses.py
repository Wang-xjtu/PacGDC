import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import conv2d, interpolate


class Gradient2D(Module):
    def __init__(self):
        super(Gradient2D, self).__init__()
        kernel_x = torch.FloatTensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        kernel_y = torch.FloatTensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        )
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        self.weight_x = Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x: Tensor):
        grad_x = conv2d(x, self.weight_x)
        grad_y = conv2d(x, self.weight_y)
        return grad_x, grad_y


class MaskedMSGradLoss(Module):
    def __init__(self):
        super(MaskedMSGradLoss, self).__init__()
        self.grad_fun = Gradient2D().cuda()

    def __gradient_loss__(self, residual) -> Tensor:
        loss_x, loss_y = self.grad_fun(residual)
        loss_x = torch.sum(torch.abs(loss_x))
        loss_y = torch.sum(torch.abs(loss_y))
        return loss_x + loss_y

    def forward(self, residual, number_valid, k=4) -> Tensor:
        loss = 0.0
        for i in range(k):
            if i == 0:
                k_residual = residual
            else:
                k_residual = interpolate(
                    k_residual, scale_factor=1 / 2, recompute_scale_factor=True
                )
            loss += self.__gradient_loss__(k_residual)
        return loss / number_valid


class MaskedDataLoss(Module):
    def __init__(
        self,
    ) -> None:
        super(MaskedDataLoss, self).__init__()

    def forward(self, residual, number_valid) -> Tensor:
        loss = torch.sum(torch.abs(residual))
        return loss / number_valid


class G2LossTool(Module):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-6
        self.dataloss = MaskedDataLoss()
        self.msgradloss = MaskedMSGradLoss()

    def maskedmeanstd(self, depth, mask):
        mask_num = torch.sum(mask, dim=(2, 3), keepdim=True)
        mask_num[mask_num == 0] = self.eps
        depth_mean = torch.sum(depth * mask, dim=(2, 3), keepdim=True) / mask_num
        depth_std = (
            torch.sum(torch.abs((depth - depth_mean) * mask), dim=(2, 3), keepdim=True)
            / mask_num
        )
        return depth_mean, depth_std + self.eps

    def standardize(self, depth, mask):
        t_d, s_d = self.maskedmeanstd(depth, mask)
        sta_depth = (depth - t_d) / s_d
        return sta_depth.split(1, dim=1)  # depth channel is 1

    def relative_loss(self, sta_pred, sta_gt, mask_gt):
        # loss in relative domain
        residual = (sta_pred - sta_gt) * mask_gt
        number_valid = torch.sum(mask_gt) + self.eps

        loss_ai = self.dataloss(residual, number_valid)
        loss_msg = self.msgradloss(residual, number_valid)

        return loss_ai + 0.5 * loss_msg

    def absolute_loss(self, pred, gt, mask_raw):
        # loss in absolute domain
        residual = (pred - gt) * mask_raw
        number_valid = torch.sum(mask_raw) + self.eps

        loss_abs = self.dataloss(residual, number_valid)
        return loss_abs

    def g2_loss(self, pred, gt, sta_pred, sta_gt, mask_raw, mask_gt):
        mask_raw = mask_raw * mask_gt  # holes in mask_raw must include holes in mask_gt
        # loss in absolute domain
        # loss_abs = self.absolute_loss(pred, gt, mask_raw)
        loss_abs = self.absolute_loss(
            pred, gt, mask_gt
        )  # change to mask_gt (valid pixels > 2)
        # loss in relative domain
        loss_rel = self.relative_loss(sta_pred, sta_gt, mask_gt)
        return loss_abs + loss_rel


class G2Loss(G2LossTool):
    def __init__(self):
        super(G2Loss, self).__init__()

    def forward(self, pred, gt, mask_raw, mask_gt):
        sta_pred, sta_gt = self.standardize(
            torch.cat([pred, gt], dim=1),
            torch.cat([mask_gt, mask_gt], dim=1),
        )
        loss = self.g2_loss(pred, gt, sta_pred, sta_gt, mask_raw, mask_gt)
        return loss

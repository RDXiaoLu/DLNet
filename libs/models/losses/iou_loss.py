import torch
from mmdet.models.builder import LOSSES


@LOSSES.register_module
class CLRNetIoULoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, lane_width=15 / 800):
        """
        LineIoU loss employed in CLRNet.
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/losses/lineiou_loss.py
        Args:
            weight (float): loss weight.
            lane_width (float): virtual lane half-width.
        """
        super(CLRNetIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.lane_width = lane_width
        

    def calc_iou(self, pred, target, pred_width, target_width):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nl, Nr), relative coordinate
            target: ground truth, shape: (Nl, Nr), relative coordinate
            pred_width (torch.Tensor): virtual lane half-widths
                for prediction at pre-defined rows, shape (Nl, Nr).
            target_width (torch.Tensor): virtual lane half-widths
                for GT at pre-defined rows, shape (Nl, Nr).
        Returns:
            torch.Tensor: calculated IoU, shape (N).
        Nl: number of lanes, Nr: number of rows.
        """
        px1 = pred - pred_width
        px2 = pred + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)

        invalid_masks = (invalid_mask < 0) | (invalid_mask >= 1.0)
        ovr[invalid_masks] = 0.0
        union[invalid_masks] = 0.0
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou

    def forward(self, pred, target):
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"
        width = torch.ones_like(target) * self.lane_width
        iou = self.calc_iou(pred, target, width, width)
        return (1 - iou).mean() * self.loss_weight


@LOSSES.register_module
class DLIoULoss(CLRNetIoULoss):
    def __init__(self, loss_weight=1.0, lane_width=7.5 / 800):
        """
        LaneIoU loss employed in CLRerNet.
        Args:
            weight (float): loss weight.
            lane_width (float): half virtual lane width.
        """
        super(DLIoULoss, self).__init__(loss_weight, lane_width)
        self.max_dx = 1e4

    def _calc_lane_width(self, pred, target, eval_shape):
        """
        Calculate the virtual LaneIoU widths for predictions
        and targets based on the tilt of pred lanes.
        Args:
            pred: lane predictions, shape: (Nl, Nr), relative coordinate.
            target: ground truth, shape: (Nl, Nr), relative coordinate.
            eval_shape (tuple): Cropped image shape corresponding
                to the area in evaluation.
        Returns:
            torch.Tensor: virtual lane half-widths for prediction
                at pre-defined rows, shape (Nl, Nr).
            torch.Tensor: virtual lane half-widths for GT
                at pre-defined rows, shape (Nl, Nr).
        Nl: number of lanes, Nr: number of rows.
        """
        img_h, img_w = eval_shape
        n_strips = pred.shape[1] - 1
        dy = img_h / n_strips * 2  # two horizontal grids
        _pred = pred.clone().detach()
        pred_dx = (
            _pred[:, 2:] - _pred[:, :-2]
        ) * img_w  # pred x difference across two horizontal grids
        pred_width = (
            self.lane_width * torch.sqrt(pred_dx.pow(2) + dy**2) / dy
        )
        pred_width = torch.cat(
            [pred_width[:, 0:1], pred_width, pred_width[:, -1:]], dim=1
        )
        target_dx = (target[:, 2:] - target[:, :-2]) * img_w
        target_dx[torch.abs(target_dx) > self.max_dx] = 0
        target_width = (
            self.lane_width * torch.sqrt(target_dx.pow(2) + dy**2) / dy
        )
        target_width = torch.cat(
            [target_width[:, 0:1], target_width, target_width[:, -1:]], dim=1
        )

        return pred_width, target_width
    
    def _calc_directional_iou(self, pred, target, img_w, length=15, aligned=True):
        """
        Calculate DLIoU and DRIoU based on the predicted and target lane positions.
        Args:
            pred: lane predictions, shape: (Nl, Nr), relative coordinate
            target: ground truth, shape: (Nl, Nr), relative coordinate
            img_w: image width
            length: extended radius for calculating directional IoU
        Returns:
            DLIoU, DRIoU
        """
        # Calculate the left and right offsets for the predicted and target lanes
        px1 = pred - length
        px2 = pred + length
        tx1 = target - length
        tx2 = target + length
        if aligned:
            invalid_mask = target
            ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
            union = torch.max(px2, tx2) - torch.min(px1, tx1)
            left_ovr = target - torch.max(px1, tx1)
            left_union = length
            right_ovr = torch.min(px2, tx2) - target 
            right_union = length
        else:
            num_pred = pred.shape[0]
            invalid_mask = target.repeat(num_pred, 1, 1)
            ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
                torch.max(px1[:, None, :], tx1[None, ...]))
            union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                    torch.min(px1[:, None, :], tx1[None, ...]))
            left_ovr = target - torch.max(px1[:, None, :], tx1[None, ...])
            left_union = length
            right_ovr = torch.min(px2[:, None, :], tx2[None, ...]) - target 
            right_union = length

        invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
        ovr[invalid_masks] = 0.
        union[invalid_masks] = 0.
        left_ovr[invalid_masks] = 0.
        right_ovr[invalid_masks] = 0.
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)

        all_shape = 72 
        DLIoU = left_ovr.sum(dim=-1) / (left_union*all_shape + 1e-9)
        DRIoU = right_ovr.sum(dim=-1) / (right_union*all_shape + 1e-9)
        
        return DLIoU, DRIoU
    
    def forward(self, pred, target, eval_shape=(320, 1640)):
        """
        Calculate the LaneIoU loss based on the predictions and targets.
        Args:
            pred: lane predictions, shape: (Nl, Nr), relative coordinate.
            target: ground truth, shape: (Nl, Nr), relative coordinate.
            eval_shape(tuple): Cropped image shape corresponding to
                the area in evaluation.
        Returns:
            torch.Tensor: LaneIoU loss value.
        Nl: number of lanes, Nr: number of rows.
        """
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"
        pred_width, target_width = self._calc_lane_width(
            pred, target, eval_shape
        )
        iou = self.calc_iou(pred, target, pred_width, target_width)
        DLIoU, DRIoU = self._calc_directional_iou(pred, target,  eval_shape[1])
        
        # Combine the losses: Laneiou, DLIoU, DRIoU
       
        return (1 - iou).mean() * self.loss_weight, (1 - DLIoU).mean() , (1 - DRIoU).mean()

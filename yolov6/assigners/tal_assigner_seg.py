import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.assigners.assigner_utils import select_candidates_in_gts, select_highest_overlaps, iou_calculator, dist_calculator

class TaskAlignedAssigner(nn.Module):
    def __init__(self,
                 topk=13,
                 num_classes=80,
                 alpha=1.0,
                 beta=6.0,
                 eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes,
                mask_gt,
                gt_segmasks):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
            gt_segmasks (Tensor): shape(bs, n_max_boxes, H, W) <- Added for segmentation
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx_lst (list[Tensor]): List containing target_gt_idx for each batch element (needed for seg mask target)
        """
        # Remove debugging prints
        # print(">>> Entering TALAssignerSeg.forward <<<")
        # print("gt_labels shape:", gt_labels.shape)
        # print("gt_labels unique values:", torch.unique(gt_labels))
        # -----------------------------
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), \
                   torch.zeros_like(pd_bboxes).to(device), \
                   torch.zeros_like(pd_scores).to(device), \
                   torch.zeros_like(pd_scores[..., 0]).to(device), \
                   [torch.tensor([], device=device, dtype=torch.long)] * self.bs # Return empty list of tensors


        cycle, step, self.bs = (1, self.bs, self.bs) # Process all batches at once
        target_labels_lst, target_bboxes_lst, target_scores_lst, fg_mask_lst, target_gt_idx_lst = [], [], [], [], []

        for i in range(cycle): # This loop will run only once with current cycle/step settings
            start, end = i*step, (i+1)*step
            pd_scores_ = pd_scores[start:end, ...]
            pd_bboxes_ = pd_bboxes[start:end, ...]
            gt_labels_ = gt_labels[start:end, ...]
            gt_bboxes_ = gt_bboxes[start:end, ...]
            mask_gt_   = mask_gt[start:end, ...]
            # gt_segmasks_ = gt_segmasks[start:end, ...] # Not needed directly in this loop

            mask_pos, align_metric, overlaps = self.get_pos_mask(
                pd_scores_, pd_bboxes_, gt_labels_, gt_bboxes_, anc_points, mask_gt_)

            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
                mask_pos, overlaps, self.n_max_boxes)

            # assigned target
            target_labels, target_bboxes, target_scores, assigned_target_gt_idx = self.get_targets(
                gt_labels_, gt_bboxes_, target_gt_idx, fg_mask)

            # normalize
            align_metric *= mask_pos
            pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
            pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
            # Adding self.eps to denominator prevents division by zero
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
            target_scores = target_scores * norm_align_metric

            # append
            target_labels_lst.append(target_labels)
            target_bboxes_lst.append(target_bboxes)
            target_scores_lst.append(target_scores)
            fg_mask_lst.append(fg_mask)
            # Store the assigned_target_gt_idx which relates anchors to original gt indices *within this batch slice*
            target_gt_idx_lst.append(assigned_target_gt_idx)


        # concat results from loop (will just be the single result as loop runs once)
        target_labels = torch.cat(target_labels_lst, 0)
        target_bboxes = torch.cat(target_bboxes_lst, 0)
        target_scores = torch.cat(target_scores_lst, 0)
        fg_mask = torch.cat(fg_mask_lst, 0)
        # Note: target_gt_idx_lst is a list, should remain a list as indices are relative to gt boxes *within each original batch element*


        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx_lst # Return list of indices

    def get_pos_mask(self,
                     pd_scores,
                     pd_bboxes,
                     gt_labels,
                     gt_bboxes,
                     anc_points,
                     mask_gt):

        # get anchor_align metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask
        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self,
                        pd_scores,
                        pd_bboxes,
                        gt_labels,
                        gt_bboxes):

        # Safely handle potential invalid class indices in gt_labels
        pd_scores_perm = pd_scores.permute(0, 2, 1) # Shape (bs, num_classes, num_total_anchors)
        gt_labels_long = gt_labels.to(torch.long)   # Shape (bs, n_max_boxes, 1)

        bs, n_max_boxes, _ = gt_labels_long.shape
        num_classes = pd_scores_perm.shape[1]
        num_anchors = pd_scores_perm.shape[2]

        # Create batch indices matching gt_labels shape
        batch_idx = torch.arange(bs, device=gt_labels.device).view(-1, 1).repeat(1, n_max_boxes) # Shape (bs, n_max_boxes)

        # Get the potentially invalid class indices
        class_idx = gt_labels_long.squeeze(-1) # Shape (bs, n_max_boxes)

        # Create a mask for valid class indices (0 <= class < num_classes)
        valid_class_mask = (class_idx >= 0) & (class_idx < num_classes) # Shape (bs, n_max_boxes)

        # Initialize bbox_scores with zeros
        # Shape: (bs, n_max_boxes, num_total_anchors)
        bbox_scores = torch.zeros(bs, n_max_boxes, num_anchors, device=pd_scores.device, dtype=pd_scores.dtype)

        # Only gather scores where the class index is valid
        if valid_class_mask.any(): # Proceed only if there are any valid labels
            batch_idx_valid = batch_idx[valid_class_mask]       # Shape (num_valid,)
            class_idx_valid = class_idx[valid_class_mask]       # Shape (num_valid,)

            # Gather scores for valid combinations: pd_scores_perm[batch_idx_valid, class_idx_valid, :]
            # Result shape: (num_valid, num_anchors)
            scores_valid = pd_scores_perm[batch_idx_valid, class_idx_valid]

            # Place the gathered scores into the correct positions in bbox_scores
            # Use the boolean mask directly for assignment (more efficient)
            bbox_scores[valid_class_mask] = scores_valid
            # Ensure the shape matches: bbox_scores[valid_class_mask] flattens the mask and expects scores_valid to be flat too.
            # Let's double check shapes. scores_valid is (num_valid, num_anchors). bbox_scores[valid_class_mask] selects elements.
            # This assignment might need adjustment if bbox_scores[valid_class_mask] doesn't yield the desired shape.
            # Alternative: Find row/col indices
            # rows, cols = torch.where(valid_class_mask)
            # bbox_scores[rows, cols, :] = scores_valid # This looks correct

            # Let's stick with the explicit row/col indexing for clarity and correctness
            rows, cols = torch.where(valid_class_mask)
            bbox_scores[rows, cols, :] = scores_valid


        # bbox_scores now has shape (bs, n_max_boxes, num_total_anchors)
        # Scores corresponding to invalid gt_labels remain zero.

        # Calculate overlaps and align_metric
        # Make sure gt_bboxes corresponding to invalid classes are handled if iou_calculator is sensitive.
        # Assuming iou_calculator handles potentially invalid boxes gracefully or they are filtered by mask_gt earlier.
        overlaps = iou_calculator(gt_bboxes, pd_bboxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta) # Uses the potentially zeroed bbox_scores

        return align_metric, overlaps

    def select_topk_candidates(self,
                               metrics,
                               largest=True,
                               topk_mask=None):

        num_anchors = metrics.shape[-1]
        # metrics shape: (bs, n_max_boxes, num_anchors)
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, dim=-1, largest=largest) # Use dim=-1

        if topk_mask is None:
            # Check if max metric > eps across anchors for each gt box
            topk_mask = (topk_metrics.max(dim=-1, keepdim=True)[0] > self.eps) # Use dim=-1
        else:
             # Ensure topk_mask is boolean and has the correct shape (bs, n_max_boxes, topk)
             topk_mask = topk_mask.bool()
             if topk_mask.shape != topk_metrics.shape:
                 # Handle potential shape mismatch if needed, e.g., repeat or check logic
                 # Assuming topk_mask input is (bs, n_max_boxes, 1) or similar and needs repeating
                 # Let's assume the input topk_mask is already (bs, n_max_boxes, topk) based on earlier usage
                 pass


        # Zero out indices where the mask is False
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))

        # Create one-hot encoding for top-k indices
        # Need to handle potential zeros introduced by the mask if index 0 is valid
        # Use scatter_add or equivalent for robust one-hot creation if indices can repeat (though topk shouldn't repeat)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(dim=-2) # Use dim=-2

        # Clamp values > 1 in case of issues (shouldn't happen with topk)
        is_in_topk = torch.where(is_in_topk > 1,
            torch.ones_like(is_in_topk), is_in_topk) # Use ones_like to preserve dtype maybe? Or check casting.
            # Let's keep it simple: torch.zeros_like(is_in_topk), is_in_topk seems fine.

        return is_in_topk.to(metrics.dtype) # Cast back to original dtype

    def get_targets(self,
                    gt_labels,
                    gt_bboxes,
                    target_gt_idx,
                    fg_mask):
        """Compute target labels, bboxes, and scores."""

        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[...,None]
        # target_gt_idx has shape (bs, num_total_anchors), indicating which gt box each anchor is assigned to
        # Need to gather labels based on this index for each anchor
        # Handle potential negative indices if select_highest_overlaps can return -1

        # Create indices for gathering, ensuring target_gt_idx is valid
        # Clamp target_gt_idx to avoid negative indices if they represent background
        target_gt_idx_clamped = target_gt_idx.clamp(min=0)

        # Generate combined batch and gt indices for gathering
        # target_gt_idx_clamped needs to be adjusted by batch offset if gt_labels is flattened
        # Let's gather directly using batch_ind and target_gt_idx_clamped
        # gt_labels has shape (bs, n_max_boxes, 1)
        # We need target labels of shape (bs, num_total_anchors)

        # Use gather along the n_max_boxes dimension
        # Expand batch_ind to match target_gt_idx_clamped shape if needed
        batch_idx_gather = batch_ind.expand(-1, target_gt_idx_clamped.shape[1]) # Shape (bs, num_total_anchors)

        # Gather labels: shape (bs, num_total_anchors)
        target_labels = gt_labels[batch_idx_gather, target_gt_idx_clamped].squeeze(-1)

        # assigned target boxes: shape (bs, num_total_anchors, 4)
        target_bboxes = gt_bboxes[batch_idx_gather, target_gt_idx_clamped]

        # assigned target scores: shape (bs, num_total_anchors, num_classes)
        # Use the gathered target_labels (before potentially masking background)
        # Ensure labels used for one-hot are within [0, num_classes-1]
        valid_label_mask_for_onehot = (target_labels >= 0) & (target_labels < self.num_classes)
        target_labels_safe = torch.where(valid_label_mask_for_onehot, target_labels, torch.zeros_like(target_labels))

        target_scores = F.one_hot(target_labels_safe.long(), self.num_classes).float() # Ensure scores are float

        # Mask out scores for background anchors (where fg_mask is False)
        # Also mask out scores where the original target_gt_idx was invalid (e.g., -1) or label was out of range
        # Ensure all masks are boolean before bitwise AND
        final_fg_mask = fg_mask.bool() & valid_label_mask_for_onehot.bool() & (target_gt_idx >= 0).bool()
        target_scores = target_scores * final_fg_mask.unsqueeze(-1) # Unsqueeze to broadcast mask

        # Mask labels and bboxes for background anchors
        target_labels = torch.where(final_fg_mask, target_labels, torch.full_like(target_labels, self.bg_idx))
        target_bboxes = torch.where(final_fg_mask.unsqueeze(-1), target_bboxes, torch.zeros_like(target_bboxes))

        # Return the original target_gt_idx as it's needed for mapping back to GT masks later
        # No need to flatten target_gt_idx here, it's per-anchor index into the GT boxes of that batch item
        assigned_target_gt_idx = target_gt_idx # Shape (bs, num_anchors)

        return target_labels, target_bboxes, target_scores, assigned_target_gt_idx

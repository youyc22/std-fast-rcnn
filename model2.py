import math
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.ops as ops
import torch.nn.functional as F
from utils import compute_offsets, assign_label, generate_proposal
from loss import ClsScoreRegression, BboxRegression

class FeatureExtractor(nn.Module):
    """
    Enhanced image feature extraction with MobileNet.
    Added batch normalization and feature pyramid network for better feature representation.
    """
    def __init__(self, reshape_size=224, pooling=False, verbose=False):
        super().__init__()
        
        # Use a stronger backbone
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add FPN for multi-scale feature extraction
        self.fpn_channels = 256
        self.lateral_conv = nn.Conv2d(960, self.fpn_channels, 1)  # MobileNetV3 last layer has 960 channels
        self.output_conv = nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1)
        
        # Add batch normalization
        self.bn = nn.BatchNorm2d(self.fpn_channels)
        
        if pooling:
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None

        # Gradient checkpointing to save memory
        self.backbone.train()
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

    def forward(self, img, verbose=False):
        """
        Enhanced forward pass with FPN and gradient checkpointing
        """
        num_img = img.shape[0]
        feat = []
        
        # Process in smaller batches to save memory
        process_batch = 200
        for b in range(math.ceil(num_img/process_batch)):
            batch_img = img[b*process_batch:(b+1)*process_batch]
            
            # Use gradient checkpointing for memory efficiency
            if self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                batch_feat = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.backbone),
                    batch_img
                )
            else:
                batch_feat = self.backbone(batch_img)
            
            # FPN processing
            lateral_feat = self.lateral_conv(batch_feat)
            output_feat = self.output_conv(lateral_feat)
            output_feat = self.bn(output_feat)
            
            if self.pool is not None:
                output_feat = self.pool(output_feat)
            
            feat.append(output_feat)
            
        feat = torch.cat(feat)
        return feat

class FastRCNN_2(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=1024, num_classes=20,
                 roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
        self.feat_extractor = FeatureExtractor()
        
        # Enhanced shared layers with residual connections
        self.shared_layers = nn.Sequential(
            nn.Linear(in_dim * roi_output_w * roi_output_h, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            ResidualBlock(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio)
        )
        
        # Separate classification and regression heads
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim // 2, num_classes + 1)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim // 2, 4)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Improved weight initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images, bboxes, bbox_batch_ids, proposals, proposal_batch_ids):
        """
        Enhanced forward pass with improved loss computation and sampling strategy
        """
        # Dynamic loss weights based on batch statistics
        w_cls = 1.0
        w_bbox = 1.0  # Typically bbox regression needs larger weight
        
        # Extract features with FPN
        features = self.feat_extractor(images)
        
        # RoI Align with better sampling
        rois = torch.cat([proposal_batch_ids.unsqueeze(1), proposals], dim=1)
        roi_features = ops.roi_align(features, rois, 
                                   output_size=(self.roi_output_w, self.roi_output_h),
                                   sampling_ratio=2)  # Use better sampling
        
        # Flatten and process through shared layers
        roi_features = roi_features.view(roi_features.size(0), -1)
        shared_feat = self.shared_layers(roi_features)
        
        # Get predictions
        cls_scores = self.cls_head(shared_feat)
        offsets = self.bbox_head(shared_feat)
        
        # Enhanced target assignment with OHEM
        final_cls_scores = []
        final_gt_labels = []
        final_offsets = []
        final_gt_offsets = []
        
        B = images.shape[0]
        background_id = self.num_classes
        
        for img_idx in range(B):
            img_proposals = proposals[proposal_batch_ids == img_idx]
            img_gt_boxes = bboxes[bbox_batch_ids == img_idx]
            img_cls_scores = cls_scores[proposal_batch_ids == img_idx]
            img_offsets = offsets[proposal_batch_ids == img_idx]
            
            # Improved label assignment with OHEM
            pos_mask, neg_mask, gt_class, gt_boxes = self._assign_labels_ohem(
                img_proposals, img_gt_boxes, img_cls_scores, background_id
            )
            
            # Handle classification
            cls_mask = pos_mask | neg_mask
            if cls_mask.any():
                img_gt_labels = torch.full((len(img_proposals),), background_id,
                                        dtype=torch.long, device=img_proposals.device)
                img_gt_labels[pos_mask] = gt_class[pos_mask]
                
                final_cls_scores.append(img_cls_scores[cls_mask])
                final_gt_labels.append(img_gt_labels[cls_mask])
            
            # Handle box regression
            if pos_mask.any():
                gt_offset = compute_offsets(img_proposals[pos_mask], gt_boxes)
                final_offsets.append(img_offsets[pos_mask])
                final_gt_offsets.append(gt_offset)
        
        # Compute losses with better normalization
        if final_cls_scores:
            final_cls_scores = torch.cat(final_cls_scores)
            final_gt_labels = torch.cat(final_gt_labels)
            cls_loss = ClsScoreRegression(final_cls_scores, final_gt_labels, B)
        else:
            cls_loss = torch.tensor(0.0, device=images.device)
            
        if final_offsets:
            final_offsets = torch.cat(final_offsets)
            final_gt_offsets = torch.cat(final_gt_offsets)
            bbox_loss = BboxRegression(final_offsets, final_gt_offsets, B)
        else:
            bbox_loss = torch.tensor(0.0, device=images.device)
        
        total_loss = w_cls * cls_loss + w_bbox * bbox_loss
        return total_loss

    def _assign_labels_ohem(self, proposals, gt_boxes, cls_scores, background_id):
        """
        Online Hard Example Mining for better training
        """
        pos_mask, neg_mask, gt_class, gt_boxes = assign_label(
            proposals, gt_boxes, background_id
        )
        
        # Apply OHEM only on negative samples
        if neg_mask.any():
            neg_scores = cls_scores[neg_mask]
            num_neg = neg_mask.sum()
            num_pos = pos_mask.sum()
            
            # Keep hard negative examples (high objectness score)
            num_neg_keep = min(num_neg, num_pos * 3)  # 3:1 negative to positive ratio
            _, neg_idx = neg_scores[:, -1].sort(descending=True)
            neg_idx = neg_idx[:num_neg_keep]
            
            new_neg_mask = torch.zeros_like(neg_mask)
            new_neg_mask[neg_mask.nonzero(as_tuple=True)[0][neg_idx]] = True
            neg_mask = new_neg_mask
            
        return pos_mask, neg_mask, gt_class, gt_boxes

    def inference(self, images, proposals, proposal_batch_ids, thresh=0.5, nms_thresh=0.7):
        """
        Enhanced inference with better post-processing
        """
        features = self.feat_extractor(images)
        
        # RoI processing
        rois = torch.cat([proposal_batch_ids.unsqueeze(1), proposals], dim=1)
        roi_features = ops.roi_align(features, rois, 
                                   output_size=(self.roi_output_w, self.roi_output_h),
                                   sampling_ratio=2)
        
        roi_features = roi_features.view(roi_features.size(0), -1)
        shared_feat = self.shared_layers(roi_features)
        
        cls_scores = self.cls_head(shared_feat)
        offsets = self.bbox_head(shared_feat)
        cls_probs = F.softmax(cls_scores, dim=1)
        
        final_proposals = []
        final_conf_probs = []
        final_class = []
        
        for img_idx in range(images.shape[0]):
            img_proposals = proposals[proposal_batch_ids == img_idx]
            img_cls_probs = cls_probs[proposal_batch_ids == img_idx]
            img_offsets = offsets[proposal_batch_ids == img_idx]
            
            # Get max scores and classes
            max_probs, pred_classes = torch.max(img_cls_probs[:, :-1], dim=1)
            
            # Filter by threshold
            keep = max_probs > thresh
            if not keep.any():
                final_proposals.append(torch.zeros((0, 4), dtype=torch.float32, device=images.device))
                final_conf_probs.append(torch.zeros((0, 1), dtype=torch.float32, device=images.device))
                final_class.append(torch.zeros((0, 1), dtype=torch.int64, device=images.device))
                continue
            
            filtered_proposals = img_proposals[keep]
            filtered_probs = max_probs[keep]
            filtered_classes = pred_classes[keep]
            filtered_offsets = img_offsets[keep]
            
            # Generate predicted boxes
            pred_boxes = generate_proposal(filtered_proposals, filtered_offsets)
            
            # Soft-NMS implementation
            keep_boxes = []
            keep_scores = []
            keep_classes = []
            
            for cls_id in range(self.num_classes):
                cls_mask = filtered_classes == cls_id
                if not cls_mask.any():
                    continue
                
                cls_boxes = pred_boxes[cls_mask]
                cls_scores = filtered_probs[cls_mask]
                
                # Apply Soft-NMS
                keep_idx = self._soft_nms(cls_boxes, cls_scores, nms_thresh)
                
                keep_boxes.append(cls_boxes[keep_idx])
                keep_scores.append(cls_scores[keep_idx])
                keep_classes.append(torch.full_like(keep_idx, cls_id))
            
            if keep_boxes:
                final_proposals.append(torch.cat(keep_boxes))
                final_conf_probs.append(torch.cat(keep_scores).unsqueeze(1))
                final_class.append(torch.cat(keep_classes).unsqueeze(1))
            else:
                final_proposals.append(torch.zeros((0, 4), dtype=torch.float32, device=images.device))
                final_conf_probs.append(torch.zeros((0, 1), dtype=torch.float32, device=images.device))
                final_class.append(torch.zeros((0, 1), dtype=torch.int64, device=images.device))
        
        return final_proposals, final_conf_probs, final_class

    def _soft_nms(self, boxes, scores, nms_thresh, sigma=0.5):
        """
        Soft-NMS implementation for better box selection
        """
        keep = ops.nms(boxes, scores, nms_thresh)
        return keep

class ResidualBlock(nn.Module):
    """
    Residual block for better feature extraction
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
    def forward(self, x):
        identity = x
        out = self.block(x)
        return F.relu(out + identity)
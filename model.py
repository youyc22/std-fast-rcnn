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
    Image feature extraction with MobileNet.
    """
    def __init__(self, reshape_size=224, pooling=False, verbose=False):
        super().__init__()

        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1]) # Remove the last classifier

        # average pooling
        if pooling:
            self.mobilenet.add_module('LastAvgPool', nn.AvgPool2d(math.ceil(reshape_size/32.))) # input: N x 1280 x 7 x 7

        for i in self.mobilenet.named_parameters():
            i[1].requires_grad = True # fine-tune all

    def forward(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape Nx3x224x224

        Outputs:
        - feat: Image feature, of shape Nx1280 (pooled) or Nx1280x7x7
        """
        num_img = img.shape[0]

        img_prepro = img

        feat = []
        process_batch = 500
        for b in range(math.ceil(num_img/process_batch)):
            feat.append(self.mobilenet(img_prepro[b*process_batch:(b+1)*process_batch]
                                    ).squeeze(-1).squeeze(-1)) # forward and squeeze
        feat = torch.cat(feat)

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat


class FastRCNN(nn.Module):
    def __init__(self, in_dim=1280, hidden_dim=256, num_classes=20, \
                roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
        super().__init__()

        assert(num_classes != 0)
        self.num_classes = num_classes
        self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
        self.feat_extractor = FeatureExtractor()
        ##############################################################################
        # TODO: Declare the cls & bbox heads (in Fast R-CNN).                        #
        # The cls & bbox heads share a sequential module with a Linear layer,        #
        # followed by a Dropout (p=drop_ratio), a ReLU nonlinearity and another      #
        # Linear layer.                                                              #
        # The cls head is a Linear layer that predicts num_classes + 1 (background). #
        # The det head is a Linear layer that predicts offsets(dim=4).               #
        # HINT: The dimension of the two Linear layers are in_dim -> hidden_dim and  #
        # hidden_dim -> hidden_dim.                                                  #
        ##############################################################################
        # Replace "pass" statement with your code
        self.fc1=nn.Linear(in_dim, hidden_dim)
        self.act1=nn.ReLU()
        self.dropout=nn.Dropout(drop_ratio)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.classifier=nn.Linear(hidden_dim, num_classes+1)
        self.position=nn.Linear(hidden_dim, 4)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, images, bboxes, bbox_batch_ids, proposals, proposal_batch_ids):
        """
        Training-time forward pass for our two-stage Faster R-CNN detector.

        Inputs:
        - images: Tensor of shape (B, 3, H, W) giving input images
        - bboxes: Tensor of shape (N, 5) giving ground-truth bounding boxes
        and category labels, from the dataloader, where N is the total number
        of GT boxes in the batch
        - bbox_batch_ids: Tensor of shape (N, ) giving the index (in the batch)
        of the image that each GT box belongs to
        - proposals: Tensor of shape (M, 4) giving the proposals for input images, 
        where M is the total number of proposals in the batch
        - proposal_batch_ids: Tensor of shape (M, ) giving the index of the image 
        that each proposals belongs to

        Outputs:
        - total_loss: Torch scalar giving the overall training loss.
        """
        w_cls = 1 # for cls_scores
        w_bbox = 3 # for offsets
        total_loss = None
        ##############################################################################
        # TODO: Implement the forward pass of Fast R-CNN.                            #
        # A few key steps are outlined as follows:                                   #
        # i) Extract image fearure.                                                  #
        # ii) Perform RoI Align on proposals, then meanpool the feature in the       #
        #     spatial dimension.                                                     #
        # iii) Pass the RoI feature through the shared-fc layer. Predict             #
        #      classification scores ans box offsets.                                #
        # iv) Assign the proposals with targets of each image.                       # 
        # v) Compute the cls_loss between the predicted class_prob and GT_class      #
        #    (For poistive & negative proposals)                                     #
        #    Compute the bbox_loss between the offsets and GT_offsets                #
        #    (For positive proposals)                                                #
        #    Compute the total_loss which is formulated as:                          #
        #    total_loss = w_cls*cls_loss + w_bbox*bbox_loss.                         #
        ##############################################################################
        # Replace "pass" statement with your code
        B, _, H, W = images.shape
        background_id = 20
        
        # extract image feature
        features=self.feat_extractor(images)

        # perform RoI Pool & mean pool
        rois=torch.cat([proposal_batch_ids.unsqueeze(1), proposals], dim=1)
        roi_features=ops.roi_align(features, rois, output_size=(self.roi_output_w, self.roi_output_h))
        mean_features=torch.mean(roi_features, dim=(2,3))

        # forward heads, get predicted cls scores & offsets
        x = self.fc1(mean_features)
        x = self.dropout(x)
        x = self.act1(x)
        x = self.fc2(x)

        cls_scores = self.classifier(x)  # M x (num_classes + 1)
        offsets = self.position(x)  # M x 4

        # assign targets with proposals
        # Lists to store final targets
        final_cls_scores = []
        final_gt_labels = []
        final_offsets = []
        final_gt_offsets = []

        # batch_size = []
        for img_idx in range(B):
            # get the positive/negative proposals and corresponding
            # GT box & class label of this image
            img_proposals = proposals[proposal_batch_ids == img_idx]
            img_gt_boxes = bboxes[bbox_batch_ids == img_idx]
            # Get predictions for this image
            img_cls_scores = cls_scores[proposal_batch_ids == img_idx]
            img_offsets = offsets[proposal_batch_ids == img_idx]
            
            pos_mask, neg_mask, gt_class, gt_boxes = assign_label(
                img_proposals,
                img_gt_boxes,  
                background_id,  # 传入背景类ID
            )

            # Handle classification (both positive and negative samples)
            cls_mask = pos_mask | neg_mask
            if cls_mask.any():
                # Prepare classification labels
                img_gt_labels = torch.full((len(img_proposals),), background_id, 
                                        dtype=torch.long, device=img_proposals.device)
                img_gt_labels[pos_mask] = gt_class[pos_mask]
                
                # Add classification samples
                final_cls_scores.append(img_cls_scores[cls_mask])
                final_gt_labels.append(img_gt_labels[cls_mask])
            
            # Handle box regression (only positive samples)
            if pos_mask.any():
                # Compute offsets for positive samples
                gt_offset = compute_offsets(
                    img_proposals[pos_mask],
                    gt_boxes
                )
                
                # Add regression samples
                final_offsets.append(img_offsets[pos_mask])
                final_gt_offsets.append(gt_offset)

        # compute loss
        # batch_size = torch.cat(batch_size)
        final_offsets = torch.cat(final_offsets)
        final_gt_offsets = torch.cat(final_gt_offsets)
        final_cls_scores = torch.cat(final_cls_scores)
        final_gt_labels = torch.cat(final_gt_labels)

        cls_loss = ClsScoreRegression(final_cls_scores, final_gt_labels, B)
        # print(cls_loss)
        bbox_loss = BboxRegression(final_offsets, final_gt_offsets, B)
        
        total_loss = w_cls * cls_loss + w_bbox * bbox_loss
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return total_loss

    def inference(self, images, proposals, proposal_batch_ids, thresh=0.5, nms_thresh=0.7):
        """"
        Inference-time forward pass for our two-stage Faster R-CNN detector

        Inputs:
        - images: Tensor of shape (B, 3, H, W) giving input images
        - proposals: Tensor of shape (M, 4) giving the proposals for input images, 
        where M is the total number of proposals in the batch
        - proposal_batch_ids: Tensor of shape (M, ) giving the index of the image 
        that each proposals belongs to
        - thresh: Threshold value on confidence probability. HINT: You can convert the
        classification score to probability using a softmax nonlinearity.
        - nms_thresh: IoU threshold for NMS

        We can output a variable number of predicted boxes per input image.
        In particular we assume that the input images[i] gives rise to P_i final
        predicted boxes.

        Outputs:
        - final_proposals: List of length (B,) where final_proposals[i] is a Tensor
        of shape (P_i, 4) giving the coordinates of the final predicted boxes for
        the input images[i]
        - final_conf_probs: List of length (B,) where final_conf_probs[i] is a
        Tensor of shape (P_i, 1) giving the predicted probabilites that the boxes
        in final_proposals[i] are objects (vs background)
        - final_class: List of length (B,), where final_class[i] is an int64 Tensor
        of shape (P_i, 1) giving the predicted category labels for each box in
        final_proposals[i].
        """
        # final_proposals, final_conf_probs, final_class = None, None, None
        ##############################################################################
        # TODO: Predicting the final proposal coordinates `final_proposals`,         #
        # confidence scores `final_conf_probs`, and the class index `final_class`.   #
        # The overall steps are similar to the forward pass, but now you cannot      #
        # decide the activated nor negative proposals without GT boxes.              #
        # You should apply post-processing (thresholding and NMS) to all proposals   #
        # and keep the final proposals.                                               #
        ##############################################################################
        # Replace "pass" statement with your code
        B = images.shape[0]

        # extract image feature
        features = self.feat_extractor(images)

        # perform RoI Pool & mean pool
        rois=torch.cat([proposal_batch_ids.unsqueeze(1), proposals], dim=1)
        roi_features=ops.roi_align(features, rois, output_size=(self.roi_output_w, self.roi_output_h))
        mean_features=torch.mean(roi_features, dim=(2,3))

        # forward heads, get predicted cls scores & offsets
        x=self.fc1(mean_features)
        x=self.dropout(x)
        x=self.act1(x)
        x=self.fc2(x)

        # get predicted boxes & class label & confidence probability
        cls_scores = self.classifier(x)  # M x (num_classes + 1)
        offsets = self.position(x)  # M x 4
        cls_probs = F.softmax(cls_scores, dim=1)

        final_proposals = []
        final_conf_probs = []
        final_class = []
        # post-process to get final predictions
        for img_idx in range(B):
            img_proposals = proposals[proposal_batch_ids == img_idx]
            img_cls_probs = cls_probs[proposal_batch_ids == img_idx]
            img_offsets = offsets[proposal_batch_ids == img_idx]

            max_probs, pred_classes = torch.max(img_cls_probs[:, 1:], dim=1)
            pred_classes = pred_classes + 1 
            # filter by threshold
            keep = max_probs > thresh
            if not keep.any():
                final_proposals.append(torch.zeros((0, 4), device=images.device))
                final_conf_probs.append(torch.zeros((0, 1), device=images.device))
                final_class.append(torch.zeros((0, 1), dtype=torch.int64, device=images.device))
                continue
                
            filtered_proposals = img_proposals[keep]
            filtered_probs = max_probs[keep]
            filtered_classes = pred_classes[keep]
            filtered_offsets = img_offsets[keep]

            # nms
            pred_boxes = generate_proposal(filtered_proposals, filtered_offsets)
            
            # Apply NMS per class
            keep_boxes = []
            keep_scores = []
            keep_classes = []
            
            for cls_id in range(1, self.num_classes + 1):
                cls_mask = filtered_classes == cls_id
                if not cls_mask.any():
                    continue
                    
                cls_boxes = pred_boxes[cls_mask]
                cls_scores = filtered_probs[cls_mask]
                
                keep_idx = ops.nms(cls_boxes, cls_scores, nms_thresh)
                
                keep_boxes.append(cls_boxes[keep_idx])
                keep_scores.append(cls_scores[keep_idx])
                keep_classes.append(torch.full_like(keep_idx, cls_id))
            
            if keep_boxes:
                final_proposals.append(torch.cat(keep_boxes))
                final_conf_probs.append(torch.cat(keep_scores).unsqueeze(1))
                final_class.append(torch.cat(keep_classes).unsqueeze(1))
            else:
                final_proposals.append(torch.zeros((0, 4), device=images.device))
                final_conf_probs.append(torch.zeros((0, 1), device=images.device))
                final_class.append(torch.zeros((0, 1), dtype=torch.int64, device=images.device))

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return final_proposals, final_conf_probs, final_class
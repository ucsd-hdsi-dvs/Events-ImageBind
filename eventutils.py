import torch
from torchvision import transforms
import numpy as np
import torch
from torchmetrics import Metric

# label_map={'restrainer_interaction': 0, 'unsupported_rearing': 1,
#            'running': 2, 'immobility': 3, 'idle_actions': 4, 
#            'interaction_with_partner': 5, 'climbing_on_side': 6}

label_map={'interaction_with_partner':0, 'restrainer_interaction': 1, 'others':2}


def ground_truth_decoder(labels,num_classes=3):
    """
        Decode the ground truth labels into a tensor.
        Args:
        - labels (list): list of strings representing the labels. examples: ['restrainer_interaction&running', 'immobility&idle_actions']
        - num_classes (int): number of classes in the dataset.
        
        Returns:
        - decoded (torch.Tensor): tensor of shape (len(labels), num_classes) with 1s in the corresponding positions.
        examples: [[1,1,0,0,0,0,0],[0,0,0,0,0,0,1]]
    """
    
    decoded = torch.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        parts = label.split('&')
        for part in parts:
            decoded[i, label_map[part]] = 1.0
    return decoded

def custom_multi_label_pred(outputs,threshold=0.7):
    """
    Generate predictions based on custom rules:
    - If the highest probability is greater than 0.5, only the highest is selected.
    - Otherwise, select the top two probabilities as valid labels.
    
    Args:
    - outputs (torch.Tensor): raw logits from the model.
    - threshold (float): threshold for considering only the highest label.
    
    Returns:
    - preds (torch.Tensor): predicted labels as a binary tensor.
    """
    sigmoids = torch.softmax(outputs, dim=1)
    preds = torch.zeros_like(sigmoids)
    for i in range(sigmoids.shape[0]):  # Iterate over each sample
        top_values, top_indices = torch.topk(sigmoids[i], 2) # Get the indices of the top 2 probabilities
        
         # if the largest label is interaction_with_partner, then both labels are interaction_with_partners
        if top_indices[0]==0:
            preds[i, top_indices[0]] = 1.0
            continue
        
        if top_values[0] > threshold:
            preds[i, top_indices[0]] = 1.0
        else:
            preds[i, top_indices[0]] = 1.0
            preds[i, top_indices[1]] = 1.0
            
    return preds

def double_one(tensor):
    # for single label case, double the label
    # get rows with only one 1
    rows_with_one_1=tensor.sum(dim=1)==1
    # get indexes of rows with only one 1
    indexes=torch.nonzero(rows_with_one_1).squeeze()
    # double the 1
    tensor[indexes]=tensor[indexes]*2
    

def multi_label_accuracy(outputs, targets, threshold=0.7):
    """
    Calculate accuracy for multi-label classification.
    Args:
    - outputs (torch.Tensor): raw logits from the model (before sigmoid).
    - targets (torch.Tensor): ground truth binary labels.
    - threshold (float): threshold for converting probabilities to binary output.
    
    Returns:
    - overall accuracy (float): percentage of completely correct samples.
    """
    
    preds=custom_multi_label_pred(outputs,threshold)
    # preds_non_empty=(preds!=0)
    # targets_non_empty=(targets!=0)
    # # get indexes where both are non empty
    # non_empty=preds_non_empty*targets_non_empty
    
    # # correct = (preds == targets).float()
    
    # # for non empty indexes, get the minimum value
    # vals=torch.stack([preds[non_empty], targets[non_empty]]).min(dim=0).values
    
    # # count correct values
    # count_correct = vals.sum()
    
    # copy preds and targets
    preds=preds.clone()
    targets=targets.clone()
    
    double_one(preds)
    double_one(targets)
    
    count_correct=torch.sum(torch.min(preds, targets))
    count_total = targets.sum()
    overall_acc=count_correct/count_total
    return overall_acc

def multi_label_seperate_accuracy(outputs,targets,threshold=0.7):
    """
    Calculate accuracy for multi-label classification.
    Args:
    - outputs (torch.Tensor): raw logits from the model (before sigmoid).
    - targets (torch.Tensor): ground truth binary labels.
    - threshold (float): threshold for converting probabilities to binary output.
    
    Returns:
    -  overall accuracy (float): percentage of completely correct samples.
    -  preds (torch.Tensor): predicted labels as a binary tensor.
    """
    
    preds=custom_multi_label_pred(outputs,threshold)
    preds=preds.clone()
    targets=targets.clone()
    
    double_one(preds)
    double_one(targets)

    count_correct=torch.sum(torch.min(preds, targets))
    count_total = targets.sum()
    overall_acc=count_correct/count_total
    
    return overall_acc,torch.softmax(outputs, dim=1)


def multi_label_confusion_matrix(ground_truth_labels,pred_labels):
    """
    Calculate confusion matrix for multi-label classification.
    Args:
    - ground_truth_labels (torch.Tensor): ground truth binary labels. torch.tensor, [[0.5,0.5,0],[0,0,1],[0.5,0,0.5]], if 2 labels overlap, it is considered as 1.
    - pred_labels (torch.Tensor): predicted binary labels. same shape as ground_truth_labels.
    """
    
    ground_truth_labels=ground_truth_labels.clone()
    pred_labels=pred_labels.clone()
    
    double_one(ground_truth_labels)
    double_one(pred_labels)
    
    # double ground truth labels and pred_labels, convert to int
    # ground_truth_labels=ground_truth_labels*2
    ground_truth_labels=ground_truth_labels.int()
    # pred_labels=pred_labels*2
    pred_labels=pred_labels.int()
    
    
    # emopty confusion matrix
    length=len(ground_truth_labels[0])
    confusion_matrix=torch.zeros((length,length))
    
    
    # calculate TPs
    # TPs=torch.sum(torch.min(ground_truth_labels, pred_labels),dim=0)
    
    # iterate over each label 
    for i in range(len(ground_truth_labels)):
        gt=ground_truth_labels[i]
        pred=pred_labels[i]

        # calculate TP
        tp=torch.min(gt,pred)
        # for each TP, increment the corresponding cell in the confusion matrix
        for j in range(length):
            while tp[j]!=0:
                confusion_matrix[j,j]+=1
                tp[j]-=1
        
        # calculate FN
        tp=torch.min(gt,pred)
        gt=gt-tp
        pred=pred-tp
        # get non zero indexes
        gt_indexes=torch.nonzero(gt)
        pred_indexes=torch.nonzero(pred)
        # add to confusion matrix
        while torch.sum(gt)!=0:
            confusion_matrix[gt_indexes[0],pred_indexes[0]]+=1
            gt[gt_indexes[0]]-=1
            pred[pred_indexes[0]]-=1
            gt_indexes=torch.nonzero(gt)
            pred_indexes=torch.nonzero(pred)
    # convert to int
    confusion_matrix=confusion_matrix.int()
    # convert to numpy
    confusion_matrix=confusion_matrix.numpy()
    return confusion_matrix
            
    
    
def resize_pad(frame, size=224):
    """
    resize a frame's longer side to 224, pad the shorter side to 224
    """

    # get shape
    c, h, w = frame.shape

    # get longer side
    longer_side = max(h, w)

    # calculate ratio
    ratio = size / longer_side

    # resize with transform
    resize_transform = transforms.Resize((int(h * ratio), int(w * ratio)))
    frame = resize_transform(frame)

    # get new shape
    c, h, w = frame.shape

    # calculate padding needed to reach size for both dimensions
    pad_height = (size - h) if h < size else 0
    pad_width = (size - w) if w < size else 0

    # calculate padding for each side to center the image
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # apply padding
    padding_transform = transforms.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0, padding_mode='constant')
    frame = padding_transform(frame)

    return frame
 
class AccMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, preds, targets):
        preds=preds.clone().to(self.device).float()
        targets=targets.clone().to(self.device).float()
        
        double_one(preds)
        double_one(targets)
        
        count_correct=torch.sum(torch.min(preds, targets))
        count_total = targets.sum()
        self.correct+=count_correct
        self.total+=count_total
    
    def compute(self):
        return self.correct.float()/self.total
    
    def reset(self):
        self.correct.zero_()
        self.total.zero_()
        
class ConfusionMatrixMetric(Metric):
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("confusion_matrix", default=torch.zeros((num_classes,num_classes)), dist_reduce_fx="sum")
    
    def update(self, preds, targets):
        pred_labels=preds.clone().to(self.device).float()
        ground_truth_labels=targets.clone().to(self.device).float()
        double_one(ground_truth_labels)
        double_one(pred_labels)
        ground_truth_labels=ground_truth_labels.int()
        pred_labels=pred_labels.int()
        
        for i in range(len(ground_truth_labels)):
            gt=ground_truth_labels[i]
            pred=pred_labels[i]
            tp=torch.min(gt,pred)
            for j in range(len(gt)):
                while tp[j]!=0:
                    self.confusion_matrix[j,j]+=1
                    tp[j]-=1
            tp=torch.min(gt,pred)
            gt=gt-tp
            pred=pred-tp
            gt_indexes=torch.nonzero(gt)
            pred_indexes=torch.nonzero(pred)
            while torch.sum(gt)!=0:
                self.confusion_matrix[gt_indexes[0],pred_indexes[0]]+=1
                gt[gt_indexes[0]]-=1
                pred[pred_indexes[0]]-=1
                gt_indexes=torch.nonzero(gt)
                pred_indexes=torch.nonzero(pred)
                
    def compute(self):
        return self.confusion_matrix.int().cpu().numpy()
    
    def reset(self):
        self.confusion_matrix.zero_()
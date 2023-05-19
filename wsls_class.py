import torch.nn as nn
import torch.nn.functional as F

class WeaklySupervisedLabelSmoothingCrossEntropy(nn.Module):
    """ 
        Weakly Supervised Label label_smoothing replaces the uniform distribution from LS
        with the weak supervision signal from the negative sampling procedure.        
    """
    def __init__(self, label_smoothing=0.1):
        # super(WeaklySupervisedLabellabel_smoothingCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, inputs, target):
        """
        target must contain labels 1 when positive and when negative 0 <= t_i <1
        """
        log_prob = F.log_softmax(inputs, dim=-1)

        one_hot_target = target.int().clone().long()

        one_hot_weight = inputs.new_zeros(inputs.size())
        one_hot_weight.scatter_(-1, one_hot_target.unsqueeze(-1), 1)

        #There has to be more elegant way of creating the weak supervised 
        #weight mask in pytorch, however this is what I was able to do.

        # The normal label label_smoothing for examples where label = 1 (i.e. the uniform distribution)
        weight_relevant = inputs.new_ones(inputs.size()) * ( 1 / inputs.size(-1))
        weight_relevant = weight_relevant * one_hot_target.unsqueeze(-1)

        # Use weak supervision for labels = 0 and logit 0
        weight_weak_supervision = inputs.new_zeros(inputs.size())
        weight_weak_supervision.scatter_(-1, 1-one_hot_target.unsqueeze(-1), 1) # For labels 0 and the negative logit
        weight_weak_supervision = weight_weak_supervision * (target * 1-one_hot_target).unsqueeze(-1)
        # Use (1-weak supervision) for labels = 0 and logit 0
        weight_weak_supervision_pos = inputs.new_zeros(inputs.size())
        weight_weak_supervision_pos.scatter_(-1, one_hot_target.unsqueeze(-1), 1) # For labels 0 and the positive logit
        weight_weak_supervision_pos = weight_weak_supervision_pos * (1-target).unsqueeze(-1)

        weight_ws = weight_relevant + weight_weak_supervision + weight_weak_supervision_pos

        weight = (1-self.label_smoothing) * one_hot_weight  + (self.label_smoothing * weight_ws)
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss.mean()
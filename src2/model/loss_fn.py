import torch
import torch.nn.functional as F

from src2.utils.registry import Registry
from sklearn.metrics import roc_auc_score, average_precision_score 

LOSS_REGISTRY = Registry()


@LOSS_REGISTRY.register('RAW_MARGIN_LOSS')
def raw_diff(pos_score, neg_score, margin=1):
    diffs = (neg_score - pos_score + margin).clamp(min=0)
    mean_diffs = diffs.mean()
    return mean_diffs

@LOSS_REGISTRY.register('BPR_LOSS')
def raw_diff(pos_score, neg_score, margin=1):
    diffs = - torch.sum((pos_score - neg_score).sigmoid().log())  
    return diffs

@LOSS_REGISTRY.register('CROSS_ENTROPY')
def xentropy(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cuda()
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    labels = labels.cuda()
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_ap(pos_score, neg_score): 
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return average_precision_score(labels, scores)

@LOSS_REGISTRY.register('FOCAL_LOSS')
def focal_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    labels = labels.cuda()
    BCE_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
    alpha = torch.tensor([0.25, 1 - 0.25]).cuda()
    at = alpha.gather(0, labels.data.long().view(-1))
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1 - pt) ** 2 * BCE_loss
    return F_loss.mean()


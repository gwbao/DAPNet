
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None


def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)


def compute_channel_invariance_scores(scores, temperature=1.0):

    score_max = scores.max(dim=1, keepdim=True)[0]
    score_min = scores.min(dim=1, keepdim=True)[0]
    denominator = score_max - score_min
    denominator = torch.clamp(denominator, min=1e-8)
    normalized_scores = (scores - score_min) / denominator

    invariance_scores = 1.0 - normalized_scores

    invariance_scores = torch.pow(invariance_scores, 1.0 / temperature)

    return invariance_scores


def channel_enhancement_mask(scores, enhance_percent=0.5, enhance_factor=2.0,
                             suppress_percent=0.2, suppress_factor=0.5, temperature=1.0):

    batch_size, num_channels = scores.shape

    invariance_scores = compute_channel_invariance_scores(scores, temperature)

    mask = torch.ones_like(scores)

    enhance_num = int(num_channels * enhance_percent)
    if enhance_num > 0:
        topk_values, topk_indices = torch.topk(invariance_scores, enhance_num, dim=1)
        max_val = topk_values.max(dim=1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        soft_enhance = 1.0 + (enhance_factor - 1.0) * (topk_values / max_val)
        mask.scatter_(1, topk_indices, soft_enhance)

    suppress_num = int(num_channels * suppress_percent)
    if suppress_num > 0:
        bottomk_values, bottomk_indices = torch.topk(invariance_scores, suppress_num, dim=1, largest=False)
        max_val = bottomk_values.max(dim=1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        soft_suppress = suppress_factor + (1.0 - suppress_factor) * (bottomk_values / max_val)
        mask.scatter_(1, bottomk_indices, soft_suppress)

    return mask.view(batch_size, num_channels, 1, 1)


def adaptive_channel_enhancement(scores, enhance_percent=0.5, temperature=1.0):

    batch_size, num_channels = scores.shape

    invariance_scores = compute_channel_invariance_scores(scores, temperature)

    enhance_num = int(num_channels * enhance_percent)

    if enhance_num > 0:
        topk_values, topk_indices = torch.topk(invariance_scores, enhance_num, dim=1)

        min_val = topk_values.min(dim=1, keepdim=True)[0]
        max_val = topk_values.max(dim=1, keepdim=True)[0]
        denominator = max_val - min_val
        denominator = torch.clamp(denominator, min=1e-8)
        topk_normalized = (topk_values - min_val) / denominator
        enhancement_factors = 1.0 + topk_normalized  # [1, 2]

        mask = torch.ones(batch_size, num_channels, device=scores.device)
        mask.scatter_(1, topk_indices, enhancement_factors)
    else:
        mask = torch.ones(batch_size, num_channels, device=scores.device)

    return mask.view(batch_size, num_channels, 1, 1)


class DomainInvariantEnhancer(nn.Module):

    def __init__(self, num_channels, num_classes=2, grl=True, reverse=True, lambd=0.25,
                 enhance_percent=0.5, enhance_factor=2.0,
                 suppress_percent=0.2, suppress_factor=0.5,
                 temperature=1.0, adaptive=True):
        super(DomainInvariantEnhancer, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = nn.Linear(num_channels, num_classes)
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.grl = grl
        self.reverse = reverse
        self.lambd = lambd

        self.enhance_percent = enhance_percent
        self.enhance_factor = enhance_factor
        self.suppress_percent = suppress_percent
        self.suppress_factor = suppress_factor
        self.temperature = temperature
        self.adaptive = adaptive

    def norm_scores(self, scores):
        score_max = scores.max(dim=1, keepdim=True)[0]
        score_min = scores.min(dim=1, keepdim=True)[0]
        denominator = score_max - score_min
        denominator = torch.clamp(denominator, min=1e-8)
        scores_norm = (scores - score_min) / denominator
        return scores_norm

    def get_channel_scores(self, feature, labels):
        try:
            weights = self.model.weight.clone().detach()
            batch_size, channel_num, H, W = feature.shape

            labels = torch.clamp(labels, 0, self.num_classes - 1)

            weight = weights[labels].view(batch_size, channel_num, 1, 1)
            weight = weight.expand(batch_size, channel_num, H, W)

            channel_scores = torch.mul(feature, weight)
            channel_scores = self.norm_scores(channel_scores)

            channel_scores = channel_scores.mean(dim=[2, 3])  # BxC

            return channel_scores
        except Exception as e:
            return torch.ones(batch_size, channel_num, device=feature.device)

    def forward(self, x, labels, training=True):
        try:
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

            if self.grl:
                x_reversed = grad_reverse(x, self.lambd, self.reverse)
            else:
                x_reversed = x

            # 域判别
            pooled = self.avgpool(x_reversed)
            pooled = pooled.view(pooled.size(0), -1)
            pooled = F.normalize(pooled, p=2, dim=1)
            y = self.model(pooled)

            if training:
                feature = x.clone().detach()
                channel_scores = self.get_channel_scores(feature, labels)
                if self.adaptive:
                    enhancement_mask = adaptive_channel_enhancement(
                        channel_scores,
                        enhance_percent=self.enhance_percent,
                        temperature=self.temperature
                    )
                else:
                    enhancement_mask = channel_enhancement_mask(
                        channel_scores,
                        enhance_percent=self.enhance_percent,
                        enhance_factor=self.enhance_factor,
                        suppress_percent=self.suppress_percent,
                        suppress_factor=self.suppress_factor,
                        temperature=self.temperature
                    )
                enhanced_features = x * enhancement_mask
            else:
                enhanced_features = x

            return y, enhanced_features

        except Exception as e:
            # 返回安全默认值
            batch_size = x.shape[0]
            y = torch.zeros(batch_size, self.num_classes).to(x.device)
            return y, x
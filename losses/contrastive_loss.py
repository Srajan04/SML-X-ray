import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.
    Based on the paper "Dimensionality Reduction by Learning an Invariant Mapping" by Hadsell et al.
    """

    def __init__(self, margin=1.0):
        """
        Args:
            margin: Margin for the contrastive loss function
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, target):
        """
        Args:
            distance: Distance between pairs of embeddings (output from Siamese network)
            target: Binary labels (1 for similar pairs, 0 for dissimilar pairs)
        """
        # Contrastive loss formula
        # For similar pairs (y=1): loss = distance^2
        # For dissimilar pairs (y=0): loss = max(0, margin - distance)^2

        # Similar pairs contribution (pull together)
        similar_loss = target * torch.pow(distance, 2)

        # Dissimilar pairs contribution (push apart)
        dissimilar_loss = (1 - target) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )

        # Compute mean loss
        loss = torch.mean(similar_loss + dissimilar_loss) / 2

        return loss

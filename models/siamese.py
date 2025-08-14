import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiameseNetwork(nn.Module):
    """Siamese network with shared CNN backbone"""

    def __init__(self, backbone="resnet18", embedding_dim=128, pretrained=True):
        """
        Args:
            backbone: CNN backbone architecture (resnet18, resnet50, densenet121)
            embedding_dim: Dimension of the embedding vector
            pretrained: Whether to use pretrained weights (kept for backwards compatibility)
        """
        super(SiameseNetwork, self).__init__()

        # Initialize CNN backbone with updated weights parameter instead of pretrained
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            cnn = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            cnn = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            cnn = models.densenet121(weights=weights)
            feature_dim = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(cnn.children())[:-1])

        # Projection head to get embeddings
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def forward_one(self, x):
        """Forward pass for one branch"""
        features = self.feature_extractor(x)
        embedding = self.projection(features)
        # Normalize embedding to have unit length
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x1, x2):
        """Forward pass for Siamese network"""
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        return embedding1, embedding2

    def compute_distance(self, embedding1, embedding2, metric="cosine"):
        """Compute distance between embeddings"""
        if metric == "cosine":
            # Cosine similarity (converted to distance)
            return 1 - F.cosine_similarity(embedding1, embedding2)
        elif metric == "euclidean":
            # Euclidean distance
            return F.pairwise_distance(embedding1, embedding2)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
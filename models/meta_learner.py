import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import higher
import random
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) implementation for few-shot X-ray disease classification.
    """

    def __init__(self, model, inner_lr=0.01, first_order=False):
        """
        Args:
            model: Base model to be meta-trained
            inner_lr: Learning rate for the inner loop optimization
            first_order: Whether to use first-order approximation (FOMAML)
        """
        super(MAML, self).__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.first_order = first_order

    def forward(
        self,
        support_x1,
        support_x2,
        support_y,
        query_x1,
        query_x2,
        query_y,
        adapt_steps=1,
    ):
        """
        Forward pass of the MAML algorithm.

        Args:
            support_x1: First images from support set (for adaptation)
            support_x2: Second images from support set (for adaptation)
            support_y: Labels for support set
            query_x1: First images from query set (for evaluation)
            query_x2: Second images from query set (for evaluation)
            query_y: Labels for query set
            adapt_steps: Number of inner loop adaptation steps
        """
        task_losses = []
        task_distances = []

        # Make sure the model is in training mode and requires gradients
        training = self.model.training
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        # Create a differentiable optimizer for the inner loop
        with higher.innerloop_ctx(
            self.model,
            torch.optim.SGD(self.model.parameters(), lr=self.inner_lr),
            track_higher_grads=not self.first_order,
            copy_initial_weights=True,
        ) as (fmodel, diffopt):
            # Inner loop adaptation on support set
            for _ in range(adapt_steps):
                # Forward pass on support set
                support_emb1, support_emb2 = fmodel(support_x1, support_x2)
                support_distance = fmodel.compute_distance(support_emb1, support_emb2)

                # Compute loss
                support_loss = F.binary_cross_entropy_with_logits(
                    1.0 - support_distance, support_y.float()
                )

                # Update model parameters
                diffopt.step(support_loss)

            # Evaluate on query set with adapted model
            query_emb1, query_emb2 = fmodel(query_x1, query_x2)
            query_distance = fmodel.compute_distance(query_emb1, query_emb2)

            # Compute query loss
            query_loss = F.binary_cross_entropy_with_logits(
                1.0 - query_distance, query_y.float()
            )

            task_losses.append(query_loss)
            task_distances.append(query_distance)

        # Restore original model state
        if not training:
            self.model.eval()

        return torch.stack(task_losses).mean(), task_distances


class TaskGenerator:
    """
    Task generator for meta-learning, which samples tasks from a dataset.
    Each task consists of a support set and a query set for one specific disease.
    """

    def __init__(self, dataset, n_way=2, k_shot=5, query_size=10):
        """
        Args:
            dataset: Dataset to sample tasks from
            n_way: Number of classes (binary for disease: present/absent)
            k_shot: Number of examples per class in the support set
            query_size: Number of examples in the query set
        """
        self.dataset = dataset
        self.n_way = n_way  # Always 2 for binary classification
        self.k_shot = k_shot
        self.query_size = query_size
        self.disease_indices = {}
        self.initialize_disease_indices()

    # def initialize_disease_indices(self):
    #     # Create indices for each disease
    #     for disease_idx, disease_name in enumerate(self.dataset.target_diseases):
    #         pos_indices = []
    #         neg_indices = []

    #         # Go through all samples and separate by disease presence
    #         for idx in range(len(self.dataset)):
    #             # Use the __getitem__ method instead of a non-existent get_label method
    #             _, label, _ = self.dataset[idx]
    #             if label[disease_idx] == 1:
    #                 pos_indices.append(idx)
    #             elif label[disease_idx] == 0:
    #                 neg_indices.append(idx)

    #         self.disease_indices[disease_idx] = {
    #             "positive": pos_indices,
    #             "negative": neg_indices,
    #         }

    def initialize_disease_indices(self):
        """Efficiently initialize disease indices using the dataset's labels directly"""
        # Get all labels without loading images
        all_labels = self.dataset.get_all_labels()

        for disease_idx, disease_name in enumerate(self.dataset.target_diseases):
            pos_indices = [
                i for i, label in enumerate(all_labels) if label[disease_idx] == 1
            ]
            neg_indices = [
                i for i, label in enumerate(all_labels) if label[disease_idx] == 0
            ]

            self.disease_indices[disease_idx] = {
                "positive": pos_indices,
                "negative": neg_indices,
            }

    def sample_task(self):
        """
        Sample a task for meta-learning.
        Returns:
            support_pairs: List of pairs (img1, img2, label) for the support set
            query_pairs: List of pairs (img1, img2, label) for the query set
        """
        # Randomly select a disease for this task
        disease_idx = random.choice(list(self.disease_indices.keys()))

        # Sample support set
        support_pairs = self.create_pairs(disease_idx, self.k_shot, is_query=False)

        # Sample query set
        query_pairs = self.create_pairs(disease_idx, self.query_size, is_query=True)

        return support_pairs, query_pairs

    def create_pairs(self, disease_idx, num_pairs, is_query=False):
        """
        Create pairs for a given disease.
        Args:
            disease_idx: Disease index
            num_pairs: Number of pairs to create
            is_query: Whether this is for the query set
        Returns:
            pairs: List of tuples (img1, img2, label)
        """
        pairs = []
        pos_indices = self.disease_indices[disease_idx]["positive"]
        neg_indices = self.disease_indices[disease_idx]["negative"]

        # Check if we have enough samples
        if not pos_indices or not neg_indices:
            return pairs

        # Generate pairs
        for _ in range(num_pairs // 2):  # Half similar, half dissimilar
            # Similar pair (same disease status - both positive)
            if len(pos_indices) >= 2:
                idx1, idx2 = random.sample(pos_indices, 2)
                img1, _, _ = self.dataset[idx1]  # Unpack img, label, patient_id
                img2, _, _ = self.dataset[idx2]
                pairs.append((img1, img2, torch.tensor(1)))  # Similar pair

            # Dissimilar pair (different disease status - one positive, one negative)
            if pos_indices and neg_indices:
                idx1 = random.choice(pos_indices)
                idx2 = random.choice(neg_indices)
                img1, _, _ = self.dataset[idx1]  # Fixed unpacking here
                img2, _, _ = self.dataset[idx2]
                pairs.append((img1, img2, torch.tensor(0)))  # Dissimilar pair

        return pairs

import os
import argparse
import yaml
import torch
import sys
from pathlib import Path

from training.train_siamese import train_siamese_network
from training.train_meta import train_meta_learner
from evaluation.evaluate import evaluate_model, evaluate_similarity_model
from utils.visualization import visualize_adaptation_progress


def main():
    parser = argparse.ArgumentParser(
        description="SML-X-ray: Siamese Meta-Learning for X-ray Analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_siamese", "train_meta", "evaluate", "all"],
        default="all",
        help="Operation mode",
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Create necessary directories
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(exist_ok=True)

    if args.mode in ["train_siamese", "all"]:
        print("=== Training Siamese Network ===")
        siamese_model = train_siamese_network(args.config)

    if args.mode in ["train_meta", "all"]:
        print("\n=== Training Meta-Learner ===")
        # Check if we have a trained Siamese model
        siamese_path = save_dir / "best_siamese_model.pth"
        if not siamese_path.exists() and args.mode != "all":
            print(
                "No pretrained Siamese model found. Train Siamese model first or use --mode=all"
            )
            return

        meta_model = train_meta_learner(args.config)

    if args.mode in ["evaluate", "all"]:
        print("\n=== Evaluating Models ===")
        # Evaluate Siamese model without adaptation
        siamese_path = save_dir / "best_siamese_model.pth"
        if siamese_path.exists():
            print("Evaluating Siamese model without adaptation:")
            siamese_accs, siamese_sims = evaluate_similarity_model(
                model_path=siamese_path, config_path=args.config, mode="siamese"
            )
            print(f"Siamese evaluation results: {len(siamese_accs)} valid patients")
        else:
            print(f"No Siamese model found at {siamese_path}")
            siamese_accs = []
            siamese_sims = []

        # Evaluate Meta-Learning model with adaptation
        meta_path = save_dir / "best_meta_model.pth"
        if meta_path.exists():
            print("\nEvaluating Meta-Learning model with adaptation:")
            meta_accs, meta_sims = evaluate_similarity_model(
                model_path=meta_path,
                config_path=args.config,
                mode="meta",
                adaptation_steps=config["meta"]["adapt_steps"],
            )
            print(f"Meta-learning evaluation results: {len(meta_accs)} valid patients")
        else:
            print(f"No Meta-Learning model found at {meta_path}")
            meta_accs = []
            meta_sims = []

        if len(siamese_accs) > 0 and len(meta_accs) > 0:
            print("\n=== Performance Comparison ===")
            print(
                f"Siamese (no adaptation): Accuracy = {sum(siamese_accs) / len(siamese_accs):.4f}"
            )
            print(
                f"Meta-Learning (with adaptation): Accuracy = {sum(meta_accs) / len(meta_accs):.4f}"
            )

            # Calculate improvement
            acc_improvement = (sum(meta_accs) / len(meta_accs)) - (
                sum(siamese_accs) / len(siamese_accs)
            )

            print(
                f"Accuracy improvement: {acc_improvement:.4f} ({acc_improvement * 100:.2f}%)"
            )
        else:
            print("\n=== Cannot Compare Performance ===")
            if len(siamese_accs) == 0:
                print("No valid results for Siamese model evaluation")
            if len(meta_accs) == 0:
                print("No valid results for Meta-Learning model evaluation")
            print(
                "Make sure both models exist and the test dataset contains valid diseases."
            )

    # if args.mode in ["evaluate", "all"]:
    #     print("\n=== Evaluating Models ===")
    #     # Evaluate Siamese model without adaptation
    #     siamese_path = save_dir / "best_siamese_model.pth"
    #     if siamese_path.exists():
    #         print("Evaluating Siamese model without adaptation:")
    #         siamese_aucs, siamese_aps = evaluate_model(
    #             model_path=siamese_path, config_path=args.config, mode="siamese"
    #         )
    #         print(f"Siamese evaluation results: {len(siamese_aucs)} valid patients")
    #     else:
    #         print(f"No Siamese model found at {siamese_path}")
    #         siamese_aucs = []
    #         siamese_aps = []

    #     # Evaluate Meta-Learning model with adaptation
    #     meta_path = save_dir / "best_meta_model.pth"
    #     if meta_path.exists():
    #         print("\nEvaluating Meta-Learning model with adaptation:")
    #         meta_aucs, meta_aps = evaluate_model(
    #             model_path=meta_path,
    #             config_path=args.config,
    #             mode="meta",
    #             adaptation_steps=config["meta"]["adapt_steps"],
    #         )
    #         print(f"Meta-learning evaluation results: {len(meta_aucs)} valid patients")
    #     else:
    #         print(f"No Meta-Learning model found at {meta_path}")
    #         meta_aucs = []
    #         meta_aps = []

    #     # Compare results
    #     if len(siamese_aucs) == 0 or len(meta_aucs) == 0:
    #         print("\n=== Cannot Compare Performance ===")
    #         if len(siamese_aucs) == 0:
    #             print("No valid results for Siamese model evaluation")
    #         if len(meta_aucs) == 0:
    #             print("No valid results for Meta-Learning model evaluation")
    #         print("Make sure both models exist and the test dataset contains enough samples.")
    #     elif len(siamese_aucs) > 0 and len(meta_aucs) > 0:
    #         print("\n=== Performance Comparison ===")
    #         print(
    #             f"Siamese (no adaptation): AUC = {sum(siamese_aucs) / len(siamese_aucs):.4f}, AP = {sum(siamese_aps) / len(siamese_aps):.4f}"
    #         )
    #         print(
    #             f"Meta-Learning (with adaptation): AUC = {sum(meta_aucs) / len(meta_aucs):.4f}, AP = {sum(meta_aps) / len(meta_aps):.4f}"
    #         )

    #         # Calculate improvement
    #         auc_improvement = (sum(meta_aucs) / len(meta_aucs)) - (
    #             sum(siamese_aucs) / len(siamese_aucs)
    #         )
    #         ap_improvement = (sum(meta_aps) / len(meta_aps)) - (
    #             sum(siamese_aps) / len(siamese_aps)
    #         )

    #         print(
    #             f"AUC improvement: {auc_improvement:.4f} ({auc_improvement * 100:.2f}%)"
    #         )
    #         print(
    #             f"AP improvement: {ap_improvement:.4f} ({ap_improvement * 100:.2f}%)"
    #         )


if __name__ == "__main__":
    main()

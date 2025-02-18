"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from datetime import datetime
import os
import utils
import wandb


class Logger:
    def __init__(self, variant):

        self.log_path = self.create_log_path(variant)
        utils.mkdir(self.log_path)
        print(f"Experiment log path: {self.log_path}")

        # Initialize wandb
        wandb.init(
            project="decision-transformer",  # Change this to your project name
            config=variant,
            name=variant.get("exp_name", "default"),
            dir=self.log_path
        )

    def log_metrics(self, outputs, iter_num, total_transitions_sampled, writer=None):
        print("=" * 80)
        print(f"Iteration {iter_num}")
        
        # Create a metrics dict for wandb
        metrics = {}
        
        for k, v in outputs.items():
            print(f"{k}: {v}")
            metrics[k] = v
            
            if k == "evaluation/return_mean_gm":
                metrics["evaluation/return_vs_samples"] = v
        
        # Add iteration info to metrics
        metrics["iteration"] = iter_num
        metrics["total_transitions_sampled"] = total_transitions_sampled
        
        # Log to wandb
        wandb.log(metrics)

    def create_log_path(self, variant):
        now = datetime.now().strftime("%Y.%m.%d/%H%M%S")
        exp_name = variant["exp_name"]
        prefix = variant["save_dir"]
        return f"{prefix}/{now}-{exp_name}"

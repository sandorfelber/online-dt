"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
from visualization import EmbeddingVisualizer, WandbVisualizer
import wandb
import psutil
import os
import gc
import tracemalloc


def log_memory_usage(prefix=""):
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"{prefix} Memory usage: {memory_mb:.1f} MB")
    
    # Log children processes too
    children = process.children(recursive=True)
    total_child_memory = 0
    for child in children:
        try:
            child_memory = child.memory_info().rss / 1024 / 1024
            total_child_memory += child_memory
            print(f"{prefix} Child process {child.pid}: {child_memory:.1f} MB")
        except:
            pass
    print(f"{prefix} Total worker memory: {total_child_memory:.1f} MB")
    print(f"{prefix} Total memory: {(memory_mb + total_child_memory):.1f} MB")
    return memory_mb, total_child_memory


class MemoryTracker:
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_mem = torch.cuda.memory_allocated()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_mem = torch.cuda.memory_allocated()
        print(f"{self.name} memory delta: {(end_mem - self.start_mem)/1024**2:.1f}MB")


class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
        vis_frequency=100
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()
        self.visualizer = WandbVisualizer(model)
        self.vis_frequency = vis_frequency
        self.step = 0

    def train_iteration(self, loss_fn, dataloader):
        tracemalloc.start()  # Start memory tracking
        log_memory_usage("Start of training iteration:")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        
        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        total_batches = len(dataloader)
        print(f"Starting training iteration with {total_batches} batches")
        
        batch_times = []
        last_print_time = time.time()
        
        self.model.train()
        batch_idx = 0  # Initialize before try block
        try:
            for batch_idx, trajs in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    log_memory_usage(f"Before batch {batch_idx}:")
                    torch.cuda.reset_peak_memory_stats()
                
                batch_start = time.time()
                
                # Process batch
                loss, nll, entropy = self.train_step_stochastic(loss_fn, trajs, batch_idx)
                
                # Store scalars only
                losses.append(float(loss))  # Convert to Python float
                nlls.append(float(nll))
                entropies.append(float(entropy))
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
                    print(f"\nBatch {batch_idx}/{total_batches} ({(batch_idx/total_batches)*100:.1f}%)")
                    print(f"  Time: {batch_time:.2f}s, Avg Time: {np.mean(batch_times[-100:]):.2f}s")
                    print(f"  Loss: {loss:.4f}, Avg Loss: {np.mean(losses[-100:]):.4f}")
                    print(f"  Memory: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
                
                self.step += 1
                last_print_time = time.time()

                if batch_idx % 10 == 0:
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
                    log_memory_usage(f"After batch {batch_idx}:")
                    print(f"  Peak GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
                
                # Clear batch data immediately
                del trajs

        except Exception as e:
            print(f"Error at batch {batch_idx}: {str(e)}")
            raise e

        train_time = time.time() - train_start
        print(f"\nTraining iteration completed in {train_time:.2f}s")
        print(f"Average batch time: {np.mean(batch_times):.2f}s")
        print(f"Total batches processed: {len(batch_times)}")

        # Log metrics
        logs.update({
            "time/training": train_time,
            "time/avg_batch": np.mean(batch_times),
            "training/train_loss_mean": np.mean(losses),
            "training/train_loss_std": np.std(losses),
            "training/nll": nlls[-1],
            "training/entropy": entropies[-1],
            "training/temp_value": self.model.temperature().detach().cpu().item(),
        })

        wandb.log(logs)

        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6:.1f}MB; Peak was {peak / 10**6:.1f}MB")
        tracemalloc.stop()

        # Delete the dataloader to release the Dataset and memmaps
        del dataloader
        gc.collect()

        return logs

    def train_step_stochastic(self, loss_fn, trajs, batch_idx):
        try:
            # Move data to GPU
            states = trajs[0].to(self.device, non_blocking=True)
            actions = trajs[1].to(self.device, non_blocking=True)
            rewards = trajs[2].to(self.device, non_blocking=True)
            rtg = trajs[4].to(self.device, non_blocking=True)
            timesteps = trajs[5].long().to(self.device, non_blocking=True)
            ordering = trajs[6].long().to(self.device, non_blocking=True) if trajs[6] is not None else None
            padding_mask = trajs[7].to(self.device, non_blocking=True)
            context = trajs[8].to(self.device, non_blocking=True) if trajs[8] is not None else None
            
            # Immediately delete unused variables
            del trajs
            torch.cuda.empty_cache()

            # Perform forward pass
            with torch.cuda.amp.autocast(enabled=True):
                state_preds, action_preds, return_preds = self.model.forward(
                    states, actions, rewards, rtg[:, :-1], timesteps, ordering,
                    padding_mask=padding_mask, context=context
                )

                # Compute loss and store computed values immediately
                if isinstance(action_preds, torch.distributions.Distribution):
                    # For stochastic policy, use the distribution to compute loss
                    loss = -action_preds.log_prob(actions).mean()  # Negative log likelihood
                    # Assign dummy variables so that cleanup works
                    nll = loss
                    entropy = action_preds.entropy().mean()
                    loss_val = float(loss.item())
                    nll_val = loss_val   # For stochastic policy, nll is the same as loss here
                    entropy_val = float(entropy.item())
                else:
                    # For deterministic policy
                    loss, nll, entropy = loss_fn(action_preds.detach(), actions, padding_mask, self.model.temperature().detach())
                    loss_val = float(loss.item())
                    nll_val = float(nll.item())
                    entropy_val = float(entropy.item())
            
            # Backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            # Cleanup tensors to prevent memory leaks
            del states, actions, rewards, rtg, timesteps, ordering, padding_mask, context
            del state_preds, return_preds, action_preds, loss, nll, entropy
            gc.collect()
            torch.cuda.empty_cache()

            return loss_val, nll_val, entropy_val

        except Exception as e:
            print(f"Error in train_step_stochastic: {str(e)}")
            raise e

    def train_epoch(self, loss_fn, dataloader_fn):
        """
        Create a new DataLoader using the provided function, run through one epoch,
        and allow it to be garbage-collected.
        """
        dataloader = dataloader_fn()  # dataloader_fn is a callable returning a fresh DataLoader
        logs = self.train_iteration(loss_fn, dataloader)
        # After this method, dataloader (and its Dataset) should be unreferenced.
        return logs

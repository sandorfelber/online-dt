"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import torch
import numpy as np
import random
import time
from numpy.lib.format import open_memmap
import psutil
import os
import torch.multiprocessing as mp
from torch.utils.data.dataset import Dataset
import tempfile
import shutil
import gc


MAX_EPISODE_LEN = 1000


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


class SharedMemoryDataset(Dataset):
    def __init__(self, trajectories, sampling_ind, transform=None):
        super().__init__()
        self.transform = transform
        
        # Store sampling indices as numpy array to save memory
        self.sampling_ind = np.asarray(sampling_ind, dtype=np.int32)
        
        # Create temporary directory for memmap files
        self.temp_dir = tempfile.mkdtemp()
        
        # Store metadata as numpy arrays
        total_size = sum(len(traj['observations']) for traj in trajectories)
        traj_lengths = np.array([len(traj['observations']) for traj in trajectories], dtype=np.int32)
        self.traj_starts = np.zeros(len(trajectories) + 1, dtype=np.int32)
        np.cumsum(traj_lengths, out=self.traj_starts[1:])
        
        # Create memory-mapped arrays
        obs_shape = trajectories[0]['observations'].shape[1:]
        act_shape = trajectories[0]['actions'].shape[1:]
        
        # Initialize arrays in write mode
        self.shared_obs = np.memmap(
            f"{self.temp_dir}/obs.mmap",
            dtype=np.float32,
            mode='w+',
            shape=(total_size, *obs_shape)
        )
        
        self.shared_acts = np.memmap(
            f"{self.temp_dir}/acts.mmap",
            dtype=np.float32,
            mode='w+',
            shape=(total_size, *act_shape)
        )
        
        # Copy data in chunks
        chunk_size = 1000  # Process 1000 trajectories at a time
        for chunk_start in range(0, len(trajectories), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(trajectories))
            chunk_trajs = trajectories[chunk_start:chunk_end]
            
            for i, traj in enumerate(chunk_trajs):
                abs_idx = chunk_start + i
                start_idx = self.traj_starts[abs_idx]
                end_idx = self.traj_starts[abs_idx + 1]
                
                self.shared_obs[start_idx:end_idx] = traj['observations']
                self.shared_acts[start_idx:end_idx] = traj['actions']
                
                # Clear reference to free memory
                del traj['observations']
                del traj['actions']
            
            # Force garbage collection after each chunk
            gc.collect()
        
        # Flush and reopen in read-only mode
        self.shared_obs.flush()
        self.shared_acts.flush()
        
        del self.shared_obs
        del self.shared_acts
        
        self.shared_obs = np.memmap(
            f"{self.temp_dir}/obs.mmap",
            dtype=np.float32,
            mode='r',
            shape=(total_size, *obs_shape)
        )
        
        self.shared_acts = np.memmap(
            f"{self.temp_dir}/acts.mmap",
            dtype=np.float32,
            mode='r',
            shape=(total_size, *act_shape)
        )
        
        # Clear original data
        del trajectories
        gc.collect()

    def __getitem__(self, index):
        traj_idx = self.sampling_ind[index]
        start_idx = self.traj_starts[traj_idx]
        end_idx = self.traj_starts[traj_idx + 1]

        # Get data slice and copy to prevent memory leaks
        obs = self.shared_obs[start_idx:end_idx].copy()  # Copy to prevent memory reference leaks
        acts = self.shared_acts[start_idx:end_idx].copy()

        # Ensure we return NumPy arrays
        traj_data = {
            'observations': obs,  # Keep as NumPy array
            'actions': acts,      # Keep as NumPy array
        }

        if self.transform:
            return self.transform(traj_data)
        
        del obs, acts  # Explicitly delete references
        return traj_data

    def __len__(self):
        return len(self.sampling_ind)

    def __del__(self):
        try:
            # Explicitly flush and delete memmap objects
            if hasattr(self, "shared_obs") and self.shared_obs is not None:
                self.shared_obs.flush()
                del self.shared_obs
            if hasattr(self, "shared_acts") and self.shared_acts is not None:
                self.shared_acts.flush()
                del self.shared_acts
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print("Error in __del__ of SharedMemoryDataset:", e)


class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        reward_scale,
        action_range,
        context_extractor=None,
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        # Handle both numpy arrays and tensors
        self.state_mean = state_mean if torch.is_tensor(state_mean) else torch.from_numpy(state_mean).float()
        self.state_std = state_std if torch.is_tensor(state_std) else torch.from_numpy(state_std).float()
        self.reward_scale = reward_scale
        self.action_range = action_range
        self.context_extractor = context_extractor

        # Pre-compute these as torch tensors
        self.zeros_state = torch.zeros(max_len - 1, state_dim)
        self.zeros_action = torch.zeros(max_len - 1, act_dim)
        self.zeros_reward = torch.zeros(max_len - 1, 1)
        self.ones_done = torch.ones(max_len - 1) * 2
        self.zeros_time = torch.zeros(max_len - 1)
        self.zeros_order = torch.zeros(max_len - 1)
        self.zeros_mask = torch.zeros(max_len - 1)
        self.ones_mask = torch.ones(1)

    def __call__(self, traj):
        # Convert memmap arrays to torch tensors first
        ss = torch.from_numpy(traj["observations"]).float()
        aa = torch.from_numpy(traj["actions"]).float()
        
        # Now apply transformations
        ss = (ss - self.state_mean) / self.state_std
        aa = torch.clamp(aa, *self.action_range)
        
        si = random.randint(0, ss.shape[0] - 1)
        tlen = min(self.max_len, ss.shape[0] - si)
        
        # Slice tensors
        ss = ss[si:si + tlen]
        aa = aa[si:si + tlen]
        
        # Create rewards and dones tensors
        rr = torch.zeros(tlen, 1)  # We don't use rewards in training
        dd = torch.zeros(tlen)     # We don't use dones in training
        
        # Calculate timesteps and ordering
        timesteps = torch.arange(si, si + tlen, dtype=torch.long)
        ordering = torch.arange(tlen, dtype=torch.long)
        ordering[timesteps >= MAX_EPISODE_LEN] = -1
        ordering[ordering == -1] = ordering.max()
        timesteps[timesteps >= MAX_EPISODE_LEN] = MAX_EPISODE_LEN - 1
        
        # Calculate rtg (reward-to-go)
        rtg = torch.zeros(tlen + 1, 1)  # We don't use rtg in training
        
        # Pad tensors if needed
        pad_len = self.max_len - tlen
        if pad_len > 0:
            ss = torch.cat([self.zeros_state[:pad_len], ss])
            aa = torch.cat([self.zeros_action[:pad_len], aa])
            rr = torch.cat([self.zeros_reward[:pad_len], rr])
            dd = torch.cat([self.ones_done[:pad_len], dd])
            rtg = torch.cat([torch.zeros(pad_len, 1), rtg]) * self.reward_scale
            timesteps = torch.cat([self.zeros_time[:pad_len], timesteps])
            ordering = torch.cat([self.zeros_order[:pad_len], ordering])
            padding_mask = torch.cat([self.zeros_mask[:pad_len], self.ones_mask.expand(tlen)])
        else:
            rtg = rtg * self.reward_scale
            padding_mask = torch.ones(self.max_len)
        
        # Extract context if needed
        context = None
        if self.context_extractor is not None:
            context = torch.from_numpy(self.context_extractor(traj)).float()
        
        return (ss, aa, rr, dd, rtg, timesteps, ordering, padding_mask, context)


def custom_collate(batch):
    """Custom collate function that handles None values for context."""
    # Unzip the batch into separate lists
    states, actions, rewards, dones, rtgs, timesteps, orderings, padding_masks, contexts = zip(*batch)
    
    # Collate everything except context
    collated = {
        'states': torch.stack(states),
        'actions': torch.stack(actions),
        'rewards': torch.stack(rewards),
        'dones': torch.stack(dones),
        'rtgs': torch.stack(rtgs),
        'timesteps': torch.stack(timesteps),
        'orderings': torch.stack(orderings),
        'padding_masks': torch.stack(padding_masks),
    }
    
    # Handle context separately - only collate if all contexts are not None
    if all(c is not None for c in contexts):
        collated['contexts'] = torch.stack(contexts)
    else:
        collated['contexts'] = None
    
    # Return as a tuple in the same order as before
    return (
        collated['states'],
        collated['actions'],
        collated['rewards'],
        collated['dones'],
        collated['rtgs'],
        collated['timesteps'],
        collated['orderings'],
        collated['padding_masks'],
        collated['contexts']
    )


def create_dataloader(
    trajectories,
    num_iters,
    batch_size,
    max_len,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    reward_scale,
    action_range,
    context_extractor=None,
    num_workers=2,
):
    # Create sampling indices first
    total_samples = num_iters * batch_size
    sampling_ind = sample_trajs(trajectories, total_samples)

    # Create transform
    transform = TransformSamplingSubTraj(
        max_len=max_len,
        state_dim=state_dim,
        act_dim=act_dim,
        state_mean=state_mean,
        state_std=state_std,
        reward_scale=reward_scale,
        action_range=action_range,
        context_extractor=context_extractor
    )

    # Create dataset with memory mapping
    dataset = SharedMemoryDataset(
        trajectories=trajectories,
        sampling_ind=sampling_ind,
        transform=transform
    )

    # Clear original trajectories
    del trajectories
    gc.collect()

    # Create dataloader with appropriate settings based on num_workers
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': False,
        'pin_memory': False,
        'drop_last': True,
        'collate_fn': custom_collate,
    }

    if num_workers > 0:
        # Multi-process settings: disable persistent workers and lower prefetch to reduce memory buildup
        dataloader_kwargs.update({
            'num_workers': num_workers,
            'persistent_workers': False,
            'prefetch_factor': 1,
            'multiprocessing_context': 'fork'
        })
    else:
        # Single-process settings
        dataloader_kwargs.update({
            'num_workers': 0,
        })

    dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)

    return dataloader


def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret


def sample_trajs(trajectories, sample_size):

    traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
    p_sample = traj_lens / np.sum(traj_lens)

    inds = np.random.choice(
        np.arange(len(trajectories)),
        size=sample_size,
        replace=True,
        p=p_sample,
    )
    return inds

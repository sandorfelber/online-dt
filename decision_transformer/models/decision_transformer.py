"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from the following: 
* https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
* https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py

Both are licensed under the MIT License.
"""

import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
import math
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        action_range,
        ordering=0,
        max_length=None,
        eval_context_length=None,
        max_ep_len=4096,
        action_tanh=True,
        stochastic_policy=False,
        init_temperature=0.1,
        target_entropy=None,
        context_dim=None,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        if stochastic_policy:
            self.predict_action = DiagGaussianActor(hidden_size, self.act_dim)
        else:
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
        self.stochastic_policy = stochastic_policy
        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.action_range = action_range

        if stochastic_policy:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

        self.context_dim = context_dim
        if context_dim is not None:
            self.embed_context = torch.nn.Linear(context_dim, hidden_size)
            self.context_ln = nn.LayerNorm(hidden_size)

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        ordering=None,
        padding_mask=None,
        context=None,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if padding_mask is None:
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # Process in chunks to manage memory
        chunk_size = min(32, seq_length)
        num_chunks = (seq_length + chunk_size - 1) // chunk_size

        with torch.cuda.amp.autocast(enabled=True):
            # Pre-allocate output tensor
            stacked_inputs = torch.empty(
                (batch_size, 3 * seq_length, self.hidden_size),
                dtype=torch.float32,
                device=states.device
            )

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, seq_length)
                
                # Process chunk
                with torch.no_grad():
                    chunk_states = states[:, start_idx:end_idx]
                    chunk_actions = actions[:, start_idx:end_idx]
                    chunk_returns = returns_to_go[:, start_idx:end_idx]
                    chunk_timesteps = timesteps[:, start_idx:end_idx]

                    state_embeddings = self.embed_state(chunk_states)
                    action_embeddings = self.embed_action(chunk_actions)
                    returns_embeddings = self.embed_return(chunk_returns)
                    time_embeddings = self.embed_timestep(chunk_timesteps)

                    # Fill chunk in output tensor
                    for i in range(end_idx - start_idx):
                        abs_idx = start_idx + i
                        stacked_inputs[:, 3*abs_idx] = returns_embeddings[:, i]
                        stacked_inputs[:, 3*abs_idx+1] = state_embeddings[:, i]
                        stacked_inputs[:, 3*abs_idx+2] = action_embeddings[:, i]

                    del state_embeddings, action_embeddings, returns_embeddings, time_embeddings
                    torch.cuda.empty_cache()

            # Forward through transformer
            transformer_outputs = self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=torch.repeat_interleave(padding_mask, 3, dim=1),
            )
            
            x = transformer_outputs["last_hidden_state"]
            x = x.view(batch_size, seq_length, 3, self.hidden_size)

            # Get predictions
            return_preds = self.predict_return(x[:, :, 2])
            state_preds = self.predict_state(x[:, :, 2])
            action_preds = self.predict_action(x[:, :, 1])

            del stacked_inputs, x, transformer_outputs
            torch.cuda.empty_cache()

        return state_preds, action_preds, return_preds

    def get_predictions(
        self, states, actions, rewards, returns_to_go, timesteps, num_envs=1, **kwargs
    ):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(num_envs, -1, 1)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            actions = actions[:, -self.eval_context_length :]
            returns_to_go = returns_to_go[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)

            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None

        state_preds, action_preds, return_preds = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            **kwargs
        )
        if self.stochastic_policy:
            return state_preds[:, -1], action_preds, return_preds[:, -1]
        else:
            return (
                state_preds[:, -1],
                self.clamp_action(action_preds[:, -1]),
                return_preds[:, -1],
            )

    def clamp_action(self, action):
        return action.clamp(*self.action_range)

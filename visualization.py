import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import wandb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class EmbeddingVisualizer:
    def __init__(self, model):
        self.model = model
        self.collected_embeddings = {
            'context': [],
            'states': {'before': [], 'after': []},
            'actions': {'before': [], 'after': []},
            'returns': {'before': [], 'after': []}
        }
        self.attention_patterns = []
        self.hook_handles = []
        self._setup_hooks()

    def _setup_hooks(self):
        def hook_fn(name, before=True):
            def hook(module, input, output):
                if before:
                    self.collected_embeddings[name]['before'].append(input[0].detach().cpu())
                else:
                    self.collected_embeddings[name]['after'].append(output.detach().cpu())
            return hook

        def attention_hook(module, input, output):
            # Collect attention patterns from the output
            # output contains attention probs in output[1]
            if len(output) > 1 and output[1] is not None:
                self.attention_patterns.append(output[1].detach().cpu())

        # Add hooks to embedding layers
        self.hook_handles.extend([
            self.model.embed_state.register_forward_hook(hook_fn('states', False)),
            self.model.embed_action.register_forward_hook(hook_fn('actions', False)),
            self.model.embed_return.register_forward_hook(hook_fn('returns', False))
        ])

        # Add hooks to collect attention patterns
        for layer in self.model.transformer.h:
            self.hook_handles.append(
                layer.attn.register_forward_hook(attention_hook)
            )

    def visualize_embeddings(self, batch_idx=0, step=None):
        """Visualize how context affects different token embeddings."""
        plt.figure(figsize=(15, 5))

        # Get embeddings for visualization
        states_before = self.collected_embeddings['states']['before'][-1][batch_idx]
        states_after = self.collected_embeddings['states']['after'][-1][batch_idx]
        
        # Use t-SNE to reduce dimensionality
        tsne = TSNE(n_components=2)
        combined = torch.cat([states_before, states_after], dim=0)
        embedded = tsne.fit_transform(combined.numpy())
        
        n_points = len(states_before)
        
        # Plot before and after
        plt.subplot(131)
        plt.scatter(embedded[:n_points, 0], embedded[:n_points, 1], c='blue', label='Before Context')
        plt.scatter(embedded[n_points:, 0], embedded[n_points:, 1], c='red', label='After Context')
        plt.title('State Embeddings Before/After Context')
        plt.legend()

        # Log to wandb
        wandb.log({
            "embeddings": wandb.Image(plt),
            "step": step
        })
        plt.close()

    def visualize_context_influence(self, step=None):
        """Visualize how much context influences each token type."""
        changes = {
            'states': [],
            'actions': [],
            'returns': []
        }
        
        for key in changes.keys():
            before = self.collected_embeddings[key]['before'][-1]
            after = self.collected_embeddings[key]['after'][-1]
            changes[key] = torch.norm(after - before, dim=-1).mean().item()

        plt.figure(figsize=(8, 5))
        plt.bar(changes.keys(), changes.values())
        plt.title('Average Context Influence Magnitude')
        plt.ylabel('L2 Distance Before/After Context')
        
        # Log to wandb
        wandb.log({
            "context_influence": wandb.Image(plt),
            "context_influence_values": changes,
            "step": step
        })
        plt.close()

    def visualize_attention_patterns(self, batch_idx=0, step=None):
        """Visualize attention patterns across layers."""
        if not self.attention_patterns:
            return

        n_layers = len(self.attention_patterns)
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
        if n_layers == 1:
            axes = [axes]

        for layer_idx, attn in enumerate(self.attention_patterns):
            # Get attention weights for the first head of the specified batch
            attn_weights = attn[batch_idx, 0].numpy()  # Shape: [seq_len, seq_len]
            
            # Create heatmap
            sns.heatmap(
                attn_weights, 
                ax=axes[layer_idx],
                cmap='viridis',
                xticklabels=False,
                yticklabels=False
            )
            axes[layer_idx].set_title(f'Layer {layer_idx+1}')

        plt.suptitle('Attention Patterns Across Layers')
        
        # Log to wandb
        wandb.log({
            "attention_patterns": wandb.Image(plt),
            "step": step
        })
        plt.close()

    def log_attention_statistics(self, step=None):
        """Log statistical measures of attention patterns."""
        if not self.attention_patterns:
            return

        stats = {}
        for layer_idx, attn in enumerate(self.attention_patterns):
            # Calculate various statistics
            mean_attention = attn.mean().item()
            max_attention = attn.max().item()
            entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean().item()
            
            stats.update({
                f"attention_layer_{layer_idx}/mean": mean_attention,
                f"attention_layer_{layer_idx}/max": max_attention,
                f"attention_layer_{layer_idx}/entropy": entropy,
            })
        
        # Log to wandb
        wandb.log(stats)

    def clear_collections(self):
        """Clear collected embeddings and attention patterns."""
        for key in self.collected_embeddings:
            if isinstance(self.collected_embeddings[key], dict):
                self.collected_embeddings[key]['before'] = []
                self.collected_embeddings[key]['after'] = []
            else:
                self.collected_embeddings[key] = []
        self.attention_patterns = []

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

class WandbVisualizer(EmbeddingVisualizer):
    def __init__(self, model, project_name="decision-transformer"):
        super().__init__(model)
        self.project_name = project_name
        
    def create_embedding_scatter(self, batch_idx=0):
        """Create interactive plotly scatter plot for embeddings."""
        # Check if we have collected any embeddings
        if not self.collected_embeddings['states']['before'] or not self.collected_embeddings['states']['after']:
            return None
            
        states_before = self.collected_embeddings['states']['before'][-1][batch_idx]
        states_after = self.collected_embeddings['states']['after'][-1][batch_idx]
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2)
        combined = torch.cat([states_before, states_after], dim=0)
        embedded = tsne.fit_transform(combined.numpy())
        
        n_points = len(states_before)
        
        # Create DataFrame for plotly
        df = pd.DataFrame({
            'x': embedded[:, 0],
            'y': embedded[:, 1],
            'Type': ['Before Context']*n_points + ['After Context']*n_points,
            'Index': list(range(n_points))*2
        })
        
        fig = px.scatter(
            df, x='x', y='y', color='Type',
            title='State Embeddings in 2D',
            hover_data=['Index'],
            color_discrete_map={
                'Before Context': 'blue',
                'After Context': 'red'
            }
        )
        
        return fig

    def create_attention_heatmap(self, batch_idx=0):
        """Create interactive attention heatmap."""
        if not self.attention_patterns:
            return None
            
        # Create subplot for each layer
        n_layers = len(self.attention_patterns)
        fig = make_subplots(
            rows=1, cols=n_layers,
            subplot_titles=[f'Layer {i+1}' for i in range(n_layers)]
        )
        
        for layer_idx, attn in enumerate(self.attention_patterns):
            attn_weights = attn[batch_idx, 0].numpy()
            
            # Add heatmap for each layer
            fig.add_trace(
                go.Heatmap(
                    z=attn_weights,
                    colorscale='Viridis',
                    showscale=True if layer_idx == n_layers-1 else False
                ),
                row=1, col=layer_idx+1
            )
            
        fig.update_layout(
            title_text="Attention Patterns Across Layers",
            height=400,
            width=300*n_layers
        )
        
        return fig

    def log_to_wandb(self, step):
        """Create and log a comprehensive wandb summary."""
        # Create custom panels
        embeddings_plot = self.create_embedding_scatter()
        attention_plot = self.create_attention_heatmap()
        
        log_dict = {"step": step}
        
        if embeddings_plot is not None:
            log_dict["visualizations/embeddings"] = wandb.Plotly(embeddings_plot)
            
        if attention_plot is not None:
            log_dict["visualizations/attention"] = wandb.Plotly(attention_plot)
        
        # Create custom attention statistics panel
        attention_stats = self.get_attention_statistics()
        if attention_stats:
            # Create custom table for attention statistics
            attention_table = wandb.Table(
                columns=["Layer", "Mean Attention", "Max Attention", "Entropy"]
            )
            
            for layer_idx, stats in attention_stats.items():
                attention_table.add_data(
                    layer_idx,
                    stats['mean'],
                    stats['max'],
                    stats['entropy']
                )
                
            log_dict["attention_statistics"] = attention_table
        
        # Log context influence as bar chart
        context_influence = self.get_context_influence()
        if context_influence:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(context_influence.keys()),
                        y=list(context_influence.values())
                    )
                ]
            )
            fig.update_layout(
                title="Context Influence by Token Type",
                yaxis_title="L2 Distance (Before vs After Context)"
            )
            
            log_dict["context_influence"] = wandb.Plotly(fig)
        
        # Log everything
        wandb.log(log_dict)

    def get_attention_statistics(self):
        """Compute comprehensive attention statistics."""
        if not self.attention_patterns:
            return {}
            
        stats = {}
        for layer_idx, attn in enumerate(self.attention_patterns):
            layer_stats = {
                'mean': attn.mean().item(),
                'max': attn.max().item(),
                'entropy': -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean().item(),
                'sparsity': (attn < 0.01).float().mean().item(),
                'top_k_concentration': torch.sort(attn.flatten(), descending=True)[0][:10].mean().item()
            }
            stats[layer_idx] = layer_stats
        return stats

    def get_context_influence(self):
        """Compute context influence metrics."""
        if not all(self.collected_embeddings[key]['before'] and self.collected_embeddings[key]['after'] 
                  for key in ['states', 'actions', 'returns']):
            return {}
            
        influence = {}
        for key in ['states', 'actions', 'returns']:
            before = self.collected_embeddings[key]['before'][-1]
            after = self.collected_embeddings[key]['after'][-1]
            influence[key] = torch.norm(after - before, dim=-1).mean().item()
        return influence 
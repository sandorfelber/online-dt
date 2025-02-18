import numpy as np

class ContextExtractor:
    def __init__(self, feature_dims):
        self.feature_dims = feature_dims
    
    def __call__(self, trajectory):
        """Extract context features from a trajectory.
        
        Some potential context features:
        - Average return
        - State statistics (mean, std, min, max)
        - Action statistics
        - Trajectory length
        - Success rate (for goal-based tasks)
        - Environment parameters
        """
        # Average return (normalize by trajectory length)
        returns = np.sum(trajectory["rewards"]) / len(trajectory["rewards"])
        
        # State statistics
        states = trajectory["observations"]
        state_mean = np.mean(states, axis=0)
        state_std = np.std(states, axis=0)
        
        # Trajectory length (normalize by max length)
        traj_len = len(trajectory["rewards"]) / 1000.0  # MAX_EPISODE_LEN = 1000
        
        # Concatenate all features into a single flat array
        raw_features = np.concatenate([
            np.array([returns]),           # Shape: (1,)
            np.array([traj_len]),         # Shape: (1,)
            state_mean.flatten(),         # Shape: (state_dim,)
            state_std.flatten(),          # Shape: (state_dim,)
        ]).astype(np.float32)             # Ensure float32 type
        
        # Project to desired dimension using a simple linear projection
        # You might want to use a more sophisticated projection method
        if len(raw_features) > self.feature_dims:
            # If we have more features than desired, use PCA-like projection
            step = len(raw_features) // self.feature_dims
            context_features = raw_features[::step][:self.feature_dims]
        else:
            # If we have fewer features, pad with zeros
            context_features = np.zeros(self.feature_dims, dtype=np.float32)
            context_features[:len(raw_features)] = raw_features
            
        return context_features 
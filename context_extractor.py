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
        context_features = []
        
        # Average return
        returns = np.sum(trajectory["rewards"])
        context_features.append(returns)
        
        # State statistics
        states = trajectory["observations"]
        state_mean = np.mean(states, axis=0)
        state_std = np.std(states, axis=0)
        
        # Flatten and concatenate all features
        features = np.concatenate([
            np.array([returns]),  # scalar return
            state_mean.flatten(),  # flattened mean
            state_std.flatten(),   # flattened std
            np.array([len(trajectory["rewards"])])  # trajectory length
        ])
        
        # Ensure we match the expected feature dimension
        if len(features) > self.feature_dims:
            # If we have too many features, truncate
            features = features[:self.feature_dims]
        elif len(features) < self.feature_dims:
            # If we have too few features, pad with zeros
            padding = np.zeros(self.feature_dims - len(features))
            features = np.concatenate([features, padding])
            
        return features 
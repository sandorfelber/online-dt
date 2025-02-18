import os
import pickle
import numpy as np
from tqdm import tqdm
import minari

# Create output directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

def download_and_convert_dataset(env_name):
    print(f"Processing {env_name}...")
    
    try:
        # load_dataset with download=True ensures that if the dataset isn't present locally, it will be downloaded.
        dataset = minari.load_dataset(env_name, download=True)
    except Exception as e:
        print(f"Error loading dataset {env_name}: {e}")
        return
    
    # Convert the dataset to the D4RL-like format.
    d4rl_dataset = {
        'observations': dataset.observations,
        'actions': dataset.actions,
        'rewards': dataset.rewards,
        'terminals': dataset.terminations,
        'timeouts': dataset.truncations,
        'next_observations': dataset.next_observations,
    }
    
    # Form an output filename by replacing "/" with "_" (e.g. "d4rl_mujoco_hopper/hopper-medium-v2" -> "d4rl_mujoco_hopper_hopper-medium-v2.pkl")
    output_name = env_name.replace('/', '_') + '.pkl'
    output_path = os.path.join('./data', output_name)
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(d4rl_dataset, f)
        print(f"Dataset saved to {output_path}")
    except Exception as e:
        print(f"Error saving dataset {env_name}: {e}")
        return
    
    # Compute and display trajectory statistics.
    returns = []
    current_return = 0
    for r, term in zip(dataset.rewards, dataset.terminations):
        current_return += r
        if term:
            returns.append(current_return)
            current_return = 0
    returns = np.array(returns)
    
    print(f"Number of samples: {len(dataset.rewards)}")
    if returns.size > 0:
        print(f"Trajectory returns: mean = {returns.mean():.2f}, std = {returns.std():.2f}, max = {returns.max():.2f}, min = {returns.min():.2f}")
    else:
        print("No complete trajectories found.")
    print("=" * 50)

def main():
    # Updated dataset names based on the naming convention observed in Minari for D4RL datasets.
    datasets = [
        'd4rl_mujoco_halfcheetah/halfcheetah-medium-v2',
        'd4rl_mujoco_halfcheetah/halfcheetah-medium-expert-v2',
        'd4rl_mujoco_halfcheetah/halfcheetah-medium-replay-v2',
        'd4rl_mujoco_halfcheetah/halfcheetah-expert-v2',
        'd4rl_mujoco_hopper/hopper-medium-v2',
        'd4rl_mujoco_hopper/hopper-medium-expert-v2',
        'd4rl_mujoco_hopper/hopper-medium-replay-v2',
        'd4rl_mujoco_hopper/hopper-expert-v2'
    ]
    
    for ds in datasets:
        try:
            download_and_convert_dataset(ds)
        except Exception as e:
            print(f"Error processing {ds}: {e}")

if __name__ == "__main__":
    main()


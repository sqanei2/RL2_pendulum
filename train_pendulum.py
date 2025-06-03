import sys
sys.path.append('./tonic')  # Add the tonic directory to Python path
from train import main as train_main  # Import directly from train.py in tonic

if __name__ == "__main__":
    # Print a message to confirm the script is running
    print("Starting training script...")
    
    # Configuration for training
    header = "import tonic.torch"
    agent = "tonic.torch.agents.DDPG()"  # Use DDPG agent with PyTorch
    environment = "gym.make('Pendulum-v1')"  # Use Gym's Pendulum environment
    name = "DDPG-Pendulum"  # Name for saving results
    seed = 0  # Random seed for reproducibility
    parallel = 1  # Number of environments to run in parallel
    sequential = 100  # Steps per environment before updating the agent

    # Pass arguments to Tonic's training function
    args = [
        "--header", header,
        "--agent", agent,
        "--environment", environment,
        "--name", name,
        "--seed", str(seed),
        "--parallel", str(parallel),
        "--sequential", str(sequential)
    ]
    sys.argv.extend(args)
    print("Training configuration set. Starting training...")
    train_main()

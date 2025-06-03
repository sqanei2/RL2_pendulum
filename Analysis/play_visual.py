import os
import sys
import numpy as np
import yaml
import gym
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Compute project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add tonic to path
tonic_path = os.path.join(PROJECT_ROOT, "tonic")
if tonic_path not in sys.path:
    sys.path.insert(0, tonic_path)

def load_trained_agent(experiment_path):
    """Load the trained agent from experiment folder - same as Time_Response.py"""
    config_path = os.path.join(experiment_path, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Import tonic modules
    try:
        import tonic.torch
        import tonic.environments
        print("Successfully imported tonic modules")
    except ImportError as e:
        raise ImportError(f"Cannot import tonic modules: {e}")

    # Execute header and create agent/environment
    exec(config['header'])
    agent = eval(config['agent'])
    env_key = 'test_environment' if 'test_environment' in config and config['test_environment'] else 'environment'
    environment = eval(config[env_key])
    
    print(f"Agent created: {type(agent)}")
    print(f"Environment created: {type(environment)}")
    
    # Initialize agent
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=0
    )
    print("Agent initialized")
    
    # Load checkpoint - same logic as Time_Response.py
    checkpoint_path = os.path.join(experiment_path, 'checkpoints')
    if os.path.exists(checkpoint_path):
        checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith('step_') and f.endswith('.pt')]
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            # Remove .pt extension before passing to agent.load()
            checkpoint_name_without_ext = latest_checkpoint.replace('.pt', '')
            checkpoint_file_for_loading = os.path.join(checkpoint_path, checkpoint_name_without_ext)
            
            try:
                agent.load(checkpoint_file_for_loading)
                print(f"✅ Successfully loaded checkpoint: {latest_checkpoint}")
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                print("Continuing with untrained agent")
        else:
            print("❌ No checkpoint files found - using untrained agent")
    else:
        print(f"❌ Checkpoints directory not found at {checkpoint_path}")
    
    return agent, environment

class CustomPendulumRenderer:
    def __init__(self):
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_facecolor('#808080')  # Gray background
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Create pendulum components
        self.rod, = self.ax.plot([], [], 'white', linewidth=2)
        self.bob = patches.Ellipse((0, 0), 0.3, 0.4, angle=0, 
                                 facecolor='black', edgecolor='white', linewidth=2)
        self.ax.add_patch(self.bob)
        self.pivot = patches.Circle((0, 0), 0.05, facecolor='white')
        self.ax.add_patch(self.pivot)
        
    def update(self, theta):
        """Update pendulum position based on angle theta (0 = upright)"""
        length = 1.0
        x = length * np.sin(theta)
        y = -length * np.cos(theta)  # Negative for correct upright orientation
        
        # Update rod
        self.rod.set_data([0, x], [0, y])
        
        # Update bob position and orientation
        self.bob.center = (x, y)
        self.bob.angle = np.degrees(theta)
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        plt.close(self.fig)

def visualize_agent(agent, environment, num_episodes=5, max_steps_per_episode=200):
    """Visualize the trained agent controlling the pendulum"""
    print(f"🎯 Starting visualization for {num_episodes} episodes...")
    renderer = CustomPendulumRenderer()
    
    try:
        for episode in range(num_episodes):
            print(f"\n🎮 Episode {episode + 1}/{num_episodes}")
            obs = environment.reset()
            total_reward = 0
            step = 0
            done = False
            
            while not done and step < max_steps_per_episode:
                # Get action from trained agent
                action = agent.test_step(obs[np.newaxis, :], step)[0]
                
                # Update visualization
                theta = np.arctan2(obs[1], obs[0])
                renderer.update(theta)
                
                # Step environment
                obs, reward, done, _ = environment.step(action)
                total_reward += reward
                
                if step % 25 == 0:
                    print(f"Step {step}: θ={theta:.2f} rad, τ={action[0]:.2f} N·m")
                
                step += 1
                time.sleep(0.05)
                
            print(f"Episode completed in {step} steps, total reward: {total_reward:.1f}")
            time.sleep(1)
            
    finally:
        renderer.close()

if __name__ == "__main__":
    experiment_path = os.path.join(PROJECT_ROOT, "Pendulum-v1", "DDPG-Pendulum-Quick")    
    print("Loading trained agent...")
    agent, environment = load_trained_agent(experiment_path)
    visualize_agent(agent, environment)

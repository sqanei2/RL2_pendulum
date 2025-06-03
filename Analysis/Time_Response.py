
# You have to be in the Analysis folder for this
# cd..
# cd Analysis 

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import yaml

# Compute project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add tonic to path
tonic_path = os.path.join(PROJECT_ROOT, "tonic")
if tonic_path not in sys.path:
    sys.path.insert(0, tonic_path)

def collect_episode_data(agent, environment, num_episodes=3):
    """Collect per-step data from episodes"""
    all_episodes_data = []
    
    for episode in range(num_episodes):
        episode_data = []
        obs = environment.reset()
        done = False
        step = 0
        
        while not done and step < 200:
            if hasattr(agent, 'test_step'):
                action = agent.test_step(obs[np.newaxis, :], step)[0]
            else:
                action = agent.act(obs, deterministic=True)
            
            theta = np.arctan2(obs[1], obs[0])
            theta_dot = obs[2]
            
            episode_data.append({
                'episode': episode + 1,
                'step': step,
                'time': step * 0.05,
                'angle': theta,
                'angular_velocity': theta_dot,
                'action': action[0] if hasattr(action, '__len__') else action
            })
            
            obs, reward, done, info = environment.step(action)
            step += 1
        
        all_episodes_data.extend(episode_data)
        print(f"Episode {episode + 1}: {step} steps")
    
    return pd.DataFrame(all_episodes_data)

def load_trained_agent(experiment_path):
    """Load the trained agent from experiment folder"""
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
    
    # Load checkpoint - FIXED to handle Tonic's automatic .pt addition
    checkpoint_path = os.path.join(experiment_path, 'checkpoints')
    if os.path.exists(checkpoint_path):
        checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith('step_') and f.endswith('.pt')]
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            # FIX: Remove .pt extension before passing to agent.load() 
            # because Tonic automatically adds .pt
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

def theoretical_pendulum_response(time, theta_0, theta_dot_0, g=9.81, L=1.0, damping=0.1):
    """Generate theoretical pendulum response"""
    omega_0 = np.sqrt(g / L)
    omega_d = omega_0 * np.sqrt(1 - damping**2)
    
    A = theta_0
    B = (theta_dot_0 + damping * omega_0 * theta_0) / omega_d
    
    return np.exp(-damping * omega_0 * time) * (A * np.cos(omega_d * time) + B * np.sin(omega_d * time))

def plot_time_response_comparison(agent_data, save_plots=True):
    """Plot agent vs theoretical time response"""
    for episode_num in agent_data['episode'].unique():
        episode_data = agent_data[agent_data['episode'] == episode_num]
        
        theta_0 = episode_data['angle'].iloc[0]
        theta_dot_0 = episode_data['angular_velocity'].iloc[0]
        time = episode_data['time'].values
        
        theta_theoretical = theoretical_pendulum_response(time, theta_0, theta_dot_0)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Angle comparison
        ax1.plot(time, episode_data['angle'], 'b-', label='Agent', linewidth=2)
        ax1.plot(time, theta_theoretical, 'r--', label='Theoretical', linewidth=2)
        ax1.set_ylabel('Angle (rad)')
        ax1.set_title(f'Episode {episode_num}: Pendulum Angle vs Time')
        ax1.legend()
        ax1.grid(True)
        
        # Angular velocity
        ax2.plot(time, episode_data['angular_velocity'], 'b-', label='Agent', linewidth=2)
        theta_dot_theoretical = np.gradient(theta_theoretical, time)
        ax2.plot(time, theta_dot_theoretical, 'r--', label='Theoretical', linewidth=2)
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('Angular Velocity Comparison')
        ax2.legend()
        ax2.grid(True)
        
        # Actions
        ax3.plot(time, episode_data['action'], 'g-', label='Agent Action', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Torque (N·m)')
        ax3.set_title('Agent Control Actions')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'time_response_episode_{episode_num}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        rmse = np.sqrt(np.mean((episode_data['angle'] - theta_theoretical)**2))
        print(f"Episode {episode_num} RMSE: {rmse:.4f} rad")

if __name__ == "__main__":
    # Path to your experiment folder (without the 0 subfolder)
    experiment_path = os.path.join(PROJECT_ROOT, "Pendulum-v1", "DDPG-Pendulum-Quick")
    
    print("Experiment path:", experiment_path)
    print("Config exists?", os.path.exists(os.path.join(experiment_path, "config.yaml")))
    print("Checkpoints directory exists?", os.path.exists(os.path.join(experiment_path, "checkpoints")))
    
    print("\nLoading trained agent...")
    try:
        agent, environment = load_trained_agent(experiment_path)
        print("Agent and environment loaded successfully")
    except Exception as e:
        print(f"Error loading agent: {e}")
        exit(1)
    
    print("\nCollecting episode data...")
    agent_data = collect_episode_data(agent, environment, num_episodes=3)
    
    # Save data
    agent_data.to_csv(os.path.join(PROJECT_ROOT, "agent_time_response_data.csv"), index=False)
    print("Saved data to agent_time_response_data.csv")
    
    print("\nPlotting time response comparison...")
    plot_time_response_comparison(agent_data)
    
    # Print summary
    print("\n=== SUMMARY ===")
    for episode_num in agent_data['episode'].unique():
        episode_data = agent_data[agent_data['episode'] == episode_num]
        print(f"Episode {episode_num}: {len(episode_data)} steps, "
              f"angle range: [{episode_data['angle'].min():.3f}, {episode_data['angle'].max():.3f}], "
              f"action range: [{episode_data['action'].min():.3f}, {episode_data['action'].max():.3f}]")

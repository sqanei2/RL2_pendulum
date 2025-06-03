import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load training log (this should exist)
file_path = "Pendulum-v1/DDPG-Pendulum-Quick/0/log.csv"
df = pd.read_csv(file_path)

# Plot 1: Learning Progress
plt.figure(figsize=(12, 8))

# Subplot 1: Episode scores over time
plt.subplot(2, 2, 1)
plt.plot(df['train/episodes'], df['train/episode_score/mean'], label='Agent Mean Score')
plt.axhline(y=0, color='r', linestyle='--', label='Theoretical Optimal')
plt.xlabel('Episodes')
plt.ylabel('Episode Score')
plt.title('Learning Progress: Agent vs Optimal')
plt.legend()
plt.grid(True)

# Subplot 2: Action statistics
plt.subplot(2, 2, 2)
plt.plot(df['train/episodes'], df['train/action/mean'], label='Mean Action')
plt.fill_between(df['train/episodes'], 
                 df['train/action/mean'] - df['train/action/std'],
                 df['train/action/mean'] + df['train/action/std'], 
                 alpha=0.3, label='±1 Std')
plt.xlabel('Episodes')
plt.ylabel('Action (Torque)')
plt.title('Agent Action Statistics')
plt.legend()
plt.grid(True)

# Subplot 3: Episode length consistency
plt.subplot(2, 2, 3)
plt.plot(df['train/episodes'], df['train/episode_length/mean'], label='Episode Length')
plt.xlabel('Episodes')
plt.ylabel('Steps per Episode')
plt.title('Episode Length Consistency')
plt.legend()
plt.grid(True)

# Subplot 4: Reward distribution
plt.subplot(2, 2, 4)
plt.plot(df['train/episodes'], df['train/episode_score/max'], label='Best Score')
plt.plot(df['train/episodes'], df['train/episode_score/min'], label='Worst Score')
plt.xlabel('Episodes')
plt.ylabel('Episode Score')
plt.title('Score Range per Episode')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print summary
print(f"Training Episodes: {df['train/episodes'].iloc[-1]}")
print(f"Final Mean Score: {df['train/episode_score/mean'].iloc[-1]:.2f}")
print(f"Best Score Achieved: {df['train/episode_score/max'].max():.2f}")

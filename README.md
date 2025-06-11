To run the training:
  ** cd "D:\your_full_path\RL2_pendulum\tonic (powershell terminal)
  python -m tonic.train --header "import tonic.torch; import tonic.environments; import gym" --agent "tonic.torch.agents.DDPG()" --environment "tonic.environments.Gym('Pendulum-v1',max_episode_steps=50)" --name "DDPG-Pendulum-Quick" --seed 0 --parallel 1 --sequential 20
**

To play with the training, I have created a .py file outside the tonic ('cause it worked around the path specification drama duh!)
Run Analysis\**play_visual.py**
If you wanna change initial condition or the action space (max/min allowable torque) to to tonic\enviroments 
the trained data are stored in Pendulum-v1\DDPG-Pendulum-Quick\checkpoints\step_500000.pt

To plot the time response run the Analysis\**Physics_vs_Agent.py**












  

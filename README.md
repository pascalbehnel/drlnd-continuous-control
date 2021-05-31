# Reacher

The reacher environment is about learning from high dimensional state space, and performing actions from a continuous action space. It contains an arm that has two joints which can be rotated freely. The goal for the agent is, by interacting with the environment, to learn how to move the joints, sothat the tip of the arm stays inside a moving target in the shape of a ball.

![](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

For every step in the environment, if the tip of the arm is within it's target ball, the Agent receives 0.1 points, thus, the goal of the Agent is to accumulate as many points as possible.

The environment is considered solved, when the Agent has accumulated an average of 30 points over the span of 100 episodes. The herein proposed agent was able to solve that environment within 179 episodes.

## Repository Content

- Continuous_Control.ipynb
  Jupyter Notebook for running this project
- agent.py
  Python file that containing the classes 'Agent', 'ReplayBuffer', 'OUNoise'
- model.py
  Python file that containing the deep learning models for the 'Actor' and the 'Critic'
- Report.md
  A Markdown File containing a more detailed explanation on this solution

## Getting Started

1. Clone this repo

2. Download the environment from one of the links below. You need only select the environment that matches your operating system: [LINUX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip), [LINUX-NO-VIZ](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip), [MAC](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip), [WIN32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip) (thanks to Udacity for providing the links)

3. (Optional) create a conda environment
   ```
   conda create -n myenv python=3.6
   ```
4. Install dependencies

   ```
   conda activate myenv
   pip install -r requirements.txt
   ```
5. Start jupyter notebook. Make sure that it is at the root of this project
# Reacher

The reacher environment is about learning from high dimensional state space, and performing actions from a continuous action space. It contains an arm that has two joints which can be rotated freely. The goal for the agent is, by interacting with the environment, to learn how to move the joints, sothat the tip of the arm stays inside a moving target in the shape of a ball.

![](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

For every step in the environment, if the tip of the arm is within it's target ball, the Agent receives 0.1 points, thus, the goal of the Agent is to accumulate as many points as possible.

The environment is considered solved, when the Agent has accumulated an average of 30 points over the span of 100 episodes. The herein proposed agent was able to solve that environment within 178 episodes.

# Learning Algorithm
The algorithm used to solve this environment is called Deep Deterministic Policy Gradient (ddpg) and is based on this [paper](https://arxiv.org/pdf/1509.02971.pdf) by Lillicrap et al. 

Since this is an Actor Critic approach, we have two models:
1. Actor - proposes an action given a specific state
2. Critic - tries to evaluate the expected return for a given action within a specific state

In addition to this, just like in the DQN approach, the authors of the paper propose to try and detach the learned models (in this case the Actor and Critic models) from the models that are used to evaluate the expected return. This results in both Actor and Critic models being duplicated, with one iteration of the models being called the 'local' models, and the others called the 'target' models. 

The local models are trained just like you would normally expect, but the target models are trained on a much slower pace, as they only get updated by copying a usually small part of the local model over to the target model, with the target model "lacking behind" as a result.

Another addition to the actor critic approach is to add a replay memory to the algorithm. The replay memory stores the experiences that the agent makes, and during the learning step, the algorithm picks a certain number of random experiences from the buffer, and only learns from them. The advantage to this is that we are not learning in the order that we gathered the experiences, which would most likely be biased.

In order to solve the 'explore / exploit' issue, the authors of the paper also introduced adding a noise vector to the action vector. They propose to use the Ornstein Uhlenbeck Noise method for this, which is what is also realized in this implementation.


# Model
Fully connected layers are used for both actor and critic networks. 

![](https://github.com/pascalbehnel/drlnd-continuous-control/blob/main/model_layout.PNG?raw=true)



# Hyperparameters

| Parameter   | Value  | Description                                  |
| ----------- | ------ | -------------------------------------------- |
| BATCH_SIZE  | 256    | How many experiences get pulled from Memory  |
| MEMORY_SIZE | 150000 | How many experiences can get saved in Memory |
| GAMMA       | 0.9    | Discount factor                              |
| TAU         | 1e-3   | Soft update factor for target parameters     |
| RATE_ACTOR  | 1e-3   | learning rate for the actor                  |
| RATE_CRITIC | 1e-3   | learning rate for the critic                 |
| SIGMA       | 0.02   | standard deviation for OUNoise               |

# Performance
The algorithm was able to solve the environment within 178 epochs. The environment is considered solved, if the Agent is able to obtain an average of 30 points over the span of 100 epochs.

![](https://github.com/pascalbehnel/drlnd-continuous-control/blob/main/performance.png?raw=true)



# Future Work

In order to decrease the number of epochs needed to solve the environment, some optimizations might help:

- Adapt number of layers. As of now i haven't played around much with this, but decreasing the number of layers in the Actor / Critic models might speed up the learning process.
- Trying to solve the 20 Agent environment. Maybe having 20 Agents training in parallel helps the network to converge.
- As of now, SIGMA is a constant. Instead, defining SIGMA in a more dynamic way, f.e. it being bigger in the beginning, and then decaying over time, might also help the network to converge, since we would like to explore more in the beginning, and less once we know more and more of the environment.


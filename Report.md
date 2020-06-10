# DeepRL-Collaboration-and-Competition
Project 3 "Collaboration and Competition" of the Deep Reinforcement Learning nanodegree.

Given the uncertainities for this project to be reviewed by the same reviewer of the last projects, I am describing my background for one last time. Basically, I am a civil engineer from Bangladesh who doesn't know civil engineering much let alone RL. RL is like Hebrew, not only RL the whole CS things seem like Hebrew to me- top to bottom I don't understand anything. I did only a Python course in my undergrad. That's it. Yet I tried this Nanodegree. Why? Firstly, I got one month free access to Udacity which I cannot afford otherwise. Secondly, I got an admission in an Erasmus Joint Master Degree Program of Hydroinformatics in Europe. My program arranges a conference named 'Symhydro'. The accepted papers of that conference get published in collarboration with Springer Water in a journal namely 'Advances in Hydroinformatics'. I read one of the papers from there- 'Large Markov Decision Processes Based Management Strategy of Inland Waterways in Uncertain Context.' I read the paper and didn't understand anything. That's how I got interest in RL and took the resolution to decode RL, though in reality I didn't understand many things while reading and doing so.

Thanks to the solution notebooks by Udacity and also by the alumni of this Nanodegree. The solution codes I use here are managed through visiting their GitHub Repositories. Now, if I pass this nanodegree and get the certificate, I will be the happiest man in the world.

## Learning Algorithm

Given that successful experience of project 2 https://github.com/Kamrujjaman-Rabin/Udacity-DeepRL-Project-2-Continuous_Control, I decided to reuse the code and hyperparameters of my DDPG Agent in order to build my new MADDPG Agent for Project 3.


### DDPG implementation

The Project goes beyond DQN. Because it includes new Deep RL techniques:
- **Actor-critic method** in which the actor computes policies to act and the critic helps to correct the policies based on its Q-values;
- **Deep Deterministic Policy Gradients (DDPG)**, which is similar to actor-critic methods but it differs because the actor produces a deterministic policy instead of stochastic policies; the critic evaluates such deterministic policy; and the actor is trained by using the deterministic policy gradient algorithm;
- **Two sets of Target and Local Networks**, which is a way to implement the double buffer technique in order to avoid oscillations caused by overestimated values;
- **Soft Updates** instead of hard updates so that the values of the local networks are slowly transferred to the target networks;
- **Replay Buffer** in order to keep training the DDPG Agent with past experiences;
- **Ornstein-Uhlenbeck(O-U) Noise** which is introduced at training in order to make the network learn in a more robust and more complete way.

Moreover, the DDPG Agent uses 2 deep neural networks to represent complex continuous states. 1 neural network for the actor and 1 neural network for the critic.

The neural network for the actor has:
- A linear fully-connected layer of dimensions state_size=`state_size` and fc1_units=128;
- The ReLu function;
- Batch normalization;
- A linear fully-connected layer of dimensions fc1_units=128 and fc2_units=128;
- The ReLu function;
- A linear fully-connected layer of dimensions fc2_units=128 and action_size=`action_size`;
- The tanh function.

The neural network for the critic has:
- A linear fully-connected layer of dimensions state_size=`state_size` and fcs1_units=128;
- The ReLu function;
- Batch normalization;
- A linear fully-connected layer of dimensions fcs1_units=128 + `action_size` and fc2_units=128;
- The ReLu function;
- A linear fully-connected layer of dimensions fc2_units=128 and output_size=1;

This implementation has the following metaparameters:

```
# replay buffer size (a very big database of SARS tuples)
BUFFER_SIZE = int(1e5)  

# minibatch size (the number of experience tuples per training iteration)
BATCH_SIZE = 128        

# discount factor (the Q-Network is aware of the intermediate future, but not the far future)
GAMMA = 0.99            

# for soft update of target parameters 
TAU = 1e-3           

# learning rate of the actor 
LR_ACTOR = 2e-4         

# learning rate of the critic 
LR_CRITIC = 2e-4        

# L2 weight decay
WEIGHT_DECAY = 0        
```


## Plot of Rewards

The DDPG Agents were trained for `775` episodes. In each episode, the agents are trained from the begining to the end of the simulation. Some episodes are larger and some episodes are shorter, depending when the ending condition of each episode appears. Each episode has many iterations. In each iteration, the DDPG Agents are trained with `BATCH_SIZE=128` experience tuples (SARS).

```
Episode 100	Average Score: 0.0110
Episode 200	Average Score: 0.0060
Episode 300	Average Score: 0.0432
Episode 400	Average Score: 0.0523
Episode 500	Average Score: 0.0455
Episode 600	Average Score: 0.0664
Episode 700	Average Score: 0.1158
Episode 775	Average Score: 0.5023
Environment solved in 775 episodes!	Average Score: 0.5023
```

The rubric asks to obtain an average score of 0.5 or more for 100 episodes. The best model was saved. In the graph, the blue lines connect the scores in each episode. Whereas the red lines connect the average scores in each episode. **The problem was solved in just 775 episodes, which is awesome!**

![Plot of rewards (training)](/images/plot-of-rewards-training.png)

After training, the saved model was loaded and tested for 50 episodes. Here are the results of such testing. You can see that, on average, the scores are greater than 0.5. 

```
Episode 1	Score: 1.1000
Episode 2	Score: 2.6000
Episode 3	Score: 0.6000
Episode 4	Score: 0.1000
Episode 5	Score: 0.0000
Episode 6	Score: 1.0000
Episode 7	Score: 1.4000
Episode 8	Score: 0.6000
Episode 9	Score: 0.1000
Episode 10	Score: 0.4000
Episode 11	Score: 0.2000
Episode 12	Score: 0.3000
Episode 13	Score: 0.1000
Episode 14	Score: 1.4000
Episode 15	Score: 2.6000
Episode 16	Score: 2.1000
Episode 17	Score: 2.3900
Episode 18	Score: 0.3000
Episode 19	Score: 0.5000
Episode 20	Score: 0.2000
Episode 21	Score: 0.1000
Episode 22	Score: 0.4000
Episode 23	Score: 0.1000
Episode 24	Score: 2.3000
Episode 25	Score: 2.6000
Episode 26	Score: 2.7000
Episode 27	Score: 0.3000
Episode 28	Score: 2.6000
Episode 29	Score: 0.9900
Episode 30	Score: 1.6000
Episode 31	Score: 0.1000
Episode 32	Score: 1.7000
Episode 33	Score: 1.6900
Episode 34	Score: 1.9900
Episode 35	Score: 2.6000
Episode 36	Score: 1.3000
Episode 37	Score: 0.4000
Episode 38	Score: 1.7000
Episode 39	Score: 2.2000
Episode 40	Score: 1.3000
Episode 41	Score: 2.6000
Episode 42	Score: 1.8000
Episode 43	Score: 1.4000
Episode 44	Score: 1.8000
Episode 45	Score: 1.9000
Episode 46	Score: 2.6000
Episode 47	Score: 2.7000
Episode 48	Score: 2.6000
Episode 49	Score: 0.2000
Episode 50	Score: 2.6000
```

In the graph, the blue lines connect the scores in each episode. The red horizontal line represents the value of 0.5. Notice that the big majority of points are greater than the red horizontal line of 0.5.

![Plot of rewards (testing)](/images/plot-of-rewards-testing.png)

## Ideas for Future Work

Since MADDPG is based on multiple DDPG Agents, I will repeat the recommendations I wrote for my Project 2:
https://github.com/Kamrujjaman-Rabin/Udacity-DeepRL-Project-2-Continuous_Control/blob/master/Report.md#ideas-for-future-work

So far, this implementation has only `5` techniques: 
- Deep Deterministic Policy Gradients (DDPG);
- Two sets of Target and Local Networks;
- Soft Updates;
- Replay Buffer;
- Ornstein-Uhlenbeck(O-U) Noise.

Future implementations can be improved by applying the following techniques:
- **Prioritized** experience replay;
- Distributed learning with multiple independent agents (TRPO, PPO, A3C, and A2C);
- Q-prop algorithm, which combines both off-policy and on-policy learning.

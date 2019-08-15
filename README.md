# physics\_learning\_rl

Reinforcement learning agent that actively learns physical motion mechanisms in a 2D simulated domain. Sample videos of trained agent episodes are accessable in `./videos/`.

Previous research paper: https://psyarxiv.com/6vr4g/


### Training
To train an active learning agent with q-value function approximator in separate framework, simply execute the following command with dependencies ready:
```
$ python learning_system.py --save_model True > exp_log/active_training_log.txt
```
or use *active_learning_bash.sh* to do the same thing.

### Branches:

* **master** & **loss_reward**:    Separate training framework. Train active agent & predictor separately, reward agent for predictor's **mean evaluation loss over 5 frames** during each action (5 frames). Then generate new training data for model predicor with trained active agent. 

* **loss_reduction**:   Concurrent training framework. Train active agent & predictor together, reward agent for predictor's **mean of training loss reduction over 5 frames** during each action (5 frames).


Model predictor was pretrained on human experiment data with `weighted average loss` and `learning rate = 1e-05` for 20 epochs.

Active agent was pretrained with reward only for catching & approaching pucks for 10000 episodes.


## Package dependency:

python 3.7+

tensorflow-gpu 1.13.1+

pyduktape 0.0.6


### File structures (master branch):

* **learning_system.py**
An integreated launch script that runs *separate training framework* in loop. *active_learning_bash.sh* does the same thing but run scripts one by one.

* **RQN_agent.py**
The active agent based on Recurrent Q-Network, includes training and testing methods.

* **random_agent.py**
Baseline agent, samples random actions.

* **interaction_env.py**
The interaction enviroment, defines action space and reward signals, communicate with JavaScript simulator.

* **config.py**
Constant parameters define enviroment and model settings.

* **js_simulator/**
JavaScript simualtor, contains data generation and environment settings.

* **model_predictor/**
The world predictor based on LSTM, includes training and testing methods.

* **model_predictor/video_generation.py**
Use `moviepy.editor` to generate videos with recorded trials data.

* **exp_log/ & model_predictor/exp_log/**
Training and testing logs of the active agent and world predictor

* **checkpoints/ & model_predictor/checkpoints/**
Checkpoints of pre-trained active agent and world predictor

# physics\_learning\_rl

Reinforcement learning agent that actively learns physical properties (motion mechanisms) in a 2D simulated domain.

### Branches:

* **master**:   train active agent & predictor together, reward agent for predictor's **change of mean training loss over 5 frames** during each action (5 frames).

* **loss_reward**:    train active agent & predictor separately, reward agent for predictor's **mean evaluation loss over 5 frames** during each action (5 frames). Then generate new training data for model predicor with trained active agent. 

Model predictor was pretrained on human experiment data with `weighted average loss` and `learning rate = 1e-05` for 20 epochs.

## Package dependency:

python 3.7+

tensorflow-gpu 1.13.1+

pyduktape 0.0.6

Previous research paper: https://psyarxiv.com/6vr4g/

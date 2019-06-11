---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: PPO2
summary: A simple implementation of the PPO2 model. Hidden state is computed using 2 MLP with Tanh activation function 
image: ppo_model.png
author: andompesta
tags: [rl]
github-link: https://github.com/andompesta/ppo2
accelerator: "cuda"

---
```python
import torch
model = torch.hub.load('andompesta/ppo2', 'ppo2', reset_param=True, force_reload=True, input_dim=4, hidden_dim=64, action_space=2, dropout=0)
```
<!-- Walkthrough a small example of using your model. Ideally, less than 25 lines of code -->

### Model Description
In reinforcement learning, policy optimization refer to the set of models that directly optimise the policy's parameters.
In this implementation we use the same latent state representation to compute the actions (trough a policy_head) and to estimate the value function (value_head).

To compute the action given an observation of the environment, we need to specify the action distributions. The default distribution is a Categorical suited for discrete action spaces.
Note that only ```torch.distributions.distribution.Distribution``` are accepted.
Moreover, the model use two fully connected layer with ```torch.Tanh``` activation function and dropout to compute the latent state given an observation of the environment.

In order to collect experience, it is possible to access the policy using the following function:
```python
def eval_fn(obs):
    """
    evaluation function. Choose an action based on the current policy
    :param obs: environment observation
    :return:
    """
    self.model.eval()
    with torch.set_grad_enabled(False):
        obs = torch.tensor(obs).float().to(device)
        value_f, action, neg_log_prob, entropy = model(obs)
        return value_f, action, neg_log_prob
``` 
Note that the input is an numpy array, while the outputs are three torch tensors (with no grad function):
1. estimated value function
2. action chosen by the policy
3. negative log-likelihood of the policy.

Training the model requires additional information:
1. estimate advantages function for each step taken
2. estimate the value function and the negative log-likelihood with the new policy parametrisation (the choose action does not change)
3. compute the surrogate loss. Note that args contains all the hyper-parameters used by the model:
    - args.clip_range = 0.2 (Clip value for the policy)
    - args.ent_coef = 0. (Entropy discount factor: we don't optimize the entropy of the policy)
    - args.vf_coef = 0.5 (Value function discount factor)
    - args.max_grad_norm = 0.5 (Maximum norm of the gradients)
```python
def train_fn(obs, returns, old_actions, old_values, old_neg_log_prbs):

    with torch.set_grad_enabled(False):
        advantages = returns - old_values
        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    model.train()
    with torch.set_grad_enabled(True):
        model.zero_grad()

        value_f, actions, neg_log_probs, entropy = model(obs, action=old_actions)

        assert(actions.sum().item() == old_actions.sum().item())

        loss, pg_loss, value_loss, entropy_mean, approx_kl = model.loss(returns, value_f, neg_log_probs, entropy, advantages,
                                                                           old_values, old_neg_log_prbs,
                                                                           args.clip_range, args.ent_coef, args.vf_coef)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
```
As output, we got:
 1. the total loss optimised
 2. the policy loss
 3. the value loss
 4. the mean entropy of our policy
 5. approximated KL-divergence between the old and new policy


### References
[Original implementation](https://github.com/openai/baselines/tree/master/baselines/ppo2)

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
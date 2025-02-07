# pylint: disable-all

"""
Original PPO implementation by Nikhil Barhate
Source: https://github.com/nikhilbarhate99/PPO-PyTorch
Changes made are:
-  Minor logging additions have been mde
-  Minor addition of gradient clipping (to help prevent help prevent exploding gradients)
-  If an MCTS action is selected as compared to PPO's action (during training), then
   an imitation loss is added to help PPO learn from MCTS
"""
"""
MIT License

Copyright (c) 2018 Nikhil Barhate

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from src.logger.logger import logger

################################## set device ##################################
logger.info(
    "============================================================================================"
)
# set device to cpu or cuda
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    logger.info("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    logger.info("Device set to : cpu")
logger.info(
    "============================================================================================"
)


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.mcts_actions = []
        self.is_mcts_better = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.mcts_actions[:]
        del self.is_mcts_better[:]


class ActorCritic(nn.Module):
    def __init__(
        self, state_dim, action_dim, has_continuous_action_space, action_std_init
    ):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init
            ).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh(),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1),
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,), new_action_std * new_action_std
            ).to(device)
        else:
            logger.info(
                "--------------------------------------------------------------------------------------------"
            )
            logger.warning(
                "Calling ActorCritic::set_action_std() on discrete action space policy"
            )
            logger.info(
                "--------------------------------------------------------------------------------------------"
            )

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    DUMMY_ACTION = -1

    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
        imitation_loss_weight=0,
    ):
        self.imitation_loss_weight = imitation_loss_weight
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init
        ).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ],
            eps=1e-5,
        )

        self.policy_old = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.nll_loss = nn.NLLLoss(reduction="none")

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            logger.info(
                "--------------------------------------------------------------------------------------------"
            )
            logger.warning(
                "Calling PPO::set_action_std() on discrete action space policy"
            )
            logger.info(
                "--------------------------------------------------------------------------------------------"
            )

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        logger.info(
            "--------------------------------------------------------------------------------------------"
        )
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                logger.info(
                    "setting actor output action_std to min_action_std : ",
                    self.action_std,
                )
            else:
                logger.info("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            logger.warning(
                "Calling PPO::decay_action_std() on discrete action space policy"
            )
        logger.info(
            "--------------------------------------------------------------------------------------------"
        )

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def select_action_pretrained(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
            return action.item(), action_logprob, state_val

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert lists to tensors

        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(device)
        )
        is_mcts_better = torch.tensor(self.buffer.is_mcts_better, dtype=torch.bool).to(
            device
        )
        mcts_actions_list = self.buffer.mcts_actions  # List, may contain None

        # Prepare mcts_actions tensor
        mcts_actions = torch.tensor(mcts_actions_list, dtype=torch.long).to(device)

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            ppo_loss = -torch.min(surr1, surr2)

            # Value function loss
            value_loss = 0.5 * self.MseLoss(state_values, rewards)

            # Entropy loss
            entropy_loss = -0.01 * dist_entropy
            # Initialize imitation loss as zeros (per-sample loss)
            imitation_loss = torch.zeros_like(ppo_loss).to(device)
            # Compute imitation loss for better MCTS actions
            if is_mcts_better.any():
                indices = is_mcts_better.nonzero(as_tuple=True)[0]

                # Get corresponding action logits and MCTS actions
                mcts_action_probs = self.policy.actor(old_states[indices])
                mcts_actions_better = mcts_actions[indices]

                # Exclude dummy actions (-1)
                valid_indices = mcts_actions_better != PPO.DUMMY_ACTION
                mcts_action_probs = mcts_action_probs[valid_indices]
                mcts_actions_better = mcts_actions_better[valid_indices]

                # Compute log probabilities
                # Add a small value to prevent taking log(0)

                # Compute imitation loss using log probabilities
                if mcts_actions_better.numel() > 0:
                    log_probs = torch.log(mcts_action_probs + 1e-10)
                    mcts_actions_better = mcts_actions_better.long()
                    imitation_losses = self.nll_loss(log_probs, mcts_actions_better)
                    # Assign imitation losses back to the appropriate positions
                    imitation_loss[indices] = imitation_losses
                else:
                    pass  # imitation_loss remains zeros where MCTS is not better

            else:
                pass  # imitation_loss remains zeros

            # Total loss
            loss = (
                ppo_loss
                + value_loss
                + entropy_loss
                + self.imitation_loss_weight * imitation_loss
            )
            # print("ppo loss========================== ", ppo_loss)
            # print("value loss======================== ", value_loss)
            # print("entropy loss====================== ", entropy_loss)
            # print(
            #     "imitation loss==================== ", weighing_factor * imitation_loss
            # )

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )

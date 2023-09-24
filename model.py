import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import vtrace

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], 1)


class conv_net(nn.Module):
    def __init__(self, observation_dim):
        super(conv_net, self).__init__()
        self.observation_dim = observation_dim
        self.net = nn.Sequential(
            nn.Conv2d(self.observation_dim[0], 8, 4, 2),
            CReLU(),
            nn.Conv2d(16, 8, 5, 1),
            CReLU(),
            nn.Conv2d(16, 8, 3, 1),
            CReLU()
        )

    def forward(self, observation):
        observation = observation / 255.
        x = self.net(observation).view(observation.size(0), -1)
        return x

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.state_dim = state_dim

        self.conv_net = conv_net(self.state_dim)

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            conv_net(self.state_dim),
                            nn.Tanh(),
                            nn.Linear(self.conv_dim(), 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            conv_net(self.state_dim),
                            nn.ReLU(),
                            nn.Linear(self.conv_dim(), 32),
                            nn.ReLU(),
                            nn.Linear(32, action_dim),
                        )
        # critic
        self.critic = nn.Sequential(
                        conv_net(self.state_dim),
                        nn.ReLU(),
                        nn.Linear(self.conv_dim(), 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def conv_dim(self):
        return self.conv_net(torch.zeros([1, * self.state_dim])).view(1, -1).size(-1)

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_logits = self.actor(state)
            action_probs = F.softmax(action_logits, -1)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = F.log_softmax(action_logits, dim=-1)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state):
        action_logits = None
        action_probs = None
        seq_len = state.size(0)
        batch_size = state.size(1)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_logits = self.actor(state.flatten(0, 1))
            action_logits = action_logits.view(seq_len, batch_size, -1)
            action_probs = F.softmax(action_logits, -1)
            dist = Categorical(action_probs)
        action_logprobs = torch.log_softmax(action_logits, dim=-1)
        dist_entropy = dist.entropy()
        state_values = self.critic(state.flatten(0, 1))
        state_values = state_values.view(seq_len, batch_size, -1)

        return action_logits, action_logprobs, action_probs, state_values, dist_entropy


class Nash:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.epsilon_threshold = 0.03
        self.gamma_averaging = 0.01
        self.roh_bar = 1
        self.c_bar = 1
        self.vtrace_gamma = 1
        self.eta = 0.5
        self.neurd_clip = 1000
        self.beta = 2
        self.value_weight = 1
        self.neurd_weight = 1
        self.grad_clip = 1000

        self.buffer = [RolloutBuffer(), RolloutBuffer()]

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_reg = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_reg.load_state_dict(self.policy.state_dict())
        self.policy_reg_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_reg_old.load_state_dict(self.policy.state_dict())

        self.action_dim = action_dim

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling Nash::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling Nash::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, id, test=False):
        state = np.expand_dims(state, 0)
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy.act(state)

            if test == False:
                self.buffer[id].states[-1].append(state)
                self.buffer[id].actions[-1].append(action)
                self.buffer[id].logprobs[-1].append(action_logprob)
                self.buffer[id].state_values[-1].append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy.act(state)

            if test == False:
                self.buffer[id].states[-1].append(state)
                self.buffer[id].actions[-1].append(action)
                self.buffer[id].logprobs[-1].append(action_logprob)
                self.buffer[id].state_values[-1].append(state_val)

            return action.item()

    def update(self, alpha):

        old_states = []
        old_actions = []
        old_logprobs = []
        old_state_values = []
        rewards = []
        for t in range(len(self.buffer[0].states)):
            old_states.append(torch.cat([torch.stack(self.buffer[0].states[t], dim=0), torch.stack(self.buffer[1].states[t], dim=0)], dim=1))
            old_actions.append(torch.cat([torch.stack(self.buffer[0].actions[t], dim=0), torch.stack(self.buffer[1].actions[t], dim=0)], dim=1))
            old_logprobs.append(torch.cat([torch.stack(self.buffer[0].logprobs[t], dim=0), torch.stack(self.buffer[1].logprobs[t], dim=0)], dim=1))
            old_state_values.append(torch.cat([torch.stack(self.buffer[0].state_values[t], dim=0), torch.stack(self.buffer[1].state_values[t], dim=0)], dim=1))
            rewards.append(torch.FloatTensor(self.buffer[0].rewards[t]))

        seq_len = len(old_states[0])
        old_states = torch.stack(old_states, dim=2)
        old_states = old_states.view(seq_len * 2, * old_states.size()[2:])
        old_actions = torch.stack(old_actions, dim=2)
        old_actions = old_actions.view(seq_len * 2, * old_actions.size()[2:])
        old_actions_oh = F.one_hot(old_actions, 10)
        old_logprobs = torch.stack(old_logprobs, dim=2)
        old_logprobs = old_logprobs.view(seq_len * 2, * old_logprobs.size()[2:])
        old_state_values = torch.stack(old_state_values, dim=2)
        old_state_values = old_state_values.view(seq_len * 2, * old_state_values.size()[2:])
        rewards = torch.stack(rewards, dim=1)
        rewards = rewards.view(seq_len, len(self.buffer[0].states), * rewards.size()[2:]).to(old_states.device)
        player_id = torch.ones(seq_len * 2, len(self.buffer[0].states)).to(old_states.device)
        player_id[::2] = 0
        valid = torch.ones(seq_len * 2, len(self.buffer[0].states)).to(old_states.device)
        masks = torch.ones(seq_len * 2, len(self.buffer[0].states), self.action_dim).to(old_states.device)

        logit, log_pi, probs, state_v, dist_entropy = self.policy.evaluate(old_states)
        pi_processed = vtrace.process_policy(probs, masks, self.action_dim, self.epsilon_threshold)
        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

        with torch.no_grad():
            _, _, probs_old, state_v_old, _ = self.policy_old.evaluate(old_states)
            _, log_pi_reg, _, _, _ = self.policy_reg.evaluate(old_states)
            _, log_pi_reg_old, _, _, _ = self.policy_reg_old.evaluate(old_states)

            log_policy_reg = log_pi - (alpha * log_pi_reg + (1 - alpha) * log_pi_reg_old)

        for player in range(2):
            reward = rewards * ((-1) ** player)
            reward = reward.repeat([1, 2]).view(seq_len * 2, -1)
            reward = reward * (1 - player_id)
            v_target_, has_played, policy_target_ = vtrace.v_trace(
                state_v_old,
                valid,
                player_id,
                torch.exp(old_logprobs),
                pi_processed,
                log_policy_reg,
                vtrace._player_others(player_id, valid, player),
                old_actions_oh,
                reward,
                player,
                lambda_=1.0,
                c=self.c_bar,
                rho=self.roh_bar,
                eta=self.eta,
                gamma=self.vtrace_gamma,
            )

            v_target_list.append(v_target_)
            has_played_list.append(has_played)
            v_trace_policy_target_list.append(policy_target_)
        loss_v = vtrace.get_loss_v([state_v] * 2, v_target_list, has_played_list)

        is_vector = torch.unsqueeze(torch.ones_like(valid), dim=-1)
        importance_sampling_correction = [is_vector] * 2

        loss_nerd = vtrace.get_loss_nerd(
            [logit] * 2,
            [pi_processed] * 2,
            v_trace_policy_target_list,
            valid,
            player_id,
            masks,
            importance_sampling_correction,
            clip=self.neurd_clip,
            threshold=self.beta,
        )

        self.optimizer.zero_grad()
        loss = self.value_weight * loss_v + self.neurd_weight * loss_nerd
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()
        params1 = self.policy.state_dict()
        params2 = self.policy_old.state_dict()
        for name1, param1 in params1.items():
            params2[name1].data.copy_(
                self.gamma_averaging * param1.data
                + (1 - self.gamma_averaging) * params2[name1].data
            )
        self.policy_old.load_state_dict(params2)

        # clear buffer
        for id in range(2):
            self.buffer[id].clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def new_buffer(self):
        for id in range(2):
            self.buffer[id].actions.append([])
            self.buffer[id].states.append([])
            self.buffer[id].logprobs.append([])
            self.buffer[id].rewards.append([])
            self.buffer[id].state_values.append([])
            self.buffer[id].is_terminals.append([])




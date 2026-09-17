"""
MP-DQN Agent for PTZ Camera Control.

Multi-Pass Parameterized Deep Q-Networks (Bester, James, Konidaris, 2019)
fixes P-DQN's false-gradient bias: in P-DQN, Q_k(s, x_1, ..., x_K) depends on
the entire concatenated parameter vector, so gradients w.r.t. x_j (j != k) leak
into Q_k. MP-DQN forwards K passes, each zeroing all params except x_k, so
Q_k only sees x_k.

This file replaces the previous PDQNAgent. Compared to it:
  - QActor now actually consumes action params (the old one accepted but ignored them).
  - The replay buffer stores action_param alongside the discrete action.
  - The param network's gradient flows through Q (which it did not before).
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QActor(nn.Module):
    """Q-network Q(s, x) -> K values, one per discrete action."""

    def __init__(self, state_dim, num_actions, param_size=1, hidden_layers=(256, 128, 64)):
        super().__init__()
        self.num_actions = num_actions
        self.param_size = param_size

        layers = []
        input_dim = state_dim + num_actions * param_size
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, num_actions)

    def forward(self, state, action_params):
        # state: (B, state_dim), action_params: (B, K*param_size)
        x = torch.cat([state, action_params], dim=-1)
        return self.head(self.trunk(x))


class ParamNet(nn.Module):
    """x(s) -> per-action continuous parameter, bounded to [0, 1] via sigmoid."""

    def __init__(self, state_dim, num_actions, param_size=1, hidden_layers=(256, 128, 64)):
        super().__init__()
        self.num_actions = num_actions
        self.param_size = param_size

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_actions * param_size))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        raw = self.network(state)
        return torch.sigmoid(raw)  # (B, K*param_size), each in [0, 1]


class ReplayBuffer:
    """Stores (s, a, x_a, r, s', done). x_a is the scalar param for the chosen action."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, action_param, reward, next_state, done):
        self.buffer.append((state, action, action_param, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, action_params, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(action_params, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class MPDQNAgent:
    """MP-DQN agent for hybrid (discrete + continuous-param) action spaces."""

    def __init__(
        self,
        state_dim=16,
        num_actions=5,
        param_size=1,
        hidden_layers=(256, 128, 64),
        learning_rate=1e-4,
        param_lr=1e-5,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=5000,
        batch_size=256,
        replay_memory_size=100000,
        target_update_freq=100,
        grad_clip=10.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.param_size = param_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        self.update_count = 0

        self.actor = QActor(state_dim, num_actions, param_size, hidden_layers).to(self.device)
        self.actor_target = QActor(state_dim, num_actions, param_size, hidden_layers).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.param_net = ParamNet(state_dim, num_actions, param_size, hidden_layers).to(self.device)
        self.param_net_target = ParamNet(state_dim, num_actions, param_size, hidden_layers).to(self.device)
        self.param_net_target.load_state_dict(self.param_net.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.param_optimizer = optim.Adam(self.param_net.parameters(), lr=param_lr)

        self.memory = ReplayBuffer(replay_memory_size)

        print(f"MP-DQN Agent initialized on {self.device}")
        print(f"State dim: {state_dim}, Num actions: {num_actions}, Param size: {param_size}")

    # ------------------------------------------------------------------ helpers

    def _multi_pass_q(self, actor, states, all_params):
        """Multi-pass Q evaluation.

        For each discrete action k, build a parameter vector that is zero
        everywhere except slot k (which holds the per-action param from
        `all_params`), forward through `actor`, and take Q_k from that pass.
        Returns a (B, K) tensor whose column k is Q(s, k, x_k) without
        contamination from other x_j.

        states:     (B, state_dim)
        all_params: (B, K * param_size)   -- per-action params, possibly with grad
        """
        B = states.size(0)
        K = self.num_actions
        P = self.param_size

        states_rep = states.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

        params_3d = all_params.view(B, K, P)
        masked = torch.zeros(B, K, K, P, device=states.device, dtype=all_params.dtype)
        idx = torch.arange(K, device=states.device)
        # masked[b, k, k, :] = params_3d[b, k, :]   (broadcasts grad through all_params)
        masked[:, idx, idx, :] = params_3d
        masked = masked.view(B * K, K * P)

        q_all = actor(states_rep, masked)        # (B*K, K)
        q_all = q_all.view(B, K, K)
        diag = q_all[:, idx, idx]                # (B, K) -> Q(s, k, x_k)
        return diag

    # ----------------------------------------------------------------- API

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
            action_param = float(np.random.uniform(0.2, 1.0))
            return action, action_param

        with torch.no_grad():
            state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            params = self.param_net(state_tensor)            # (1, K*P), already sigmoid'd
            q_diag = self._multi_pass_q(self.actor, state_tensor, params)  # (1, K)
            action = int(q_diag.argmax(dim=1).item())
            action_param = float(params.view(1, self.num_actions, self.param_size)[0, action, 0].item())
        return action, action_param

    def store_transition(self, state, action, action_param, reward, next_state, done):
        self.memory.push(state, action, action_param, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return {}

        states_np, actions_np, action_params_np, rewards_np, next_states_np, dones_np = \
            self.memory.sample(self.batch_size)

        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        action_params = torch.from_numpy(action_params_np).to(self.device)  # (B,) scalar param
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)

        # epsilon schedule (linear decay)
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 / self.epsilon_decay))

        B = states.size(0)
        K = self.num_actions
        P = self.param_size

        # ----- TD target with multi-pass on target nets -----
        with torch.no_grad():
            next_params = self.param_net_target(next_states)              # (B, K*P)
            q_next_diag = self._multi_pass_q(self.actor_target, next_states, next_params)  # (B, K)
            q_max_next = q_next_diag.max(dim=1)[0]                        # (B,)
            target = rewards + (1.0 - dones) * self.gamma * q_max_next

        # ----- Q-network loss using only the executed (a, x_a) per sample -----
        # Build a "masked" param vector that is zero except slot a = x_a.
        exec_params = torch.zeros(B, K, P, device=self.device)
        exec_params[torch.arange(B, device=self.device), actions, 0] = action_params
        exec_params = exec_params.view(B, K * P)

        q_pred_all = self.actor(states, exec_params)                       # (B, K)
        y_pred = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)     # (B,)
        loss_actor = F.smooth_l1_loss(y_pred, target)

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # ----- Param-net loss: maximize sum_k Q(s, k, x_k(s)) via multi-pass -----
        # Freeze actor params so the param optimizer only moves param_net.
        for p in self.actor.parameters():
            p.requires_grad_(False)

        params_pred = self.param_net(states)                               # (B, K*P) with grad
        q_diag = self._multi_pass_q(self.actor, states, params_pred)       # (B, K)
        param_loss = -q_diag.sum(dim=1).mean()

        self.param_optimizer.zero_grad()
        param_loss.backward()
        nn.utils.clip_grad_norm_(self.param_net.parameters(), self.grad_clip)
        self.param_optimizer.step()

        for p in self.actor.parameters():
            p.requires_grad_(True)

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.param_net_target.load_state_dict(self.param_net.state_dict())

        return {
            'loss_actor': float(loss_actor.item()),
            'loss_param': float(param_loss.item()),
            'epsilon': self.epsilon,
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'param_net': self.param_net.state_dict(),
            'param_net_target': self.param_net_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'param_optimizer': self.param_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.param_net.load_state_dict(ckpt['param_net'])
        self.param_net_target.load_state_dict(ckpt['param_net_target'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.param_optimizer.load_state_dict(ckpt['param_optimizer'])
        print(f"Model loaded from {path}")


if __name__ == '__main__':
    agent = MPDQNAgent(state_dim=16, num_actions=5, batch_size=8, device='cpu')
    rng = np.random.default_rng(0)
    state = rng.standard_normal(16).astype(np.float32)
    action, param = agent.choose_action(state)
    print(f"Chosen action: {action}, param: {param:.3f}")

    for _ in range(64):
        s = rng.standard_normal(16).astype(np.float32)
        s2 = rng.standard_normal(16).astype(np.float32)
        a = int(rng.integers(0, 5))
        x = float(rng.uniform(0.0, 1.0))
        agent.store_transition(s, a, x, float(rng.standard_normal()), s2, False)

    stats = agent.update()
    print(f"Update stats: {stats}")
    print("MP-DQN smoke test complete")

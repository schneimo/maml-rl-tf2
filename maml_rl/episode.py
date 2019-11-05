import numpy as np

import tensorflow as tf


class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95):
        self.batch_size = batch_size
        self.gamma = gamma

        self._observations_list = [[] for _ in range(batch_size)]
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]
        self._mask_list = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None

    @property
    def observations(self):
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size) + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._observations_list[i])
                observations[:length, i] = np.stack(self._observations_list[i], axis=0)
            self._observations = observations
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                               + action_shape, dtype=self._actions_list[0][0].dtype)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                actions[:length, i] = np.stack(self._actions_list[i], axis=0)
            self._actions = actions
        return self._actions

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rewards_list[i])
                rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
            self._rewards = rewards
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards#.cpu().numpy()
            mask = self.mask#.cpu().numpy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i]
                returns[i] = return_
            self._returns = returns
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                mask[:length, i] = 1.0
            self._mask = mask
        return self._mask

    def gae(self, values, tau=1.0):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = tf.squeeze(values, axis=2)  # .detach() # TODO: Maybe stop_gradient instead of detach
        # Padding with (0, 0, 0, 1) means in PyTorch to pad the first dimension with 1
        values = tf.pad(values * self.mask, [[0, 1], [0, 0]])

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        advantages = tf.TensorArray(tf.float32, *deltas.shape)
        gae = tf.zeros_like(deltas[0], dtype=tf.float32)

        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * tau + deltas[i]
            advantages = advantages.write(i, gae)
        advantages = advantages.stack()
        # tf.reshape(advantages, shape=(1, advantages.shape[-1]))
        return advantages

    def append(self, observations, actions, rewards, batch_ids):
        for observation, action, reward, batch_id in zip(
                observations, actions, rewards, batch_ids):
            if batch_id is None:
                continue
            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(action.dtype)) #action.astype(np.float32)
            self._rewards_list[batch_id].append(reward.astype(np.float32))

    def __len__(self):
        return max(map(len, self._rewards_list))

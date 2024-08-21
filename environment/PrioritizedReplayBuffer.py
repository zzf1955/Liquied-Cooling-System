import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import RolloutBufferSamples

# 实现优先经验回放的 ReplayBuffer
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, alpha=0.6, beta=0.4, epsilon=1e-6, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, observation_space, action_space, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, *args, **kwargs):
        super(PrioritizedReplayBuffer, self).add(*args, **kwargs)
        self.priorities[self.pos - 1] = self.max_priority  # 设置新样本为最大优先级

    def sample(self, batch_size, env=None):
        if self.full:
            probabilities = self.priorities ** self.alpha
        else:
            probabilities = self.priorities[:self.pos] ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(probabilities), batch_size, p=probabilities)
        samples = self._get_samples(indices, env)

        # 根据优先级调整样本权重
        weights = (len(self) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = np.abs(priorities) + self.epsilon
        self.max_priority = max(self.max_priority, priorities.max())

    def _get_samples(self, batch_inds, env):
        # 继承自 stable_baselines3 的 ReplayBuffer 的方法
        return super(PrioritizedReplayBuffer, self)._get_samples(batch_inds, env)
    
if __name__ == "__main__":
    PBF = PrioritizedReplayBuffer()

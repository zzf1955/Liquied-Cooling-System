#!/usr/bin/env python
"""
DSAC 预实验脚本 - 4组6电池小规模测试

修改内容：
1. 使用 DummyVectorEnv (更稳定)
2. 小规模: 4组 x 6电池
3. 减少训练轮数用于快速验证
"""

import os
import torch
import numpy as np
import argparse
import itertools
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from diffusion import Diffusion
from diffusion.model import MLP
from datetime import datetime
from BatteryEnv.multi_battery_env import make_env

module_name = "dsac_policy.pth"

# ============ 小规模预实验参数 ============
num_batteries_per_group = 6   # 原来4 -> 现在6
num_groups = 4                # 原来2 -> 现在4
episode_steps = 200           # 每轮步数

# 网络参数
actor_lr = 3e-4
critic_lr = 3e-4
actor_hidden_dims = [256, 256]
critic_hidden_dims = [256, 256]
diffusion_steps = 5
trait_dim = 256
seed = 42

# 训练参数
alpha = 0.05
tau = 0.005
buffer_size = 100000
epoch = 500          # 减少用于快速验证
step_per_epoch = 200
episode_per_collect = 1
episode_per_test = 1
repeat_per_collect = 1
update_per_step = 1
batch_size = 256     # 减小batch
gamma = 0.95
n_step = 3
training_num = 4      # 4个并行环境
test_num = 1

# 日志
logdir = f"log/pretest_{num_batteries_per_group}_{num_groups}/"
time_now = datetime.now().strftime('%b%d-%H%M%S')
log_path = os.path.join(logdir, 'dsac', str(time_now))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wd = 0.005

# ============ 创建环境 ============
# 使用 DummyVectorEnv (更稳定)
env, train_envs, test_envs = make_env(
    num_batteries_per_group=num_batteries_per_group,
    num_groups=num_groups,
    episode_steps=episode_steps,
    log_path=log_path,
    con=True,
    num_train_envs=training_num,
    num_test_envs=test_num,
    use_subproc=False,  # 使用 DummyVectorEnv
    debug=False,
)

print(f"环境创建完成: {num_groups}组 x {num_batteries_per_group}电池")
print(f"状态维度: {env.observation_space.shape}")
print(f"动作维度: {env.action_space.shape}")


# ============ 网络定义 ============
def create_actor(state_shape, action_shape):
    actor_net = MLP(
        state_dim=state_shape,
        action_dim=trait_dim,
        hidden_dim=actor_hidden_dims
    )
    actor = Diffusion(
        input_dim=state_shape,
        output_dim=trait_dim,
        model=actor_net,
        max_action=1.,
        n_timesteps=diffusion_steps
    ).to(device)
    actor_prob = ActorProb(
        preprocess_net=actor,
        action_shape=action_shape,
        unbounded=False,
        device=device,
        preprocess_net_output_dim=trait_dim
    ).to(device)
    actor_optim = torch.optim.Adam(
        actor_prob.parameters(),
        lr=actor_lr,
        weight_decay=wd
    )
    return actor_prob, actor_optim


def create_critic(state_shape, action_shape):
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=critic_hidden_dims,
        activation=nn.Mish,
        concat=True,
        device=device
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(
        critic1.parameters(),
        lr=critic_lr,
        weight_decay=wd
    )

    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=critic_hidden_dims,
        activation=nn.Mish,
        concat=True,
        device=device
    )
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(
        critic2.parameters(),
        lr=critic_lr,
        weight_decay=wd
    )

    return critic1, critic1_optim, critic2, critic2_optim


def main():
    # 获取状态和动作维度
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    state_shape = state_shape[0]
    action_shape = action_shape[0]
    print(f"state_dims: {state_shape}")
    print(f"action_dims: {action_shape}")

    # 创建网络
    actor, actor_optim = create_actor(state_shape, action_shape)
    critic1, critic1_optim, critic2, critic2_optim = create_critic(state_shape, action_shape)

    # 创建策略 - 使用位置参数
    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        estimation_step=n_step,
    )

    # 创建Buffer
    buffer = VectorReplayBuffer(
        total_size=buffer_size,
        buffer_num=training_num,
        ignore_obs_next=True,
        n_step=n_step,
    )

    # 创建Collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # 日志
    tb_log_path = os.path.join(log_path, 'tensorboard')
    writer = SummaryWriter(tb_log_path)
    logger = TensorboardLogger(writer, update_interval=1)

    # 训练函数
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, module_name))

    # 开始训练
    print("\n" + "="*50)
    print("开始训练! 查看日志: tensorboard --logdir=log/pretest_6_4/")
    print("="*50 + "\n")

    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=None,
        episode_per_collect=episode_per_collect,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=episode_per_test,
        batch_size=batch_size,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=update_per_step,
        test_in_train=False,
    )

    print(f"\n训练完成!")
    print(f"最佳测试 reward: {result['best_reward']}")

    # 保存最终模型
    torch.save(policy.state_dict(), os.path.join(log_path, 'final_policy.pth'))
    print(f"模型已保存到: {os.path.join(log_path, 'final_policy.pth')}")


if __name__ == '__main__':
    main()

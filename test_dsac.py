import os
import torch
import numpy as np
import pandas as pd # 引入 pandas 用于保存结果
from torch import nn
from tianshou.data import Collector, VectorReplayBuffer # 虽然测试不用buffer， 但可能make_env需要
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import SACPolicy
from diffusion import Diffusion
from diffusion.model import MLP
from BatteryEnv.muti_battery_env import make_env # 确保这个导入路径正确

# --- 从训练脚本复制的参数 ---
# 环境参数
num_batteries_per_group = 4
num_groups = 2
episode_steps = 200
env_temp = 298
change_steps = 128 # 确保这个参数与训练时一致

# 模型参数
actor_hidden_dims = [256, 256]
critic_hidden_dims = [256, 256]
diffusion_steps = 5
trait_dim = 256

# 策略参数 (加载后会被覆盖一部分，但gamma等需要)
alpha = 0.05 # 如果需要自动调整alpha，这里可以忽略
tau = 0.005
gamma = 0.95
n_step = 3

# --- 测试参数 ---
policy_path = "policy/dsac.pth" # 策略文件路径
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
test_num = 10 # 使用多少个并行的测试环境 (对应 test_envs 的数量)
evaluate_num = 5 # 运行多少次评估（每次评估会收集 test_num 个 episode）
result_save_path = "test_results.csv" # 保存结果的文件路径



# --- 从训练脚本复制的模型创建函数 ---
def create_actor(state_shape, action_shape):
    # Actor network
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
    return actor_prob

def create_critic(state_shape, action_shape):
    # Critic networks
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=critic_hidden_dims,
        activation=nn.Mish,
        concat=True,
        device=device
    )
    critic1 = Critic(net_c1, device=device).to(device)

    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=critic_hidden_dims,
        activation=nn.Mish,
        concat=True,
        device=device
    )
    critic2 = Critic(net_c2, device=device).to(device)
    return critic1, critic2

def main():
    # --- 环境设置 ---
    # 创建环境，注意 training_num 可以设为0或1，test_num 设为所需的并行测试环境数
    env, train_envs, test_envs = make_env(
        num_batteries_per_group=num_batteries_per_group,
        num_groups=num_groups,
        episode_steps=episode_steps,
        log_path="test_log/dsac/", # 测试时通常不需要日志
        con=True
    )

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    # 如果环境返回的是 Box 类型，state_shape可能是 (dim,) 的元组
    if isinstance(state_shape, tuple):
        state_shape = state_shape[0]
    if isinstance(action_shape, tuple):
        action_shape = action_shape[0]

    print("State shape:", state_shape)
    print("Action shape:", action_shape)
    print(f"Number of test environments: {test_num}")

    # --- 随机种子 ---
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Tianshou 的 make_env 通常会处理环境种子

    # --- 模型和策略实例化 ---
    actor = create_actor(state_shape, action_shape)
    critic1, critic2 = create_critic(state_shape, action_shape)

    # 测试时不需要优化器，所以传入 None
    policy = SACPolicy(
        actor=actor,
        actor_optim=None,
        critic1=critic1,
        critic1_optim=None,
        critic2=critic2,
        critic2_optim=None,
        tau=tau,
        gamma=gamma,
        alpha=alpha, # alpha 值会从加载的状态中恢复
        estimation_step=n_step,
        action_space=env.action_space # 传递 action_space
    )

    # --- 加载策略 ---
    print(f"Loading policy from: {policy_path}")
    try:
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        print("Policy loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Policy file not found at {policy_path}")
        return
    except Exception as e:
        print(f"Error loading policy state_dict: {e}")
        return

    # --- 创建测试 Collector ---
    test_collector = Collector(policy, test_envs)

    # --- 评估循环 ---
    print(f"Evaluating the loaded policy for {evaluate_num} trials...")
    policy.eval()  # 设置为评估模式
    # 如果 alpha 是可学习的，也设置为评估模式
    if hasattr(policy, 'log_alpha') and isinstance(policy.log_alpha, torch.Tensor):
         policy.log_alpha.requires_grad = False
    if hasattr(policy, 'alpha') and not isinstance(policy.alpha, float):
         policy.alpha = policy.alpha.item()

    all_results = []
    for i in range(evaluate_num):
        print(f"--- Evaluation Trial {i + 1}/{evaluate_num} ---")
        test_collector.reset() # 重置收集器和环境
        # 收集 test_num 个 episode 的数据 (每个并行环境跑一个 episode)
        result = test_collector.collect(n_episode=test_num, render=False)
        all_results.append(result)
        print(f"Trial {i + 1} Results: {result}")
        # 如果您的环境有类似 get_stc 的方法来保存额外数据，可以在这里调用
        # 例如: test_envs.workers[0].env.get_stc(f"trial_{i+1}_data.csv")
        # 注意：这只会从第一个并行环境获取数据

    # --- 结果处理与保存 ---
    # 从结果字典列表中提取关键指标，例如 'rew' (平均奖励) 和 'len' (平均长度)
    summary_results = []
    for res in all_results:
        summary_results.append({
            'n_episodes': res['n/ep'],
            'mean_reward': res['rew'],
            'std_reward': res['rew_std'],
            'mean_length': res['len'],
            'std_length': res['len_std']
        })

    # 将汇总结果转换为 DataFrame 并保存
    results_df = pd.DataFrame(summary_results)
    print("\n--- Overall Test Summary ---")
    print(results_df)
    try:
        results_df.to_csv(result_save_path, index=False)
        print(f"Results saved to {result_save_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

if __name__ == '__main__':
    main()

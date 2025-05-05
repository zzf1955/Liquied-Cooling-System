# evaluate.py
import numpy as np
# 假设你的 MutiBatteryEnv 类定义在 muti_battery_env.py 文件中
# 请根据你的实际文件结构进行调整
from BatteryEnv.muti_battery_env import MutiBatteryEnv


def evaluate_agent(agent, env_config, num_episodes=10, render=False):
    """
    评估 Agent 在 MutiBatteryEnv 环境中的表现。

    Args:
        agent: 实现了 act(state) 和 reset() 方法的 Agent 对象。
        env_config (dict): 环境配置参数字典。
        num_episodes (int): 评估的回合数。
        render (bool): 是否在评估过程中渲染环境状态。

    Returns:
        tuple: (平均每回合总奖励, 每回合奖励列表)
    """
    try:
        env = MutiBatteryEnv(**env_config)
    except Exception as e:
        print(f"创建 MutiBatteryEnv 时出错: {e}")
        print("请确保环境配置正确且 MutiBatteryEnv 类可用。")
        return -1, [] # 返回错误指示

    episode_rewards = []
    print(f"开始评估，共 {num_episodes} 回合...")

    for episode in range(num_episodes):
        # 重置 Agent 状态 (如果 agent 需要)
        agent.reset()

        # 重置环境获取初始状态
        try:
            state, _ = env.reset()
            if not isinstance(state, np.ndarray):
                 # 确保状态是 numpy 数组，以便 agent 处理
                 state = np.array(state, dtype=np.float32)
        except Exception as e:
            print(f"错误：调用 env.reset() 时出错: {e}")
            continue # 跳过此回合

        episode_reward = 0
        terminated = False
        truncated = False
        step_count = 0

        while not terminated and not truncated:
            # Agent 根据当前状态选择动作
            try:
                action = agent.act(state)
                 # 确保动作是 numpy 数组
                action = np.array(action, dtype=np.float32)
                # 检查动作形状是否符合预期 (num_groups, 2)
                expected_shape = (env.num_groups, 2)
                if action.shape != expected_shape:
                    print(f"警告：Agent 返回的动作形状 {action.shape} 与预期的 {expected_shape} 不符。")
                    # 尝试重塑或采取纠正措施，或者直接报错退出
                    # 这里我们先打印警告，如果 step 失败会进一步报错
            except Exception as e:
                print(f"错误：调用 agent.act(state) 时出错: {e}")
                break # 中断此回合

            # 环境执行动作
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                if not isinstance(next_state, np.ndarray):
                    next_state = np.array(next_state, dtype=np.float32) # 确保状态是 numpy 数组
            except Exception as e:
                print(f"错误：调用 env.step(action) 时出错: {e}")
                break # 中断此回合

            # 累积奖励
            episode_reward += reward
            step_count += 1

            # 更新状态
            state = next_state

            # 可选：渲染环境状态
            if render:
                try:
                    env.render()
                    print(f"Step: {step_count}, Reward: {reward:.4f}")
                except Exception as e:
                    print(f"警告：调用 env.render() 时出错: {e}")

        episode_rewards.append(episode_reward)
        print(f"回合 {episode + 1}/{num_episodes} 结束, 总步数: {step_count}, 总奖励: {episode_reward:.4f}")

    if not episode_rewards:
        print("没有成功完成的回合。")
        average_reward = 0
    else:
        average_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"\n评估结束 ({num_episodes} 回合):")
        print(f"  平均奖励: {average_reward:.4f}")
        print(f"  奖励标准差: {std_reward:.4f}")
        print(f"  最高奖励: {np.max(episode_rewards):.4f}")
        print(f"  最低奖励: {np.min(episode_rewards):.4f}")

    return average_reward, episode_rewards

# --- 示例 Agent (随机策略) ---
class RandomAgent:
    """一个简单的随机 Agent 示例"""
    def __init__(self, num_groups):
        # 动作空间是 Box(low=-1, high=1, shape=(num_groups, 2))
        self.action_shape = (num_groups, 2)

    def act(self, state):
        """返回一个在 [-1, 1] 范围内的随机动作"""
        return np.random.uniform(low=-1.0, high=1.0, size=self.action_shape).astype(np.float32)

    def reset(self):
        """随机 Agent 通常不需要重置内部状态"""
        pass

# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 定义环境配置 (应与训练或测试脚本中的配置一致)
    env_configuration = {
        "num_batteries_per_group": 13,
        "num_groups": 4,
        "max_steps": 512,          # 评估时通常使用与训练相同的最大步数
        "max_current": 10,
        "min_current": 0,
        "env_temp": 298,
        "change_steps": 128       # 在评估时，电流变化逻辑可能需要确认是否适用
        # 可以添加 'randomize_init_current': False 来使用固定的初始电流（如果环境支持）
        # 'randomize_init_current': False
    }

    # 2. 创建或加载你的 Agent
    #    这里我们使用上面定义的 RandomAgent 作为示例
    #    你需要替换成你实际训练好的 Agent 实例
    print("正在创建 Agent (示例：随机 Agent)...")
    # agent_to_evaluate = load_my_trained_agent("path/to/agent_model") # 加载你的 Agent
    agent_to_evaluate = RandomAgent(num_groups=env_configuration["num_groups"])

    # 3. 运行评估
    print("开始运行评估函数...")
    # 设置评估的回合数和是否渲染
    avg_reward, all_rewards = evaluate_agent(agent=agent_to_evaluate,
                                             env_config=env_configuration,
                                             num_episodes=5,  # 运行 5 个回合进行评估
                                             render=False)   # 设置为 True 可以观察每一步的状态

    print(f"\n评估完成。平均奖励: {avg_reward:.4f}")

    # 你可以进一步处理 all_rewards，例如绘制奖励曲线等
    # import matplotlib.pyplot as plt
    # plt.plot(all_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Agent Evaluation Rewards")
    # plt.show()

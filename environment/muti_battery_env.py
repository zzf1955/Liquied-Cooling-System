import numpy as np
from gymnasium import spaces
from muti_battery_module import MutiBattery as MB
import gymnasium as gym

class MutiBatteryEnv(gym.Env):
    def __init__(self, num_batteries=3, episode_steps=400, max_current=10, min_current=0, env_temp=298, change_steps=128, current_change_prob=0.01, **kwargs):
        super(MutiBatteryEnv, self).__init__()
        self.battery_system = MB(num_batteries=num_batteries, **kwargs)  # 初始化多电池系统

        # 动作空间定义：每个电池的冷却液入口温度和流速，范围为[-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_batteries, 2), dtype=np.float32)

        # 状态空间定义：每个电池的核心温度、顶部表面平均温度、电池电流，展平为1D向量
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] * num_batteries, dtype=np.float32),
            high=np.array([500, 500, 10] * num_batteries, dtype=np.float32),
            dtype=np.float32
        )

        # 参数范围
        self.temp_range = (270, 330)  # 温度范围
        self.flow_rate_range = (0, 5)  # 流速范围

        # 环境温度（室温）
        self.environment_temp = env_temp  # 开尔文

        # episode 参数
        self.current_step = 0
        self.max_steps = episode_steps
        self.change_steps = change_steps
        self.num_batteries = num_batteries
        self.max_current = max_current
        self.min_current = min_current
        self.current_change_prob = current_change_prob  # 电流改变的概率

    def step(self, actions):
        # 将标准化的动作映射到实际的温度和流速范围
        for i, action in enumerate(actions):
            inlet_temp = self._map_to_range(action[0], self.temp_range)
            flow_rate = self._map_to_range(action[1], self.flow_rate_range)
            self.battery_system.batteries[i].set_action(inlet_temp, flow_rate)

        # 运行电池系统
        self.battery_system.run(t_seconds=1)

        # 每次step以一定概率随机改变每个电池的输出电流
        for battery in self.battery_system.batteries:
            if np.random.rand() < self.current_change_prob:
                battery.current = np.random.uniform(self.min_current, self.max_current)

        # 获取当前状态并展平为1D向量
        states = []
        for battery in self.battery_system.batteries:
            core_temp = battery.get_core_temperature()
            top_surface_avg_temp = battery.get_top_surface_average_temperature()
            current = battery.current
            states.extend([core_temp, top_surface_avg_temp, current])

        state = np.array(states, dtype=np.float32)

        # 计算每个电池的温度差和奖励
        rewards = []
        for i in range(self.num_batteries):
            delta_temp = abs(state[i * 3] - self.environment_temp)  # 核心温度
            rewards.append(np.exp(-delta_temp / 4 + 2.3) - 5)
        reward = np.mean(rewards)  # 取所有电池的平均奖励

        # 更新计步器
        self.current_step += 1

        # 判断终止条件：任意电池的核心温度超过370或低于270
        terminated = any(state[i * 3] > 310 or state[i * 3] < 290 for i in range(self.num_batteries))

        # 判断截断条件：步数超过max_steps
        truncated = bool(self.current_step >= self.max_steps)

        # 添加调试信息
        info = {}
        
        return state, reward, terminated, truncated, info

    def reset(self, seed=233, randomize_current=True):
        # 重置环境状态
        super().reset(seed=seed)
        np.random.seed(seed)

        for battery in self.battery_system.batteries:
            battery.inlet_temp = self.environment_temp
            battery.flow_rate = 0.1  # 初始化流速为默认值

            if randomize_current:
                battery.current = np.random.uniform(self.min_current, self.max_current)
            else:
                battery.current = 0

        self.current_step = 0

        # 返回初始状态并展平为1D向量
        initial_states = []
        for battery in self.battery_system.batteries:
            initial_states.extend([self.environment_temp, self.environment_temp, battery.current])

        initial_state = np.array(initial_states, dtype=np.float32)
        
        # 添加调试信息
        info = {}
        
        return initial_state, info

    def render(self):
        for i, battery in enumerate(self.battery_system.batteries):
            core_temp = battery.get_core_temperature()
            top_surface_temp = battery.get_top_surface_average_temperature()
            current = battery.current
            inlet_temp, flow_rate = battery.get_action()

            print(f"Battery {i+1} - Core Temp: {core_temp:.2f} K, Top Surface Temp: {top_surface_temp:.2f} K, Current: {current:.2f} A, Action: [Inlet Temp: {inlet_temp:.2f} K, Flow Rate: {flow_rate:.2f} m/s]")

    def _map_to_range(self, value, value_range):
        """将标准化的[-1, 1]值映射到给定的实际范围"""
        min_val, max_val = value_range
        return (value + 1) * 0.5 * (max_val - min_val) + min_val

    def _map_from_range(self, value, value_range):
        """将实际的值从给定的范围映射回标准化的[-1, 1]"""
        min_val, max_val = value_range
        return 2 * (value - min_val) / (max_val - min_val) - 1

def test_muti_battery_env():
    env = MutiBatteryEnv(num_batteries=1, episode_steps=512, max_current=10, min_current=0, env_temp=298, change_steps=128)
    initial_state, _ = env.reset(randomize_current=True)
    
    print("Initial State:")
    env.render()  # 显示初始状态

    steps = 999999  # 运行多少次迭代
    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")

        # 动作：使用当前环境设置的动作参数
        actions = []
        for i in range(env.num_batteries):
            inlet_temp, flow_rate = env.battery_system.batteries[i].get_action()
            # 将实际温度和流速映射回 [-1, 1] 的动作空间
            norm_inlet_temp = env._map_from_range(inlet_temp, env.temp_range)
            norm_flow_rate = env._map_from_range(flow_rate, env.flow_rate_range)
            actions.append([norm_inlet_temp, norm_flow_rate])

        actions = np.array(actions, dtype=np.float32)

        # 运行模拟
        state, reward, terminated, truncated, info = env.step(actions)

        # 显示当前状态和动作
        env.render()

        # 检查是否终止
        if terminated or truncated:
            print("Environment reached terminal state.")
            break

        # 等待用户输入
        user_input = input("Press Enter to continue, or 'b' to modify action and current: ").strip()

        if user_input.lower() == 'b':
            for i in range(env.num_batteries):
                try:
                    inlet_temp = float(input(f"Enter new inlet cooling temperature for Battery {i+1} (K): "))
                    flow_rate = float(input(f"Enter new cooling flow rate for Battery {i+1} (m/s): "))
                    current = float(input(f"Enter new output current for Battery {i+1} (A): "))

                    # 更新电池的动作和输出电流
                    norm_inlet_temp = env._map_from_range(inlet_temp, env.temp_range)
                    norm_flow_rate = env._map_from_range(flow_rate, env.flow_rate_range)
                    env.battery_system.batteries[i].set_action(inlet_temp, flow_rate)
                    env.battery_system.batteries[i].current = current

                except ValueError:
                    print("Invalid input, using previous values.")
        
        elif user_input == '':
            continue

    print("Test completed.")

if __name__ == "__main__":
    # 假设你的环境名为MyEnv
    env = MutiBatteryEnv(episode_steps=400,num_batteries=1)

    #test_muti_battery_env()
    #env = gym.make("BipedalWalker-v3")
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import SAC
    import PrioritizedReplayBuffer

    # 检查环境是否兼容
    #check_env(env)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[128,128],  # 策略网络
            qf=[256,256]   # 价值网络
        )
    )

    buffer_size = 1000000  # 根据需要调整
    replay_buffer = PrioritizedReplayBuffer(buffer_size=buffer_size, observation_space=env.observation_space, action_space=env.action_space)

    model = SAC('MlpPolicy', env, replay_buffer=replay_buffer, verbose=1, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=1000000000000)

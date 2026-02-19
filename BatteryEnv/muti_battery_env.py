import numpy as np
from gymnasium import spaces
from BatteryEnv.muti_battery_module import MutiBattery as MB
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
import gymnasium as gym
import os
import pandas as pd

class MutiBatteryEnv(gym.Env):
    def __init__(self, 
                num_batteries_per_group=13, 
                num_groups=4, 
                max_steps=400, 
                max_battery_tmp=313, 
                min_battery_tmp=288,
                env_temp=300, 
                current_change_prob=0.05,
                current_mu=20,          # 电流均值
                current_sigma=2.5,      # 电流标准差
                current_clip_range=(15, 35),  # 电流截断范围
                flow_rate_range = (0,6),
                inlet_temp_range = (273,310),
                log_step = 1,
                log_path = "",
                con = True
                ):
        
        super(MutiBatteryEnv, self).__init__()
        
        # 初始化多电池系统
        self.battery_system = MB(num_batteries_per_group=num_batteries_per_group, 
                                num_groups=num_groups,
                                env_temp = env_temp)
        
        self.num_batteries_per_group = num_batteries_per_group
        self.num_groups = num_groups
        self.total_batteries = num_batteries_per_group * num_groups
        self.allrew = []

        # 配置电压观测边界
        single_voltage_range = (2.5, 3.65)
        total_voltage_low = num_groups * num_batteries_per_group * single_voltage_range[0]
        total_voltage_high = num_groups * num_batteries_per_group * single_voltage_range[1]

        # 参数范围
        self.inlet_temp_range = inlet_temp_range  # 温度范围 (K)
        self.flow_rate_range = flow_rate_range  # 流速范围 (m/s)
        self.max_battery_tmp = max_battery_tmp
        self.min_battery_tmp = min_battery_tmp

        if not con:
            self.num_flow_actions = int(self.flow_rate_range[1] - self.flow_rate_range[0]) + 1
            self.num_temp_actions = int(self.inlet_temp_range[1] - self.inlet_temp_range[0]) + 1
            total_actions = self.num_flow_actions + self.num_temp_actions
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_actions,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0]*num_groups*3 + [current_clip_range[0]]*num_groups + [total_voltage_low]*num_groups, dtype=np.float32),
            high=np.array([500]*num_groups*3 + [current_clip_range[1]]*num_groups + [total_voltage_high]*num_groups, dtype=np.float32),
            dtype=np.float32
        )
        self.con = con

        # 环境温度（室温）
        self.environment_temp = env_temp  # 开尔文

        # episode 参数
        self.current_step = 0
        self.max_steps = max_steps
        self.num_groups = num_groups
        
        self.current_change_prob = current_change_prob  # 电流改变的概率

        self.current_mu = current_mu
        self.current_sigma = current_sigma

        self.flow_rate_action_log = []
        self.intel_temp_action_log = []
        self.core_temp_log = []
        self.current_log = []

        self.episode_cnt = 0
        self.log_step = log_step
        self.log_path = log_path

        self.reset()
        

    def step(self, action):
        last_action = self.battery_system.batteries[0].get_action()

        if not self.con:
            expected_shape = (self.num_flow_actions + self.num_temp_actions,)
            assert action.shape == expected_shape, f"Expected action shape {expected_shape}, but got {action.shape}"

            # 分割 action 向量
            action_flow = action[:self.num_flow_actions]
            action_temp = action[self.num_flow_actions:]

            # 使用 argmax 选择动作索引 (对应于 "选最大下标")
            flow_rate_index = np.argmax(action_flow)
            inlet_temp_index = np.argmax(action_temp)

            # 根据索引和范围计算实际的 flow_rate 和 inlet_temp
            # 假设范围内的值是连续整数，步长为 1
            flow_rate = self.flow_rate_range[0] + flow_rate_index
            inlet_temp = self.inlet_temp_range[0] + inlet_temp_index
        else:
            inlet_temp = self._map_to_range(action[0], self.inlet_temp_range)
            flow_rate = self._map_to_range(action[1], self.flow_rate_range)

        # if self.battery_system.batteries[0].current == self.current_mu:
        #     flow_rate  = 6 * np.random.rand()
        #     inlet_temp = 280 + np.random.rand() * np.random.rand() * 30
        # else:
        #     flow_rate  = 6 * np.random.rand() * np.random.rand()
        #     inlet_temp = self.environment_temp-20+ np.random.rand() * np.random.rand() * 20

        action = np.array([flow_rate,inlet_temp],dtype=np.float32)

        self.battery_system.batteries[0].set_action(inlet_temp, flow_rate)
        self.flow_rate_action_log.append(flow_rate)
        self.intel_temp_action_log.append(inlet_temp)

        current_log_ = []
        for group_index in range(self.num_groups):
            if np.random.rand()<=self.current_change_prob:
                if self.battery_system.get_group_current(group_idx=group_index) == 0:
                    self.battery_system.set_group_current(group_idx=group_index,current=self.current_mu)
                else:
                    self.battery_system.set_group_current(group_idx=group_index,current=0)
            current_log_.append(self.battery_system.get_group_current(group_idx=group_index))
        self.current_log.append(current_log_)

        # 运行电池系统
        self.battery_system.run(t_seconds=5)

        # 获取状态
        state = self._get_state()

        # 计算奖励
        reward = self._calculate_reward(state, action, last_action)

        # 更新计步器
        self.current_step += 1

        # 判断终止条件：任意电池组的平均核心温度超过阈值
        done = False
        for i in range(self.num_groups):
            group_core_temp = state[i * 5]  # 每组的平均核心温度
            
            # 温度终止条件
            if group_core_temp > self.max_battery_tmp or group_core_temp < self.min_battery_tmp:
                done = True
                break

        # 判断截断条件：步数超过max_steps
        truncated = bool(self.current_step >= self.max_steps)

        # 添加调试信息
        info = {
            'reward': reward,
            'group_temps': [state[i * 5] for i in range(self.num_groups)],
            'group_voltages': [state[i * 5 + 4] for i in range(self.num_groups)],
            'actions': action
        }
        self.allrew.append(reward)
        return state, reward, done, truncated, info

    def _get_state(self):
        """获取环境的当前状态"""
        state = []
        global_current = self.battery_system.batteries[0].current  # 所有电池电流相同

        # 获取每个组的平均状态
        for group_idx in range(self.num_groups):
            start_idx = group_idx * self.num_batteries_per_group
            end_idx = start_idx + self.num_batteries_per_group
            group_batteries = self.battery_system.batteries[start_idx:end_idx]
            
            # 计算组内平均值
            avg_core_temp = sum(b.get_core_temperature() for b in group_batteries) / len(group_batteries)
            avg_top_temp = sum(b.get_top_surface_average_temperature() for b in group_batteries) / len(group_batteries)
            avg_bottom_temp = sum(b.get_bottom_surface_average_temperature() for b in group_batteries) / len(group_batteries)
            total_voltage = sum(b.get_voltage() for b in group_batteries)
            
            state.extend([avg_core_temp, avg_top_temp, avg_bottom_temp, global_current, total_voltage])
        
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, state, action, last_action):
        """计算奖励函数"""
        rewards = []

        flow_rate,inlet_temp = action[0],action[1]

        # 计算每个组的奖励
        core_temps = []
        for i in range(self.num_groups):
            # 核心温度差异
            core_temp = state[i * 5]  # 当前组的平均核心温度
            core_temps.append(core_temp)

            delta_temp = abs(core_temp - self.environment_temp)
            
            group_reward = max(-np.exp(delta_temp) + 3 , -2)
            rewards.append(group_reward)

        self.core_temp_log.append(core_temps)
        
        d_temp_penalty = abs(inlet_temp - last_action[0])/(self.inlet_temp_range[1]-self.inlet_temp_range[0])
        d_flow_penalty = (flow_rate)/(self.flow_rate_range[1]-self.flow_rate_range[0])

        # 平均奖励
        reward = np.mean(rewards) - d_temp_penalty - d_flow_penalty
        return reward

    def reset(self, seed=233, randomize_init_current=False):
        # 记录有效episode
        if self.allrew:
            self.episode_cnt+=1
            if self.episode_cnt % self.log_step == 0:
                # 确保日志目录存在
                if self.log_path and not os.path.exists(self.log_path):
                    os.makedirs(self.log_path)

                # 构建日志文件名
                log_file_path = os.path.join(self.log_path, f"log_{self.episode_cnt}.csv")

                # 检查列表长度是否一致，如果不一致可能需要调整数据准备逻辑
                if not (len(self.allrew) == len(self.flow_rate_action_log) == len(self.intel_temp_action_log) == len(self.core_temp_log) == len(self.current_log)):
                    print(f"Warning: Log data lists have different lengths at episode {self.episode_cnt}. Skipping log generation for this episode.")
                    assert False, "Log data lists have different lengths"
                else:
                    # 准备数据写入 DataFrame
                    log_data = {
                        'reward': self.allrew,
                        'flow_rate_action': self.flow_rate_action_log,
                        'inlet_temp_action': self.intel_temp_action_log
                    }

                    # 为每个组的 current 和 core_temp 创建单独的列
                    for i in range(self.num_groups):
                        # 提取第 i 组的所有时间步的 current 值
                        log_data[f'current_group_{i}'] = [step_currents[i] for step_currents in self.current_log]
                        # 提取第 i 组的所有时间步的 core_temp 值
                        log_data[f'core_temp_group_{i}'] = [step_temps[i] for step_temps in self.core_temp_log]

                    try:
                        df = pd.DataFrame(log_data)
                        df.to_csv(log_file_path, index=False)
                        print(f"Log saved to {log_file_path}")
                    except Exception as e:
                        print(f"Error writing log file {log_file_path}: {e}")

        self.allrew = []
        self.flow_rate_action_log = []
        self.intel_temp_action_log = []
        self.core_temp_log = []
        self.current_log = []

        super().reset(seed=seed)
        np.random.seed(seed)   

        # 重置所有电池的状态
        for battery in self.battery_system.batteries:
            battery.inlet_temp = self.environment_temp
            battery.flow_rate = 0  # 初始化流速为默认值

            if randomize_init_current:
                current = np.random.normal(self.current_mu, self.current_sigma)
                current = np.clip(current, *self.current_clip_range)
                for battery in self.battery_system.batteries:
                    battery.current = current
            else:
                for battery in self.battery_system.batteries:
                    battery.current = 0

        self.current_step = 0
        self.battery_system.reset()

        # 获取初始状态
        initial_state = self._get_state()
        
        # 添加调试信息
        info = {}
        return initial_state, info

    def render(self):
        """输出每个组的状态信息"""
        # 输出每个组的平均值
        for group_idx in range(self.num_groups):
            # 计算范围
            start_idx = group_idx * self.num_batteries_per_group
            end_idx = start_idx + self.num_batteries_per_group
            group_batteries = self.battery_system.batteries[start_idx:end_idx]
            
            # 计算组内平均值
            avg_core_temp = sum(b.get_core_temperature() for b in group_batteries) / len(group_batteries)
            avg_top_temp = sum(b.get_top_surface_average_temperature() for b in group_batteries) / len(group_batteries)
            avg_bottom_temp = sum(b.get_bottom_surface_average_temperature() for b in group_batteries) / len(group_batteries)
            current = self.battery_system.get_group_current(group_idx)
            total_voltage = sum(b.get_voltage() for b in group_batteries)
            
            # 获取第一个电池的冷却参数作为组的代表
            inlet_temp, flow_rate = group_batteries[0].get_action()
            
            print(f"Group {group_idx+1} - Avg Core Temp: {avg_core_temp:.2f} K, "
                  f"Top Temp: {avg_top_temp:.2f} K, Bottom Temp: {avg_bottom_temp:.2f} K, "
                  f"Current: {current:.2f} A, Total Voltage: {total_voltage:.2f} V, "
                  f"Action: [Inlet Temp: {inlet_temp:.2f} K, Flow Rate: {flow_rate:.2f} m/s]")
            
            # 可选：显示组内的一些电池
            if group_idx == 0:  # 只对第一组详细显示
                for i in range(start_idx, start_idx + min(3, self.num_batteries_per_group)):
                    battery = self.battery_system.batteries[i]
                    core_temp = battery.get_core_temperature()
                    print(f"  - Battery {i}: Core Temp: {core_temp:.2f} K")

    def _map_to_range(self, value, value_range):
        """将标准化的[-1, 1]值映射到给定的实际范围"""
        min_val, max_val = value_range
        return (value + 1) * 0.5 * (max_val - min_val) + min_val

    def _map_from_range(self, value, value_range):
        """将实际的值从给定的范围映射回标准化的[-1, 1]"""
        min_val, max_val = value_range
        return 2 * (value - min_val) / (max_val - min_val) - 1

def test_muti_battery_env():
    # 创建环境：4组，每组13个电池
    env = MutiBatteryEnv(num_batteries_per_group=13, num_groups=4, 
                         max_steps=512, max_current=10, min_current=0, 
                         env_temp=298, change_steps=128)
    
    initial_state, _ = env.reset(randomize_init_current=False)
    
    print("Initial State:")
    env.render()  # 显示初始状态

    # 动作：使用当前环境设置的动作参数
    actions = []
    for group_idx in range(env.num_groups):
        # 获取该组第一个电池的动作作为组的代表
        start_idx = group_idx * env.num_batteries_per_group
        inlet_temp, flow_rate = env.battery_system.batteries[start_idx].get_action()
        
        # 将实际温度和流速映射回 [-1, 1] 的动作空间
        norm_inlet_temp = env._map_from_range(inlet_temp, env.inlet_temp_range)
        norm_flow_rate = env._map_from_range(flow_rate, env.flow_rate_range)
        actions.append([norm_inlet_temp, norm_flow_rate])

    steps = 999999  # 运行多少次迭代
    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")

        actions = np.array(actions, dtype=np.float32)

        # 运行模拟
        state, reward, terminated, truncated, info = env.step(actions)

        # 显示当前状态和动作
        env.render()
        print(f"Reward: {reward:.4f}")

        # 检查是否终止
        if terminated or truncated:
            print("Environment reached terminal state.")
            break

        # 等待用户输入
        user_input = input("Press Enter to continue, or 'b' to modify actions and currents: ").strip()

        if user_input.lower() == 'b':
            for group_idx in range(env.num_groups):
                try:
                    print(f"--- Group {group_idx+1} ---")
                    inlet_temp = float(input(f"Enter inlet cooling temperature for Group {group_idx+1} (K): "))
                    flow_rate = float(input(f"Enter cooling flow rate for Group {group_idx+1} (m/s): "))
                    current = float(input(f"Enter output current for Group {group_idx+1} (A): "))

                    # 映射到动作空间
                    norm_inlet_temp = env._map_from_range(inlet_temp, env.inlet_temp_range)
                    norm_flow_rate = env._map_from_range(flow_rate, env.flow_rate_range)

                    real_inlet_temp = env._map_to_range(norm_inlet_temp,env.inlet_temp_range)
                    real_flow_rate = env._map_to_range(norm_flow_rate, env.flow_rate_range)
                    
                    # 更新动作
                    actions[group_idx] = [norm_inlet_temp, norm_flow_rate]
                    
                    # 更新电池组的电流
                    start_idx = group_idx * env.num_batteries_per_group
                    end_idx = start_idx + env.num_batteries_per_group
                    for i in range(start_idx, end_idx):
                        env.battery_system.batteries[i].current = current

                except ValueError:
                    print("Invalid input, using previous values.")
        
        if user_input == "stop":
            break
        elif user_input == '':
            continue

    print("Test completed.")

def make_env(num_batteries_per_group=13,
            num_groups=4,
            episode_steps=512,
            log_path="",
            con=True,
            num_train_envs=8,
            num_test_envs=4,
            use_subproc=True):
    """
    创建向量化环境，支持多进程并行训练

    参数:
        num_batteries_per_group: 每组电池数量
        num_groups: 电池组数量
        episode_steps: 每个episode的最大步数
        log_path: 日志保存路径
        con: 是否使用连续动作空间
        num_train_envs: 训练环境数量
        num_test_envs: 测试环境数量
        use_subproc: 是否使用SubprocVectorEnv（True使用多进程，False使用DummyVectorEnv）
    """

    def _select_env(evaluate=False, seed=None):
        """创建独立的环境实例"""
        env = MutiBatteryEnv(num_batteries_per_group=num_batteries_per_group,
                            num_groups=num_groups,
                            max_steps=episode_steps,
                            log_path=log_path,
                            con=con)
        if seed is not None:
            env.reset(seed=seed)
        return env

    # 根据是否使用子进程选择向量化环境类型
    if use_subproc and num_train_envs > 1:
        # 使用SubprocVectorEnv进行多进程并行
        train_envs = SubprocVectorEnv(
            [lambda: _select_env(seed=i) for i in range(num_train_envs)],
            wait_num=num_train_envs // 2,
            timeout=0.1
        )
        test_envs = SubprocVectorEnv(
            [lambda: _select_env(seed=num_train_envs + i) for i in range(num_test_envs)],
            wait_num=num_test_envs // 2,
            timeout=0.1
        )
    else:
        # 使用DummyVectorEnv作为后备
        train_envs = DummyVectorEnv(
            [lambda: _select_env(seed=i) for i in range(max(1, num_train_envs))]
        )
        test_envs = DummyVectorEnv(
            [lambda: _select_env(seed=num_train_envs + i) for i in range(max(1, num_test_envs))]
        )

    # 创建一个主环境实例用于获取配置信息
    main_env = _select_env()

    return main_env, train_envs, test_envs


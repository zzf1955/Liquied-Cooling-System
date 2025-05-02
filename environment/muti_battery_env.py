import numpy as np
from gymnasium import spaces
from BatteryEnv.muti_battery_module import MutiBattery as MB
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
import gymnasium as gym

class MutiBatteryEnv(gym.Env):
    def __init__(self, 
                num_batteries_per_group=13, 
                num_groups=4, 
                episode_steps=400, 
                max_battery_tmp=3, 
                min_battery_tmp=-3,
                max_current=30, 
                min_current=0, 
                min_battery_voltage=2.5,
                max_battery_voltage=3.65,
                env_temp=298, 
                change_steps=128, 
                current_change_prob=0.01,
                current_mu=25,          # 电流均值
                current_sigma=2.5,      # 电流标准差
                current_clip_range=(15, 35),  # 电流截断范围
                tmp_threshold=5, 
                random_current=False, 
                **kwargs):
        
        super(MutiBatteryEnv, self).__init__()
        
        # 初始化多电池系统
        self.battery_system = MB(num_batteries_per_group=num_batteries_per_group, 
                                num_groups=num_groups, **kwargs)
        
        self.num_batteries_per_group = num_batteries_per_group
        self.num_groups = num_groups
        self.total_batteries = num_batteries_per_group * num_groups
        self.allrew = []

        # 配置电压观测边界
        single_voltage_range = (2.5, 3.65)
        total_voltage_low = num_groups * num_batteries_per_group * single_voltage_range[0]
        total_voltage_high = num_groups * num_batteries_per_group * single_voltage_range[1]

        # 动作空间定义：每个电池组的冷却液入口温度和流速，范围为[-1, 1]
        # 每个组只需要一个动作（组内所有电池使用相同的冷却策略）
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_groups, 2), dtype=np.float32)

        # 状态空间定义：
        # 1. 每个电池组的平均核心温度
        # 2. 每个电池组的平均顶部表面温度
        # 3. 每个电池组的平均底部表面温度
        # 4. 每个电池组的电流
        # 5. 每个电池组的电压
        # 修改状态空间（电流范围调整）
        self.observation_space = spaces.Box(
            low=np.array([0]*num_groups*3 + [current_clip_range[0]]*num_groups + [total_voltage_low]*num_groups, dtype=np.float32),
            high=np.array([500]*num_groups*3 + [current_clip_range[1]]*num_groups + [total_voltage_high]*num_groups, dtype=np.float32),
            dtype=np.float32
        )

        # 参数范围
        self.temp_range = (env_temp+min_battery_tmp*2, env_temp+max_battery_tmp*2)  # 温度范围 (K)
        self.flow_rate_range = (0, 5)  # 流速范围 (m/s)
        self.max_battery_tmp = env_temp+max_battery_tmp
        self.min_battery_tmp = env_temp+min_battery_tmp
        self.threshold_tmp = tmp_threshold

        # 环境温度（室温）
        self.environment_temp = env_temp  # 开尔文

        # episode 参数
        self.current_step = 0
        self.max_steps = episode_steps
        self.change_steps = change_steps
        self.num_groups = num_groups

        self.min_battery_voltage = min_battery_voltage 
        self.max_battery_voltage = max_battery_voltage 
        
        self.max_current = max_current
        self.min_current = min_current
        self.current_change_prob = current_change_prob  # 电流改变的概率
        self.random_current = random_current

        self.current_mu = current_mu
        self.current_sigma = current_sigma
        self.current_clip_range = current_clip_range

        self.flow_rate_action_log = []
        self.intel_temp_action_log = []
        
        self.reset()
        
        # if not random_current:
        #     # 预生成电流方案（按组）
        #     self.fixed_current_list = []
        #     for _ in range(episode_steps):
        #         if not self.fixed_current_list:
        #             # 初始时电流值为 0
        #             current_values = [0] * self.num_groups
        #         else:
        #             # 基于上一步的电流值，可能随机改变
        #             current_values = self.fixed_current_list[-1][:]
        #             for j in range(self.num_groups):
        #                 if np.random.rand() < current_change_prob:
        #                     current_values[j] = np.random.random() * max_current
                
        #         self.fixed_current_list.append(current_values)
        
        # 修改预生成电流的逻辑
        if not random_current:
            self.fixed_current_list = []
            for _ in range(episode_steps):
                # 生成单个电流值用于所有组
                current = np.random.normal(current_mu, current_sigma)
                current = np.clip(current, *current_clip_range)
                self.fixed_current_list.append([current] * self.num_groups)

    def step(self, actions):
        # 将标准化的动作映射到实际的温度和流速范围，并应用到每个组
        assert np.all(actions >= -1) and np.all(actions <= 1), \
        f"Action values out of [-1, 1] range. Min: {np.min(actions)}, Max: {np.max(actions)}"
    
        if actions.ndim == 1:
            # 验证一维数组元素数量是否匹配目标形状
            expected_elements = self.action_space.shape[0] * self.action_space.shape[1]
            assert len(actions) == expected_elements, \
                f"1D action array length mismatch. Expected {expected_elements}, got {len(actions)}"
            # 重塑为二维数组
            actions = actions.reshape(self.action_space.shape)
        elif actions.ndim == 2:
            # 直接验证二维数组形状
            assert actions.shape == self.action_space.shape, \
                f"2D action shape mismatch. Expected {self.action_space.shape}, got {actions.shape}"
        else:
            raise AssertionError(f"Invalid action dimensions: {actions.ndim}")

        for group_idx, action in enumerate(actions):
            inlet_temp = self._map_to_range(action[0], self.temp_range)
            flow_rate = self._map_to_range(action[1], self.flow_rate_range)
            self.flow_rate_action_log.append(flow_rate)
            self.intel_temp_action_log.append(inlet_temp)
            
            # 将相同的动作应用到该组中的所有电池
            start_idx = group_idx * self.num_batteries_per_group
            end_idx = start_idx + self.num_batteries_per_group
            
            # 如果是第一组，直接使用设定的入口温度
            if group_idx == 0:
                for i in range(start_idx, end_idx):
                    self.battery_system.batteries[i].set_action(inlet_temp, flow_rate)
            else:
                # 对于其他组，使用前一组的出口温度作为入口温度
                prev_group_last_battery = self.battery_system.batteries[(group_idx-1) * self.num_batteries_per_group + self.num_batteries_per_group - 1]
                prev_outlet_temp = prev_group_last_battery.intel_temp
                
                for i in range(start_idx, end_idx):
                    self.battery_system.batteries[i].set_action(prev_outlet_temp, flow_rate)

        # 运行电池系统
        self.battery_system.run(t_seconds=1)

        # 统一电流处理逻辑
        if False:
            if self.random_current:
                # 生成新的全局电流值
                current = np.random.normal(self.current_mu, self.current_sigma)
                current = np.clip(current, *self.current_clip_range)
                
                # 应用到所有电池
                for battery in self.battery_system.batteries:
                    battery.current = current
            else:
                # 使用预生成的全局电流值
                current = self.fixed_current_list[self.current_step][0]  # 所有组相同
                for battery in self.battery_system.batteries:
                    battery.current = current

        # 获取状态
        state = self._get_state()

        # 计算奖励
        reward = self._calculate_reward(state)

        # 更新计步器
        self.current_step += 1

        # 判断终止条件：任意电池组的平均核心温度超过阈值或电压超出范围
        terminated = False
        for i in range(self.num_groups):
            group_core_temp = state[i * 5]  # 每组的平均核心温度
            
            # 温度终止条件
            if group_core_temp > self.max_battery_tmp or group_core_temp < self.min_battery_tmp:
                terminated = True
                break
                
            # 电压终止条件：检查组内每个电池的单体电压
            start_idx = i * self.num_batteries_per_group
            end_idx = start_idx + self.num_batteries_per_group
            for battery_idx in range(start_idx, end_idx):
                battery = self.battery_system.batteries[battery_idx]
                battery_voltage = battery.get_voltage()
                if battery_voltage < self.min_battery_voltage or battery_voltage > self.max_battery_voltage:
                    terminated = True
                    break
            if terminated:
                break

        # 判断截断条件：步数超过max_steps
        truncated = bool(self.current_step >= self.max_steps)

        # 添加调试信息
        info = {
            'reward': reward,
            'group_temps': [state[i * 5] for i in range(self.num_groups)],
            'group_voltages': [state[i * 5 + 4] for i in range(self.num_groups)],
            'actions': actions
        }
        # print(fr, end = '')
        # print(it)
        # if terminated or truncated:
        #     print("stop!")
        self.allrew.append(reward)
        return state, reward, terminated, truncated, info

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
            # 串联时电流相同，直接获取组内任意一个电池的电流
            # current = self.battery_system.get_group_current(group_idx)
            # 串联时总电压为所有电池电压之和
            total_voltage = sum(b.get_voltage() for b in group_batteries)
            
            state.extend([avg_core_temp, avg_top_temp, avg_bottom_temp, global_current, total_voltage])
        
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, state):
        """计算奖励函数"""
        rewards = []
        
        # 计算每个组的奖励
        for i in range(self.num_groups):
            # 核心温度差异
            core_temp = state[i * 5]  # 当前组的平均核心温度
            delta_temp = abs(core_temp - self.environment_temp)
            
            # 使用指数奖励函数
            group_reward = -delta_temp+2
            rewards.append(group_reward)
        
        # 平均奖励
        reward = np.mean(rewards)
        return reward

    def reset(self, seed=233, randomize_init_current=False):
        # 重置环境状态
        if self.flow_rate_action_log and self.allrew and self.intel_temp_action_log:
            print()
        print(sum(self.allrew))
        self.allrew = []
        self.flow_rate_action_log = []
        self.intel_temp_action_log = []
        super().reset(seed=seed)
        #np.random.seed(seed)

        # 重置所有电池的状态
        for battery in self.battery_system.batteries:
            battery.inlet_temp = self.environment_temp
            battery.flow_rate = 0.1  # 初始化流速为默认值

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
                         episode_steps=512, max_current=10, min_current=0, 
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
        norm_inlet_temp = env._map_from_range(inlet_temp, env.temp_range)
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
                    norm_inlet_temp = env._map_from_range(inlet_temp, env.temp_range)
                    norm_flow_rate = env._map_from_range(flow_rate, env.flow_rate_range)

                    real_inlet_temp = env._map_to_range(norm_inlet_temp,env.temp_range)
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
            max_current=10, 
            min_current=0, 
            env_temp=298, 
            change_steps=128):

    env = MutiBatteryEnv(num_batteries_per_group=num_batteries_per_group, num_groups=num_groups, 
                         episode_steps=episode_steps, max_current=max_current, min_current=min_current, 
                         env_temp=env_temp, change_steps=change_steps)
    
    def _select_env(evaluate = False):
        return env

    env = _select_env()

    train_envs = DummyVectorEnv(
        [lambda: _select_env() for _ in range(1)])
    test_envs = DummyVectorEnv(
        [lambda: _select_env(True) for _ in range(1)])
    
    return env,train_envs,test_envs

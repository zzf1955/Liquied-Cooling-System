import numpy as np
from gymnasium import spaces
import gymnasium as gym
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
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
                current_mu=30,
                current_sigma=10,
                current_clip_range=(0, 50),
                flow_rate_range = (0, 3),
                inlet_temp_range = (288, 295),
                log_step = 100, # 增加步数，减少写入频率
                log_path = "",
                con = True,
                env_index = 0 # 新增：标识环境索引，防止日志冲突
                ):
        
        super(MutiBatteryEnv, self).__init__()
        from BatteryEnv.multi_battery_module import MultiBattery as MB
        
        self.battery_system = MB(num_batteries_per_group=num_batteries_per_group, 
                                num_groups=num_groups,
                                env_temp = env_temp)
        
        self.num_batteries_per_group = num_batteries_per_group
        self.num_groups = num_groups
        self.total_batteries = num_batteries_per_group * num_groups
        self.env_temp = env_temp
        self.env_index = env_index

        # 参数范围
        self.inlet_temp_range = inlet_temp_range 
        self.flow_rate_range = flow_rate_range 
        self.max_battery_tmp = max_battery_tmp
        self.min_battery_tmp = min_battery_tmp
        self.current_clip_range = current_clip_range

        # 1. 动作空间
        if not con:
            # 离散动作逻辑保持不变，但建议优先使用 con=True
            self.num_flow_actions = int(self.flow_rate_range[1] - self.flow_rate_range[0]) + 1
            self.num_temp_actions = int(self.inlet_temp_range[1] - self.inlet_temp_range[0]) + 1
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_flow_actions + self.num_temp_actions,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # 2. 状态空间重构 (归一化到约 [0, 1] 或 [-1, 1])
        # 每组包含：[avg_core, max_core, min_core, avg_top, avg_bottom, current, total_voltage, last_avg_core] -> 8位
        # 我们使用相对温度：T_actual - T_env
        self.observation_space = spaces.Box(
            low=-50.0, 
            high=100.0, 
            shape=(num_groups * 8,), 
            dtype=np.float32
        )
        
        self.con = con
        self.current_step = 0
        self.max_steps = max_steps
        self.current_change_prob = current_change_prob
        self.current_mu = current_mu
        self.current_sigma = current_sigma
        
        # 记录器
        self.last_action = None
        self.last_obs_raw = None 
        self.reset_logs()
        self.log_path = log_path
        self.log_step = log_step
        self.episode_cnt = 0

    def reset_logs(self):
        self.allrew = []
        self.flow_rate_action_log = []
        self.intel_temp_action_log = []
        self.core_temp_log = [] # 记录每组平均温
        self.current_log = []

    def _get_obs(self):
        """获取并归一化状态"""
        obs = []
        for i in range(self.num_groups):
            stats = self.battery_system.get_group_stats(i)
            start_idx = i * self.num_batteries_per_group
            end_idx = start_idx + self.num_batteries_per_group
            group_temps = [b.get_core_temperature() for b in self.battery_system.batteries[start_idx:end_idx]]
            
            # 关键：提取极值以感知串行温差
            max_t = max(group_temps) - self.env_temp
            min_t = min(group_temps) - self.env_temp
            avg_t = stats['avg_core'] - self.env_temp
            top_t = stats['avg_top'] - self.env_temp
            bot_t = stats['avg_bot'] - self.env_temp
            
            curr = stats['current'] / self.current_clip_range[1] # 归一化电流
            volt = stats['total_voltage'] / (self.num_batteries_per_group * 3.65) # 归一化电压
            
            # 历史趋势：记录上一时刻的平均温偏差
            if self.last_obs_raw is None:
                delta_h = avg_t
            else:
                delta_h = self.last_obs_raw[i * 8] 

            obs.extend([avg_t, max_t, min_t, top_t, bot_t, curr, volt, delta_h])
        
        res = np.array(obs, dtype=np.float32)
        self.last_obs_raw = res.copy()
        return res

    def step(self, action):
        # 1. 动作映射
        if not self.con:
            flow_rate = self.flow_rate_range[0] + np.argmax(action[:self.num_flow_actions])
            inlet_temp = self.inlet_temp_range[0] + np.argmax(action[self.num_flow_actions:])
        else:
            inlet_temp = self._map_to_range(action[0], self.inlet_temp_range)
            flow_rate = self._map_to_range(action[1], self.flow_rate_range)

        # 2. 执行动作
        self.battery_system.set_group_controls(flow_rate, inlet_temp)
        
        # 3. 电流随机变化逻辑 (全局统一控制)
        current_step_info = []

        # 获取当前系统中第一组的电流作为基准状态
        # 所有组电流步调一致，我们只需要查看一组的状态即可
        base_current = self.battery_system.get_group_stats(0)['current']

        # 统一判断：整个系统是否在这一步发生电流切换
        if self.np_random.random() <= self.current_change_prob:
            # 如果当前是 0，则切换到均值电流；如果当前有电流，则归零
            new_curr = self.current_mu if base_current == 0 else 0
            
            # 统一设置所有组的电流，确保物理一致性
            for i in range(self.num_groups):
                self.battery_system.set_group_current(i, new_curr)

        # 获取更新后的统一电流值用于记录
        final_current = self.battery_system.get_group_stats(0)['current']
        current_step_info = [final_current] * self.num_groups
        self.current_log.append(current_step_info)

        # 4. 物理模拟
        self.battery_system.run(t_seconds=5)
        
        # 5. 获取新状态与奖励
        obs = self._get_obs()
        reward = self._calculate_reward(obs, [inlet_temp, flow_rate])
        
        self.current_step += 1
        
        # 6. 终止条件
        done = False
        for i in range(self.num_groups):
            core_avg = obs[i*8] + self.env_temp
            if core_avg > self.max_battery_tmp or core_avg < self.min_battery_tmp:
                done = True
                reward -= 50 # 额外惩罚崩溃状态
                break
        
        truncated = self.current_step >= self.max_steps
        
        # 记录数据用于日志
        self.allrew.append(reward)
        self.flow_rate_action_log.append(flow_rate)
        self.intel_temp_action_log.append(inlet_temp)
        self.current_log.append(current_step_info)
        self.core_temp_log.append([obs[i*8] + self.env_temp for i in range(self.num_groups)])

        self.last_action = [inlet_temp, flow_rate]
        
        return obs, reward, done, truncated, {"raw_reward": reward}

    def _calculate_reward(self, obs, current_action):
        """
        优化后的奖励函数
        """
        target_temp_offset = 298.15 - self.env_temp # 目标是相对于环境的偏移量
        total_r = 0
        
        inlet_temp, flow_rate = current_action
        
        for i in range(self.num_groups):
            idx = i * 8
            avg_t, max_t, min_t = obs[idx], obs[idx+1], obs[idx+2]
            last_avg_t = obs[idx+7]
            
            # A. 温度偏差惩罚 (使用平方项，偏差越大惩罚呈指数级上升)
            dist = abs(avg_t - target_temp_offset)
            total_r -= 0.5 * (dist ** 2)
            
            # B. 组内温差惩罚 (解决串行冷却梯度问题)
            # max_t 和 min_t 的差值反映了冷却不均
            total_r -= 1.5 * (max_t - min_t)
            
            # C. 趋势惩罚：如果已经热了还在升温，重罚
            if avg_t > target_temp_offset and avg_t > last_avg_t:
                total_r -= 2.0 * (avg_t - last_avg_t)

        # D. 控制成本与平滑度
        energy_penalty = 0.2 * flow_rate
        smooth_penalty = 0
        if self.last_action is not None:
            smooth_penalty = 0.1 * abs(inlet_temp - self.last_action[0])
            
        return total_r - energy_penalty - smooth_penalty

    def reset(self, seed=None, options=None):
        # 只有在测试环境或特定间隔才保存日志，避免多进程写冲突
        if self.allrew and self.log_path and self.episode_cnt % self.log_step == 0:
            self._save_csv_log()
            
        self.episode_cnt += 1
        super().reset(seed=seed)
        self.battery_system.reset()
        self.current_step = 0
        self.last_action = None
        self.last_obs_raw = None
        self.reset_logs()
        
        return self._get_obs(), {}

    def _save_csv_log(self):
        # 增加 env_index 防止多进程冲突
        os.makedirs(self.log_path, exist_ok=True)
        log_file = os.path.join(self.log_path, f"env_{self.env_index}_ep_{self.episode_cnt}.csv")
        try:
            df = pd.DataFrame({
                'reward': self.allrew,
                'flow': self.flow_rate_action_log,
                'inlet': self.intel_temp_action_log,
                'avg_temp_g0': [t[0] for t in self.core_temp_log]
            })
            df.to_csv(log_file, index=False)
        except:
            pass

    def _map_to_range(self, value, val_range):
        return (value + 1) * 0.5 * (val_range[1] - val_range[0]) + val_range[0]
    
    def _map_from_range(self, value, value_range):
        """将实际的值从给定的范围映射回标准化的[-1, 1]"""
        min_val, max_val = value_range
        return 2 * (value - min_val) / (max_val - min_val) - 1
    

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
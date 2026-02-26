"""
多电池储能系统液冷热管理强化学习环境

状态空间：每组电池 [顶部温度, 底部温度, 核心温度, 核心最大温度, 核心最小温度, 电压, 电流, 历史核心温度]
动作空间：[冷却液入口温度, 冷却液流速]
奖励函数：惩罚超温、组内/组间温差、控制成本
"""

import numpy as np
from gymnasium import spaces
import gymnasium as gym
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
import os
import pandas as pd
from loguru import logger
from BatteryEnv.multi_battery_module import MultiBattery as MB

class MultiBatteryEnv(gym.Env):
    """
    多电池储能系统液冷热管理环境

    状态空间 (每组8维):
        - top_t: 顶部平均温度 (K)
        - bottom_t: 底部平均温度 (K)
        - core_t: 核心平均温度 (K)
        - core_max_t: 核心最大温度 (K) [用于组内温差计算]
        - core_min_t: 核心最小温度 (K) [用于组内温差计算]
        - voltage: 总电压 (V)
        - current: 电流 (A)
        - last_core: 上一时刻核心温度 (K) [用于趋势]

    动作空间 (2维):
        - inlet_temp: 冷却液入口温度 (288-295 K, 即 15-22°C)
        - flow_rate: 冷却液流速 (0-3 m/s)

    奖励函数:
        - 超温惩罚: 温度超过安全边界时惩罚
        - 组内温差惩罚: 同一组内电芯温差
        - 组间温差惩罚: 不同电池组之间的温差
        - 控制成本: λ * flow_rate + μ * |Δinlet_temp|
    """

    def __init__(
        self,
        num_batteries_per_group: int = 13,
        num_groups: int = 4,
        max_steps: int = 400,
        max_battery_tmp: float = 313.0,  # 40°C
        min_battery_tmp: float = 288.0,  # 15°C
        target_temp: float = 298.0,  # 25°C - 目标温度
        env_temp: float = 300.0,  # 环境温度 27°C
        current_mu: float = 30.0,  # 电流均值
        current_sigma: float = 10.0,  # 电流标准差
        current_clip_range: tuple = (0.0, 50.0),
        flow_rate_range: tuple = (0.0, 3.0),
        inlet_temp_range: tuple = (288.0, 295.0),  # 15-22°C
        current_change_prob: float = 0.05,  # 电流重采样概率
        lambda_cost: float = 0.5,  # 流速成本系数
        mu_cost: float = 0.5,  # 入口温度平滑成本系数
        log_step: int = 100,
        log_path: str = "",
        con: bool = True,
        env_index: int = 0,
        debug: bool = True,
    ):
        """
        初始化多电池RL环境

        Args:
            num_batteries_per_group: 每组电池数量
            num_groups: 电池组数量
            max_steps: 每个episode最大步数
            max_battery_tmp: 最大安全温度 (K)
            min_battery_tmp: 最小安全温度 (K)
            target_temp: 目标温度 (K)
            env_temp: 环境温度 (K)
            current_mu: 电流均值 (A)
            current_sigma: 电流标准差 (A)
            current_clip_range: 电流截断范围
            flow_rate_range: 流速范围 (m/s)
            inlet_temp_range: 入口温度范围 (K)
            current_change_prob: 电流重采样概率
            lambda_cost: 流速成本系数
            mu_cost: 入口温度平滑成本系数
            log_step: 日志保存间隔
            log_path: 日志路径
            con: 是否使用连续动作空间
            env_index: 环境索引（多进程用）
            debug: 是否开启调试输出
        """
        super(MultiBatteryEnv, self).__init__()

        # 导入物理模型
        

        # 物理系统
        self.battery_system = MB(
            num_batteries_per_group=num_batteries_per_group,
            num_groups=num_groups,
            env_temp=env_temp,
        )

        # === 配置参数 ===
        self.num_batteries_per_group = num_batteries_per_group
        self.num_groups = num_groups
        self.total_batteries = num_batteries_per_group * num_groups
        self.env_temp = env_temp
        self.target_temp = target_temp
        self.env_index = env_index
        self.debug = debug

        # 温度边界
        self.max_battery_tmp = max_battery_tmp
        self.min_battery_tmp = min_battery_tmp

        # 动作范围
        self.inlet_temp_range = inlet_temp_range  # (288, 295) K
        self.flow_rate_range = flow_rate_range  # (0, 3) m/s

        # 电流参数
        self.current_mu = current_mu
        self.current_sigma = current_sigma
        self.current_clip_range = current_clip_range
        self.current_change_prob = current_change_prob

        # 奖励函数系数
        self.lambda_cost = lambda_cost
        self.mu_cost = mu_cost

        # === 动作空间 ===
        # 连续模式: [-1, 1] -> 映射到实际范围
        if con:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )
        else:
            # 离散模式
            self.num_flow_actions = int(flow_rate_range[1] - flow_rate_range[0]) + 1
            self.num_temp_actions = int(inlet_temp_range[1] - inlet_temp_range[0]) + 1
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_flow_actions + self.num_temp_actions,),
                dtype=np.float32,
            )

        # === 状态空间 ===
        # 每组8维: [top_t, bottom_t, core_t, core_max_t, core_min_t, voltage, current, last_core]
        # 使用实际物理单位
        obs_dim = num_groups * 8
        self.observation_space = spaces.Box(
            low=np.array([0.0] * obs_dim, dtype=np.float32),
            high=np.array(
                [500.0] * obs_dim, dtype=np.float32  # 温度/电压/电流统一上界
            ),
            dtype=np.float32,
        )

        self.con = con
        self.current_step = 0
        self.max_steps = max_steps

        # === 状态跟踪 ===
        self.last_action = None  # 上一步的动作 [inlet_temp, flow_rate]
        self.last_core_temps = None  # 上一步每组的平均核心温度

        # === 日志记录 ===
        self.episode_cnt = 0
        self.log_step = log_step
        self.log_path = log_path
        self.reset_logs()

        # 调试日志
        if self.debug:
            logger.info(
                f"[Env {env_index}] 初始化完成: {num_groups}组×{num_batteries_per_group}电池"
            )
            logger.info(
                f"[Env {env_index}] 动作空间: inlet_temp={inlet_temp_range}K, flow_rate={flow_rate_range}m/s"
            )
            logger.info(
                f"[Env {env_index}] 温度边界: [{min_battery_tmp}, {max_battery_tmp}]K"
            )
            logger.info(
                f"[Env {env_index}] 状态空间: {num_groups}组 × 8维 = {obs_dim}维"
            )

    def reset_logs(self):
        """重置日志数据"""
        self.allrew = []
        self.flow_rate_log = []
        self.inlet_temp_log = []
        self.core_temp_log = []  # 每步每组的平均核心温度
        self.core_max_temp_log = []  # 每步每组的最大核心温度
        self.core_min_temp_log = []  # 每步每组的最小核心温度
        self.current_log = []
        self.voltage_log = []
        self.reward_breakdown_log = []  # 奖励分解

    def _get_obs(self) -> np.ndarray:
        """
        获取当前观测状态

        Returns:
            np.ndarray: 状态数组，形状 (num_groups * 8,)
                每组: [top_t, bottom_t, core_t, core_max_t, core_min_t, voltage, current, last_core]
        """
        obs = []

        for i in range(self.num_groups):
            # 获取该组的统计信息
            stats = self.battery_system.get_group_stats(i)

            # 获取组内所有电芯的核心温度（用于计算组内温差）
            start_idx = i * self.num_batteries_per_group
            end_idx = start_idx + self.num_batteries_per_group
            group_temps = [
                b.get_core_temperature()
                for b in self.battery_system.batteries[start_idx:end_idx]
            ]

            # 实际温度值
            top_t = stats["avg_top"]
            bottom_t = stats["avg_bot"]
            core_t = stats["avg_core"]
            core_max_t = max(group_temps)  # 核心最大温度
            core_min_t = min(group_temps)  # 核心最小温度
            voltage = stats["total_voltage"]
            current = stats["current"]

            # 历史核心温度（趋势）
            if self.last_core_temps is not None:
                last_core = self.last_core_temps[i]
            else:
                last_core = core_t  # 初始时刻

            obs.extend([top_t, bottom_t, core_t, core_max_t, core_min_t, voltage, current, last_core])

        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        执行一步环境交互

        Args:
            action: 动作数组，形状 (2,) 或 (n_actions,)

        Returns:
            obs: 新状态
            reward: 奖励
            done: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # === 1. 动作解析 ===
        if not self.con:
            # 离散动作
            flow_rate = self.flow_rate_range[0] + np.argmax(
                action[: self.num_flow_actions]
            )
            inlet_temp = self.inlet_temp_range[0] + np.argmax(
                action[self.num_flow_actions :]
            )
        else:
            # 连续动作: action[0] -> inlet_temp, action[1] -> flow_rate
            # 先裁剪到 [-1, 1] 范围
            action = np.clip(action, -1.0, 1.0)
            inlet_temp = self._map_to_range(action[0], self.inlet_temp_range)
            flow_rate = self._map_to_range(action[1], self.flow_rate_range)

        # 记录当前动作
        current_action = [inlet_temp, flow_rate]

        # === 2. 应用控制 ===
        self.battery_system.set_group_controls(flow_rate, inlet_temp)

        # === 3. 电流随机变化 ===
        # 每步有小概率重新采样电流（模拟负载变化）
        if self.np_random.random() <= self.current_change_prob:
            # 正态分布采样电流
            new_current = self.np_random.normal(self.current_mu, self.current_sigma)
            new_current = np.clip(new_current, *self.current_clip_range)

            # 统一设置所有组的电流
            for i in range(self.num_groups):
                self.battery_system.set_group_current(i, new_current)

            if self.debug:
                logger.debug(
                    f"[Env {self.env_index}] 电流重采样: {new_current:.2f}A"
                )

        # === 4. 物理模拟 ===
        self.battery_system.run(t_seconds=5)

        # === 5. 获取新状态 ===
        obs = self._get_obs()

        # 保存当前核心温度作为历史
        current_core_temps = []
        for i in range(self.num_groups):
            idx = i * 8 + 2  # core_t 的位置
            current_core_temps.append(obs[idx])
        self.last_core_temps = current_core_temps

        # === 6. 计算奖励 ===
        reward, reward_breakdown = self._calculate_reward(
            obs, current_action, self.last_action
        )

        # === 7. 终止条件 ===
        self.current_step += 1
        done = False
        truncated = False

        # 温度超限检测
        for i in range(self.num_groups):
            core_t = obs[i * 8 + 2]  # core_t
            if core_t > self.max_battery_tmp or core_t < self.min_battery_tmp:
                done = True
                reward -= 50.0  # 额外惩罚
                if self.debug:
                    logger.warning(
                        f"[Env {self.env_index}] 温度超限! 组{i}: {core_t:.2f}K"
                    )
                break

        # 步数截断
        if self.current_step >= self.max_steps:
            truncated = True

        # === 8. 记录日志 ===
        self.allrew.append(reward)
        self.flow_rate_log.append(flow_rate)
        self.inlet_temp_log.append(inlet_temp)
        self.core_temp_log.append(
            [obs[i * 8 + 2] for i in range(self.num_groups)]
        )
        self.core_max_temp_log.append(
            [obs[i * 8 + 3] for i in range(self.num_groups)]
        )
        self.core_min_temp_log.append(
            [obs[i * 8 + 4] for i in range(self.num_groups)]
        )
        self.current_log.append(
            [self.battery_system.get_group_stats(i)["current"] for i in range(self.num_groups)]
        )
        self.voltage_log.append(
            [self.battery_system.get_group_stats(i)["total_voltage"] for i in range(self.num_groups)]
        )
        self.reward_breakdown_log.append(reward_breakdown)

        # 保存动作供下一步使用
        self.last_action = current_action

        # 调试输出
        if self.debug and self.current_step % 50 == 0:
            logger.debug(
                f"[Env {self.env_index}] Step {self.current_step}: "
                f"core_t={obs[2]:.2f}K, reward={reward:.2f}, "
                f"action=[inlet={inlet_temp:.2f}K, flow={flow_rate:.2f}m/s]"
            )

        info = {
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "group_temps": [obs[i * 8 + 2] for i in range(self.num_groups)],
            "group_temps_max": [obs[i * 8 + 3] for i in range(self.num_groups)],
            "group_temps_min": [obs[i * 8 + 4] for i in range(self.num_groups)],
            "actions": current_action,
        }

        return obs, reward, done, truncated, info

    def _calculate_reward(
        self, obs: np.ndarray, current_action: list, last_action: list
    ) -> tuple:
        """
        计算奖励函数

        奖励组成:
            1. 温度偏差惩罚: 偏离目标温度的平方
            2. 温度达标奖励: 温度在目标范围内时获得正奖励
            3. 组间温差惩罚: 不同组之间的温差
            4. 控制成本: λ * flow_rate + μ * |Δinlet_temp|

        Args:
            obs: 当前观测
            current_action: 当前动作 [inlet_temp, flow_rate]
            last_action: 上一步动作

        Returns:
            total_reward: 总奖励
            breakdown: 奖励分解字典
        """
        inlet_temp, flow_rate = current_action
        target_temp = self.target_temp

        # 初始化奖励分解
        breakdown = {
            "temp_penalty": 0.0,
            "temp_reward": 0.0,
            "inter_group_temp_diff_penalty": 0.0,
            "control_cost": 0.0,
            "total": 0.0,
        }

        total_reward = 0.0

        # === 1. 温度偏差惩罚 ===
        for i in range(self.num_groups):
            core_t = obs[i * 8 + 2]  # 核心平均温度
            dist = abs(core_t - target_temp)
            temp_penalty = 0.5 * (dist ** 2)  # 平方惩罚
            total_reward -= temp_penalty
            breakdown["temp_penalty"] -= temp_penalty

        # === 2. 温度达标奖励 (新增) ===
        # 当温度在目标范围附近时给予正奖励，让训练更容易收敛
        temp_tolerance = 5.0  # 温度容差范围 ±5K
        for i in range(self.num_groups):
            core_t = obs[i * 8 + 2]
            if abs(core_t - target_temp) < temp_tolerance:
                # 温度越接近目标，奖励越高
                reward_factor = 1.0 - abs(core_t - target_temp) / temp_tolerance
                temp_reward = 10.0 * reward_factor  # 最高10分
                total_reward += temp_reward
                breakdown["temp_reward"] += temp_reward

        # === 3. 组间温差惩罚 ===
        if self.num_groups > 1:
            group_avg_temps = [obs[i * 8 + 2] for i in range(self.num_groups)]
            inter_diff = max(group_avg_temps) - min(group_avg_temps)
            total_reward -= 1.5 * inter_diff  # 组间温差系数
            breakdown["inter_group_temp_diff_penalty"] -= 1.5 * inter_diff

        # === 4. 控制成本 ===
        # cost = λ * R_flow + μ * |T_inlet(t) - T_inlet(t-1)|
        flow_cost = self.lambda_cost * flow_rate
        smooth_cost = 0.0
        if last_action is not None:
            smooth_cost = self.mu_cost * abs(inlet_temp - last_action[0])

        total_reward -= flow_cost + smooth_cost
        breakdown["control_cost"] = -(flow_cost + smooth_cost)

        breakdown["total"] = total_reward
        return total_reward, breakdown

    def reset(self, seed: int = None, options: dict = None):
        """
        重置环境

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            obs: 初始观测
            info: 额外信息
        """
        # 保存日志
        if self.allrew and self.log_path and self.episode_cnt % self.log_step == 0:
            self._save_csv_log()

        self.episode_cnt += 1

        # 调用父类reset（设置随机种子）
        super().reset(seed=seed)

        # 重置物理系统
        self.battery_system.reset()

        # 重置状态
        self.current_step = 0
        self.last_action = None
        self.last_core_temps = None
        self.reset_logs()

        # 获取初始观测
        obs = self._get_obs()

        # 初始化历史温度
        self.last_core_temps = [obs[i * 8 + 2] for i in range(self.num_groups)]

        if self.debug:
            logger.info(
                f"[Env {self.env_index}] Episode {self.episode_cnt} 开始: 初始温度={obs[2]:.2f}K"
            )

        return obs, {}

    def _save_csv_log(self):
        """保存CSV日志"""
        if not self.log_path:
            return

        os.makedirs(self.log_path, exist_ok=True)
        log_file = os.path.join(
            self.log_path, f"env_{self.env_index}_ep_{self.episode_cnt}.csv"
        )

        try:
            data = {
                "step": list(range(len(self.allrew))),
                "reward": self.allrew,
                "flow_rate": self.flow_rate_log,
                "inlet_temp": self.inlet_temp_log,
            }

            # 每组的数据
            for i in range(self.num_groups):
                data[f"core_temp_g{i}"] = [t[i] for t in self.core_temp_log]
                data[f"core_max_temp_g{i}"] = [t[i] for t in self.core_max_temp_log]
                data[f"core_min_temp_g{i}"] = [t[i] for t in self.core_min_temp_log]
                data[f"current_g{i}"] = [c[i] for c in self.current_log]

            df = pd.DataFrame(data)
            df.to_csv(log_file, index=False)

            if self.debug:
                logger.info(f"[Env {self.env_index}] 日志保存: {log_file}")

        except Exception as e:
            logger.error(f"[Env {self.env_index}] 日志保存失败: {e}")

    def _map_to_range(self, value: float, val_range: tuple) -> float:
        """
        将标准化值 [-1, 1] 映射到实际范围

        Args:
            value: 标准化值
            val_range: 目标范围 (min, max)

        Returns:
            实际值
        """
        min_val, max_val = val_range
        return (value + 1) * 0.5 * (max_val - min_val) + min_val

    def _map_from_range(self, value: float, val_range: tuple) -> float:
        """
        将实际值映射到标准化范围 [-1, 1]

        Args:
            value: 实际值
            val_range: 实际范围 (min, max)

        Returns:
            标准化值
        """
        min_val, max_val = val_range
        return 2 * (value - min_val) / (max_val - min_val) - 1

    def render(self, mode: str = "human"):
        """渲染环境状态"""
        print(f"\n=== Episode {self.episode_cnt}, Step {self.current_step} ===")

        for i in range(self.num_groups):
            stats = self.battery_system.get_group_stats(i)
            # 获取组内温差
            start_idx = i * self.num_batteries_per_group
            end_idx = start_idx + self.num_batteries_per_group
            group_temps = [
                b.get_core_temperature()
                for b in self.battery_system.batteries[start_idx:end_idx]
            ]
            temp_diff = max(group_temps) - min(group_temps)

            print(
                f"Group {i}: "
                f"Core={stats['avg_core']:.2f}K (Δ={temp_diff:.2f}K), "
                f"Top={stats['avg_top']:.2f}K, "
                f"Bottom={stats['avg_bot']:.2f}K, "
                f"V={stats['total_voltage']:.2f}V, "
                f"I={stats['current']:.2f}A"
            )

        if self.last_action:
            print(
                f"Action: inlet_temp={self.last_action[0]:.2f}K, flow_rate={self.last_action[1]:.2f}m/s"
            )

        if self.allrew:
            print(f"Reward: {self.allrew[-1]:.4f}")


def make_env(
    num_batteries_per_group: int = 13,
    num_groups: int = 4,
    episode_steps: int = 512,
    log_path: str = "",
    con: bool = True,
    num_train_envs: int = 8,
    num_test_envs: int = 4,
    use_subproc: bool = True,
    debug: bool = False,
):
    """
    创建向量化环境，支持多进程并行训练

    Args:
        num_batteries_per_group: 每组电池数量
        num_groups: 电池组数量
        episode_steps: 每个episode的最大步数
        log_path: 日志保存路径
        con: 是否使用连续动作空间
        num_train_envs: 训练环境数量
        num_test_envs: 测试环境数量
        use_subproc: 是否使用SubprocVectorEnv
        debug: 是否开启调试输出

    Returns:
        main_env: 主环境实例（用于获取配置）
        train_envs: 训练用向量化环境
        test_envs: 测试用向量化环境
    """

    def _select_env(evaluate: bool = False, env_idx: int = 0, seed: int = None):
        """创建独立的环境实例"""
        env = MultiBatteryEnv(
            num_batteries_per_group=num_batteries_per_group,
            num_groups=num_groups,
            max_steps=episode_steps,
            log_path=log_path,
            con=con,
            env_index=env_idx,
            debug=debug,
        )
        if seed is not None:
            env.reset(seed=seed)
        return env

    # 创建向量化环境
    if use_subproc and num_train_envs > 1:
        # 使用多进程
        train_envs = SubprocVectorEnv(
            [lambda i=i: _select_env(env_idx=i, seed=i) for i in range(num_train_envs)],
            wait_num=num_train_envs // 2,
            timeout=0.1,
        )
        test_envs = SubprocVectorEnv(
            [
                lambda i=i: _select_env(env_idx=num_train_envs + i, seed=num_train_envs + i)
                for i in range(num_test_envs)
            ],
            wait_num=num_test_envs // 2,
            timeout=0.1,
        )
    else:
        # 使用单进程
        train_envs = DummyVectorEnv(
            [lambda i=i: _select_env(env_idx=i, seed=i) for i in range(max(1, num_train_envs))]
        )
        test_envs = DummyVectorEnv(
            [
                lambda i=i: _select_env(env_idx=num_train_envs + i, seed=num_train_envs + i)
                for i in range(max(1, num_test_envs))
            ]
        )

    # 主环境实例
    main_env = _select_env(env_idx=0, seed=0)

    logger.info(
        f"环境创建完成: {num_groups}组×{num_batteries_per_group}电池, "
        f"训练环境={num_train_envs}, 测试环境={num_test_envs}"
    )

    return main_env, train_envs, test_envs

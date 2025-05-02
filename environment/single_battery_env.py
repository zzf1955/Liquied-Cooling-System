import numpy as np
from gymnasium import spaces
from BatteryEnv.single_battery_module import SingleBattery as SB
import gymnasium as gym
from tianshou.env import SubprocVectorEnv
import torch


class SingleBatteryEnv(gym.Env):
    def __init__(self, 
                 episode_steps=512, 
                 max_current=30,  min_current=0, 
                 min_voltage=2.5,  max_voltage=3.65, 
                 min_inlet_temp = 270, max_inlet_temp = 330,
                 min_flow_rate = 0, max_flow_rate = 5,
                 env_temp=298, 
                 temp_bound = 5,
                 change_steps=128):
        super(SingleBatteryEnv, self).__init__()
        
        # 初始化新版本的电池模块，使用真实参数
        self.battery = SB()  # 使用默认参数，已经设置为真实电池参数

        # 动作空间定义：冷却液入口温度（270到330开尔文）和流速（0到5 m/s）
        self.action_space = spaces.Box(low=np.array([min_inlet_temp, min_flow_rate], dtype=np.float32), 
                                       high=np.array([max_inlet_temp, max_flow_rate], dtype=np.float32), dtype=np.float32)

        # 状态空间定义：核心温度、顶部表面平均温度、底部表面平均温度、电池电流、电池电压
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 2.5], dtype=np.float32), 
            high=np.array([500, 500, 500, 30, 3.65], dtype=np.float32), 
            dtype=np.float32
        )

        # 环境温度（室温）
        self.environment_temp = env_temp  # 开尔文
        self.temp_bound = temp_bound
        self.min_inlet_temp = min_inlet_temp
        self.max_inlet_temp = max_inlet_temp
        self.min_flow_rate = min_flow_rate
        self.max_flow_rate = max_flow_rate

        # episode 参数
        self.current_step = 0
        self.max_steps = episode_steps
        self.change_steps = change_steps

        # 最大和最小电流
        self.max_current = max_current
        self.min_current = min_current

        # 最大和最小电压
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage

    def get_action(self, action):
        # 将归一化的action从[-1,1]线性映射到实际动作范围
        # action: [-1, 1] 的numpy数组或PyTorch张量
        # 返回: 物理动作参数 [温度, 流速]
        
        # 计算实际范围
        temp_range = self.max_inlet_temp - self.min_inlet_temp
        flow_range = self.max_flow_rate - self.min_flow_rate
        
        # 线性映射公式：scaled = (action + 1) * range / 2 + min
        scaled_temp = (action[0] + 1.0) * temp_range / 2.0 + self.min_inlet_temp
        scaled_flow = (action[1] + 1.0) * flow_range / 2.0 + self.min_flow_rate
        
        # 处理PyTorch张量输入（兼容性）
        if isinstance(action, torch.Tensor):
            return torch.tensor([scaled_temp, scaled_flow], dtype=torch.float32, device=self.device)
        # 默认返回numpy数组
        return np.array([scaled_temp, scaled_flow], dtype=np.float32)
        

    def step(self, action):
        # action = self.get_action(action)
        # 动作：冷却液入口温度和流速
        inlet_temp, flow_rate = action

        # 设置冷却参数并运行电池
        self.battery.set_action(inlet_temp, flow_rate)
        self.battery.run(t_seconds=1)

        # 每隔一定步数，随机设置电池输出
        # 建议采用动态电流波动（更常见场景）
        if self.current_step % self.change_steps == 0:
            # 改为正态分布更符合实际
            self.battery.current = np.clip(
                np.random.normal(loc=25, scale=2.5),  # 均值25A，标准差2.5A
                self.min_current, 
                self.max_current
            )

        # 获取当前状态
        core_temperature = self.battery.get_core_temperature()  # 核心温度
        top_surface_avg_temp = self.battery.get_top_surface_average_temperature()  # 顶部表面平均温度
        bottom_surface_avg_temp = self.battery.get_bottom_surface_average_temperature()  # 底部表面平均温度
        current = self.battery.current  # 当前电流
        voltage = self.battery.get_voltage()  # 当前电压
        
        state = np.array([core_temperature, top_surface_avg_temp, bottom_surface_avg_temp, current, voltage], dtype=np.float32)

        # 计算温度差
        delta_temp = abs(core_temperature - self.environment_temp)

        # 指数计算
        reward = np.exp(-delta_temp / 4)

        # 更新计步器
        self.current_step += 1

        # 原有温度终止条件
        temp_terminated = not ( self.environment_temp - self.temp_bound < core_temperature and 
                          core_temperature < self.environment_temp+self.temp_bound)
        # 新增电压终止条件
        voltage_terminated = voltage < self.min_voltage or voltage > self.max_voltage
        done = temp_terminated or voltage_terminated

        # 判断截断条件：步数超过max_steps
        truncated = bool(self.current_step >= self.max_steps)

        # 添加调试信息
        info = {
            'core_temp': core_temperature,
            'top_temp': top_surface_avg_temp,
            'bottom_temp': bottom_surface_avg_temp,
            'current': current,
            'voltage': voltage
        }
        
        return state, reward, done, truncated, info

    def reset(self, seed=None, randomize_current=False):
        # 重置环境状态
        super().reset(seed=seed)
        np.random.seed(seed)

        self.battery.inlet_temp = self.environment_temp
        self.battery.flow_rate = 0.1  # 初始化流速为默认值
        self.current_step = 0

        if randomize_current:
            # 随机设置电流
            self.battery.current = np.random.uniform(self.min_current, self.max_current)
        else:
            # 重置电流为0
            self.battery.current = 0

        # 重置电池温度分布
        self.battery.reset()

        # 返回初始状态：核心温度、顶部表面温度、底部表面温度、电流、电压
        initial_state = np.array([
            self.environment_temp, 
            self.environment_temp, 
            self.environment_temp,
            self.battery.current,
            self.battery.get_voltage()
        ], dtype=np.float32)
        
        # 添加调试信息
        info = {}
        
        return initial_state, info

    def render(self):
        core_temp = self.battery.get_core_temperature()
        top_surface_temp = self.battery.get_top_surface_average_temperature()
        bottom_surface_temp = self.battery.get_bottom_surface_average_temperature()
        current = self.battery.current
        voltage = self.battery.get_voltage()
        inlet_temp, flow_rate = self.battery.get_action()

        # 输出简洁的信息
        print(f"Core Temp: {core_temp:.2f} K, Top: {top_surface_temp:.2f} K, Bottom: {bottom_surface_temp:.2f} K, "
              f"Current: {current:.2f} A, Voltage: {voltage:.2f} V, Action: [Inlet: {inlet_temp:.2f} K, Flow: {flow_rate:.2f} m/s]")

def make_env(episode_steps=512, 
            max_current=30, 
            min_current=0, 
            min_voltage=2.5, 
            max_voltage=3.65, 
            env_temp=298, 
            change_steps=128):

    env = SingleBatteryEnv(episode_steps=episode_steps,
                           min_current=min_current,
                           max_current=max_current,
                           min_voltage=min_voltage,
                           max_voltage=max_voltage,
                           env_temp=env_temp,
                           change_steps=change_steps)
    
    def _select_env(evaluate = False):
        return env

    env = _select_env()

    train_envs = SubprocVectorEnv(
        [lambda: _select_env() for _ in range(1)])
    test_envs = SubprocVectorEnv(
        [lambda: _select_env(True) for _ in range(1)])
    
    return env,train_envs,test_envs


def test_single_battery_env():
    env = SingleBatteryEnv(episode_steps=512, max_current=10, min_current=0, env_temp=298, change_steps=128)
    initial_state, _ = env.reset(randomize_current=True)
    print(f"Initial State: Core Temp: {initial_state[0]:.2f} K, Top Surface Temp: {initial_state[1]:.2f} K, "
          f"Bottom Surface Temp: {initial_state[2]:.2f} K, Current: {initial_state[3]:.2f} A, Voltage: {initial_state[4]:.2f} V")

    steps = 9999999999999999  # 运行多少次迭代
    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")

        # 动作：使用当前环境设置的动作参数
        action = env.battery.get_action()

        # 运行模拟
        state, reward, terminated, truncated, info = env.step(action)

        # 显示当前状态和动作
        env.render()
        print(f"Reward: {reward:.4f}")

        # 检查是否终止
        if terminated or truncated:
            print("Environment reached terminal state.")
            break

        # 等待用户输入
        user_input = input("Press Enter to continue, or 'b' to modify action and current: ").strip()

        if user_input.lower() == 'b':
            try:
                inlet_temp = float(input("Enter new inlet cooling temperature (K): "))
                flow_rate = float(input("Enter new cooling flow rate (m/s): "))
                current = float(input("Enter new output current (A): "))

                # 更新电池的动作和输出电流
                env.battery.set_action(inlet_temp, flow_rate)
                env.battery.current = current

            except ValueError:
                print("Invalid input, using previous values.")
        
        elif user_input == '':
            continue

    print("Test completed.")

if __name__ == 'main':
    test_single_battery_env()

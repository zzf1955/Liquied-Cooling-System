import numpy as np
from gymnasium import spaces
from single_battery_module import SingleBattery as SB
import gymnasium as gym

class SingleBatteryEnv(gym.Env):
    def __init__(self, episode_steps=512, max_current=10, min_current=0, env_temp=298, change_steps=128):
        super(SingleBatteryEnv, self).__init__()
        self.battery = SB()  # 初始化电池

        # 动作空间定义：冷却液入口温度（270到330开尔文）和流速（0到5 m/s）
        self.action_space = spaces.Box(low=np.array([270, 0], dtype=np.float32), high=np.array([330, 5], dtype=np.float32), dtype=np.float32)

        # 状态空间定义：核心温度、顶部表面平均温度、电池电流
        self.observation_space = spaces.Box(low=np.array([0, 0, 0], dtype=np.float32), high=np.array([500, 500, 10], dtype=np.float32), dtype=np.float32)

        # 环境温度（室温）
        self.environment_temp = env_temp  # 开尔文

        # episode 参数
        self.current_step = 0
        self.max_steps = episode_steps
        self.change_steps = change_steps

        # 最大和最小电流
        self.max_current = max_current
        self.min_current = min_current

    def step(self, action):
        # 动作：冷却液入口温度和流速
        inlet_temp, flow_rate = action

        # 设置冷却参数并运行电池
        self.battery.set_action(inlet_temp, flow_rate)
        self.battery.run(t_seconds=1)

        # 每隔一定步数，随机设置电池输出
        if self.current_step % self.change_steps == 0:
            self.battery.current = np.random.uniform(self.min_current, self.max_current)

        # 获取当前状态
        core_temperature = self.battery.get_core_temperature()  # 核心温度
        top_surface_avg_temp = self.battery.get_top_surface_average_temperature()  # 顶部表面平均温度
        current = self.battery.current  # 当前电流
        state = np.array([core_temperature, top_surface_avg_temp, current], dtype=np.float32)

        # 计算温度差
        delta_temp = abs(core_temperature - self.environment_temp)

        # 奖励计算：温度差越小，奖励越高

        temp_threshold = 5

        #指数计算
        reward = np.exp(-delta_temp / 4 + 2.3) - temp_threshold

        #绝对温差计算
        #reward = delta_temp  + temp_threshold

        # 更新计步器
        self.current_step += 1

        # 判断终止条件：核心温度超过370或低于270
        terminated = bool(core_temperature > 350 or core_temperature < 270)

        # 判断截断条件：步数超过max_steps
        truncated = bool(self.current_step >= self.max_steps)

        # 添加调试信息
        info = {}
        
        return state, reward, terminated, truncated, info

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

        # 返回初始状态：核心温度、表面温度、电流
        initial_state = np.array([self.environment_temp, self.environment_temp, self.battery.current], dtype=np.float32)
        
        # 添加调试信息
        info = {}
        
        return initial_state, info

    def render(self):
        core_temp = self.battery.get_core_temperature()
        top_surface_temp = self.battery.get_top_surface_average_temperature()
        current = self.battery.current
        inlet_temp, flow_rate = self.battery.get_action()

        # 输出简洁的信息
        print(f"Core Temp: {core_temp:.2f} K, Top Surface Temp: {top_surface_temp:.2f} K, {current:.2f} A, Action: [{inlet_temp:.2f} K, {flow_rate:.2f} m/s]")

def test_single_battery_env():
    env = SingleBatteryEnv(episode_steps=512, max_current=10, min_current=0, env_temp=298, change_steps=128)
    initial_state, _ = env.reset(randomize_current=True)
    print(f"Initial State: Core Temp: {initial_state[0]:.2f} K, Top Surface Temp: {initial_state[1]:.2f} K, Current: {initial_state[2]:.2f} A")

    steps = 9999999999999999  # 运行多少次迭代
    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")

        # 动作：使用当前环境设置的动作参数
        action = env.battery.get_action()

        # 运行模拟
        state, reward, terminated, truncated, info = env.step(action)

        # 显示当前状态和动作
        env.render()

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

if __name__ == "__main__":
    test_single_battery_env()


if __name__ == "__main__":
    # 假设你的环境名为MyEnv
    env = SingleBatteryEnv(episode_steps=512)

    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import A2C

    # 检查环境是否兼容
    check_env(env)

    # 如果没有报错，你可以尝试训练
    model = A2C('MlpPolicy', env, verbose=2)
    model.learn(total_timesteps=512*512)

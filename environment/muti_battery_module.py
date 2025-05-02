# from BatteryEnv.single_battery_module import SingleBattery
from single_battery_module import SingleBattery
import matplotlib.pyplot as plt
import numpy as np

class MutiBattery:
    def __init__(self, num_batteries_per_group=13, num_groups=4, **kwargs):
        """
        初始化多电池系统
        
        参数:
            num_batteries_per_group: 每组电池的数量，默认为13
            num_groups: 电池组的数量，默认为4
            **kwargs: 传递给SingleBattery的其他参数
        """
        self.num_batteries_per_group = num_batteries_per_group
        self.num_groups = num_groups
        self.total_batteries = num_batteries_per_group * num_groups
        
        # 初始化电池（配置串联电压）
        kwargs['voltage'] = 3.2  # 强制单体电压一致
        self.batteries = [SingleBattery(**kwargs) for _ in range(self.total_batteries)]

        # 冷却液初始温度，入口温度
        self.initial_coolant_temp = kwargs.get('inlet_temp', 294)
            
        # 记录每个电池所属的组
        self.battery_groups = {}
        for i in range(self.total_batteries):
            group_id = i // num_batteries_per_group
            self.battery_groups[i] = group_id
            
        # 初始化温度记录
        self.temperature_history = [[] for _ in range(self.total_batteries)]
        self.coolant_history = [[] for _ in range(self.total_batteries)]
        self.time_steps = []
        self.total_time = 0  # 添加总时间变量

    def reset(self):
        """重置所有电池"""
        for battery in self.batteries:
            battery.reset()

    def run(self, t_seconds):
        """运行模拟指定时间"""
        num_steps = int(t_seconds / self.batteries[0].dt)
        for step in range(num_steps):
            # 第一个电池使用初始温度
            # self.batteries[0].inlet_temp = self.initial_coolant_temp
            
            # 按顺序处理每个电池
            for i in range(self.total_batteries):
                battery = self.batteries[i]
                
                # 更新热生成和温度分布
                battery.update_heat_generation()
                battery.temperature = battery.update_temperature_distribution()
                
                # 应用液冷散热
                battery.apply_cooling()
                
                # 冷却后再次更新温度分布
                battery.temperature = battery.diffuse_cooling()
                
                # 如果不是最后一个电池，将当前电池的出口温度传递给下一个电池
                if i < self.total_batteries - 1:
                    self.batteries[i + 1].inlet_temp = battery.inlet_temp
                
                # 记录温度
                self.temperature_history[i].append(battery.get_core_temperature())
                self.coolant_history[i].append(battery.inlet_temp)
            
            # 记录时间步（使用累积时间）
            current_time = self.total_time + step * self.batteries[0].dt
            self.time_steps.append(current_time)
            
            # 应用组内电池之间的热传导
            self.apply_inter_battery_heat_transfer()
        
        # 更新总时间
        self.total_time += t_seconds

    def apply_inter_battery_heat_transfer(self):
        """应用组内电池之间的前后方向热传导"""
        for group in range(self.num_groups):
            group_batteries = self.batteries[group*self.num_batteries_per_group : (group+1)*self.num_batteries_per_group]
            
            # 二维排列假设：每行13个电池，按前后方向排列
            for row in range(len(group_batteries)-1):
                front = group_batteries[row]
                rear = group_batteries[row+1]
                
                # 前电池后表面与后电池前表面传导
                front_surface = front.temperature[..., -2]  # 前电池的后表面
                rear_surface = rear.temperature[..., 1]    # 后电池的前表面
                
                temp_diff = front_surface - rear_surface
                heat_transfer = front.thermal_conductivity * temp_diff / front.cell_length * front.dt
                
                front.temperature[..., -2] -= heat_transfer / (front.density * front.specific_heat)
                rear.temperature[..., 1] += heat_transfer / (rear.density * rear.specific_heat)

    def get_all_core_temperatures(self):
        """获取所有电池的核心温度"""
        return [battery.get_core_temperature() for battery in self.batteries]

    def get_all_top_surface_average_temperatures(self):
        """获取所有电池顶面的平均温度"""
        return [battery.get_top_surface_average_temperature() for battery in self.batteries]
    
    def get_all_bottom_surface_average_temperatures(self):
        """获取所有电池底面的平均温度"""
        return [battery.get_bottom_surface_average_temperature() for battery in self.batteries]
    
    def get_group_average_temperatures(self):
        """获取每组电池的平均核心温度"""
        group_temps = []
        for group in range(self.num_groups):
            start_idx = group * self.num_batteries_per_group
            end_idx = start_idx + self.num_batteries_per_group
            group_batteries = self.batteries[start_idx:end_idx]
            avg_temp = sum(b.get_core_temperature() for b in group_batteries) / len(group_batteries)
            group_temps.append(avg_temp)
        return group_temps

    def get_group_average_voltages(self):
        """获取每组电池的总电压（串联）"""
        group_voltages = []
        for group in range(self.num_groups):
            start_idx = group * self.num_batteries_per_group
            end_idx = start_idx + self.num_batteries_per_group
            group_batteries = self.batteries[start_idx:end_idx]
            # 串联时总电压为所有电池电压之和
            total_voltage = sum(b.get_voltage() for b in group_batteries)
            group_voltages.append(total_voltage)
        return group_voltages

    def get_group_current(self, group_idx):
        """获取指定组的电流（串联时组内所有电池电流相同）"""
        start_idx = group_idx * self.num_batteries_per_group
        # 返回组内任意一个电池的电流即可
        return self.batteries[start_idx].current

    def plot_temperature_history(self, battery_indices=None):
        """绘制温度变化曲线"""
        if battery_indices is None:
            battery_indices = [0, 13, 26, 39]  # 默认显示每组的第一个电池
            
        plt.figure(figsize=(12, 6))
        
        # 绘制核心温度
        plt.subplot(1, 2, 1)
        for i in battery_indices:
            plt.plot(self.time_steps, self.temperature_history[i], 
                    label=f'battery {i+1} core temperature')
        plt.xlabel('time (s)')
        plt.ylabel('temperature (K)')
        plt.title('core temperaterature change')
        plt.legend()
        plt.grid(True)
        
        # 绘制冷却液温度
        plt.subplot(1, 2, 2)
        for i in battery_indices:
            plt.plot(self.time_steps, self.coolant_history[i], 
                    label=f'battery {i+1} coolant temperature')
        plt.xlabel('time (s)')
        plt.ylabel('temperature (K)')
        plt.title('coolant temperaterature change')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def test_muti_battery():
    # 创建4组电池，每组13个
    muti_battery = MutiBattery(num_batteries_per_group=13, num_groups=4)

    t_seconds = 1  # 每次运行的模拟时间
    # steps = 999999999  # 运行多少次迭代
    steps = 2000
    
    # flow_rate_state = [0.2 * i for i in range(30)]
    flow_rate_state = [4]
    # flow_rate_state = [3]
    # inlet_temp_state = [288 + 0.4 * i for i in range(30)]
    inlet_temp_state = [293, 294, 293, 292, 291]

    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")
        print(f"Current total simulation time: {muti_battery.total_time:.2f} seconds")        
        
        for i in range(muti_battery.total_batteries):
            battery = muti_battery.batteries[i]
            # battery.set_action(inlet_temp_state[step % 30], flow_rate_state[step % 30])
            battery.set_action(inlet_temp_state[int(step / 400)], flow_rate_state[0])

        # 运行模拟
        muti_battery.run(t_seconds)

        # 显示每个组的平均核心温度
        group_temps = muti_battery.get_group_average_temperatures()
        for i, temp in enumerate(group_temps):
            print(f"Group {i + 1} Average Core Temperature: {temp:.2f} K")
        
        # 显示选定的一些电池的详细信息
        selected_batteries = [0, 13, 26, 39]  # 每组的第一个电池
        for i in selected_batteries:
            battery = muti_battery.batteries[i]
            core_temp = battery.get_core_temperature()
            top_temp = battery.get_top_surface_average_temperature()
            bottom_temp = battery.get_bottom_surface_average_temperature()
            current = battery.current
            flow_rate = battery.flow_rate
            inlet_temp = battery.inlet_temp
            group_id = i // muti_battery.num_batteries_per_group
            
            print(f"Battery {i} (Group {group_id}): Core={core_temp:.2f}K, Top={top_temp:.2f}K, Bottom={bottom_temp:.6f}K, " 
                  f"Current={current:.6f}A, Flow={flow_rate:.2f}m/s, Inlet={inlet_temp:.6f}K")

        # 等待用户输入
        # user_input = input("Press Enter to continue, or 'b' to modify battery parameters: ").strip()

        # if user_input.lower() == 'b':
        #     # 允许用户修改组级别的参数
        #     for group in range(muti_battery.num_groups):
        #         try:
        #             print(f"--- Group {group + 1} Parameters ---")
        #             current = float(input(f"Enter output current for all batteries in Group {group + 1} (A): "))
        #             flow_rate = float(input(f"Enter cooling flow rate for all batteries in Group {group + 1} (m/s): "))
        #             inlet_temp = float(input(f"Enter inlet cooling temperature for all batteries in Group {group + 1} (K): "))
                    
        #             # 更新该组所有电池的参数
        #             start_idx = group * muti_battery.num_batteries_per_group
        #             end_idx = start_idx + muti_battery.num_batteries_per_group
        #             for i in range(start_idx, end_idx):
        #                 battery = muti_battery.batteries[i]
        #                 battery.current = current
        #                 battery.flow_rate = flow_rate
        #                 battery.inlet_temp = inlet_temp
        #         except ValueError:
        #             print("Invalid input, using previous values.")

        # elif user_input == '':
        #     continue

    # 绘制温度变化曲线
    muti_battery.plot_temperature_history()

if __name__ == "__main__":
    test_muti_battery()

'''
test input:
冷却液高温
0
0.1
298
0
0.1
298
0
0.1
330
0
0.1
298

高输出
0
0.1
298
10
0.1
298
0
0.1
298
0
0.1
298

冷却液降温
0
0.1
298
10
0.1
270
0
0.1
298
0
0.1
298

'''
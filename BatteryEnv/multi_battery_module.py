import numpy as np
import matplotlib.pyplot as plt
from BatteryEnv.single_battery_module import SingleBattery

class MultiBattery:
    def __init__(self, num_batteries_per_group=13, num_groups=4, env_temp=300, voltage=3.2):
        """
        初始化多电池系统

        参数:
            num_batteries_per_group: 每组电池的数量，默认为13
            num_groups: 电池组的数量，默认为4
            env_temp: 环境温度
            voltage: 电池电压
        """
        self.num_batteries_per_group = num_batteries_per_group
        self.num_groups = num_groups
        self.total_batteries = num_batteries_per_group * num_groups

        # 初始化电池（配置串联电压）
        self.batteries = [SingleBattery(voltage=voltage, env_temperature=env_temp) for _ in range(self.total_batteries)]

        # 冷却液初始温度，入口温度
        self.initial_coolant_temp = env_temp

        # 记录每个电池所属的组
        self.battery_groups = {}
        for i in range(self.total_batteries):
            group_id = i // num_batteries_per_group
            self.battery_groups[i] = group_id

        # 初始化温度记录
        self.temperature_history = [[] for _ in range(self.total_batteries)]
        self.coolant_history = [[] for _ in range(self.total_batteries)]
        self.time_steps = []

        self.total_time = 0.0  # 添加总时间变量

        self.reset_logs()
    
    def reset_logs(self):
        """重置数据记录器"""
        self.temperature_history = [[] for _ in range(self.total_batteries)]
        self.coolant_history = [[] for _ in range(self.total_batteries)]
        self.time_steps = []
        
    def reset(self):
        """级联重置所有电池状态"""
        for battery in self.batteries:
            battery.reset()
        self.total_time = 0.0
        self.reset_logs()
        
    def _apply_inter_battery_heat_transfer(self, dt):
        """
        修正后的组间传热逻辑：消除 100 倍误差
        """
        for group in range(self.num_groups):
            start = group * self.num_batteries_per_group
            
            for i in range(self.num_batteries_per_group - 1):
                front = self.batteries[start + i]
                rear = self.batteries[start + i + 1]
                
                # 表面温度提取
                # t_f = front.temperature[..., -2]  # 前电池后表面
                # t_r = rear.temperature[..., 1]   # 后电池前表面
                
                t_f = front.temperature[-2, 1:-1, 1:-1]  # 前电池后表面
                t_r = rear.temperature[1, 1:-1, 1:-1]   # 后电池前表面
                
                # 计算热量交换
                temp_diff = t_f - t_r
                
                # 物理计算：ΔT = alpha * dt / L^2 * (T_diff)
                # 这保证了热量交换在电池间是真实且守恒的
                alpha = front.thermal_conductivity / (front.density * front.specific_heat)
                temp_step = alpha * temp_diff * dt / (front.cell_length**2)
                
                front.temperature[-2, 1:-1, 1:-1] -= temp_step
                rear.temperature[1, 1:-1, 1:-1] += temp_step
                # front.temperature[..., -2] -= temp_step
                # rear.temperature[..., 1] += temp_step
    
    def run(self, t_seconds):
        """
        核心运行逻辑：物理计算与 RL 接口对齐
        """
        # 以第一个电池的物理步长为基准
        dt = self.batteries[0].dt
        num_steps = int(t_seconds / dt)
        
        for _ in range(num_steps):
            # 1. 串联冷却链条更新
            # 冷却液从第一个电池流向最后一个，入口温度实时流转
            current_coolant_in = self.batteries[0].inlet_temp 
            
            for i in range(self.total_batteries):
                battery = self.batteries[i]
                battery.inlet_temp = current_coolant_in
                
                # 执行物理原子操作
                battery.update_heat_generation()
                battery.update_temperature_distribution()
                
                # apply_cooling 立即返回该电池的出口温度，作为下一个的入口
                current_coolant_in = battery.apply_cooling()
                
                battery.diffuse_cooling()
            
            # 2. 组间传热 (每步物理步长平衡一次)
            self._apply_inter_battery_heat_transfer(dt)

        # 3. 数据采样：每个 RL Step 执行完记录一次（极大提升训练速度）
        self.total_time += t_seconds
        for i, battery in enumerate(self.batteries):
            self.temperature_history[i].append(battery.get_core_temperature())
            self.coolant_history[i].append(battery.inlet_temp)
        self.time_steps.append(self.total_time)
    
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
    
    def get_group_stats(self, group_idx):
        """获取指定组的聚合状态（均值/总和）"""
        start = group_idx * self.num_batteries_per_group
        end = start + self.num_batteries_per_group
        group_bs = self.batteries[start:end]
        
        return {
            'avg_core': np.mean([battery.get_core_temperature() for battery in group_bs]),
            'avg_top': np.mean([battery.get_top_surface_average_temperature() for battery in group_bs]),
            'avg_bot': np.mean([battery.get_bottom_surface_average_temperature() for battery in group_bs]),
            'total_voltage': sum(battery.get_voltage() for battery in group_bs),
            'current': group_bs[0].current
        }
        
    def set_group_controls(self, flow_rate, inlet_temp):
        """统一设置所有电池的流速，并设置第一个电池的入口温度"""
        for battery in self.batteries:
            battery.flow_rate = flow_rate
        self.batteries[0].inlet_temp = inlet_temp
    
    def set_group_current(self, group_idx,current):
        start_idx = group_idx * self.num_batteries_per_group
        for battery_idx in range(start_idx,start_idx+self.num_batteries_per_group):
            self.batteries[battery_idx].current = current
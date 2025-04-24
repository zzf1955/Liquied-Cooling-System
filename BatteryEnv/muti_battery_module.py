from BatteryEnv.single_battery_module import SingleBattery

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

        # 冷却流道温度链式传递
        self.coolant_temp_chain = [kwargs.get('inlet_temp', 294)] * self.num_groups
            
        # 记录每个电池所属的组
        self.battery_groups = {}
        for i in range(self.total_batteries):
            group_id = i // num_batteries_per_group
            self.battery_groups[i] = group_id

    def reset(self):
        """重置所有电池"""
        for battery in self.batteries:
            battery.reset()

    def run(self, t_seconds):
        """运行模拟指定时间"""
        num_steps = int(t_seconds / self.batteries[0].dt)
        for _ in range(num_steps):
            self._update_coolant_inlet_temps()
            # 更新每个电池的热生成和温度分布
            for battery in self.batteries:
                battery.update_heat_generation()
                battery.temperature = battery.update_temperature_distribution()
                battery.apply_cooling()
            
            # 应用组内电池之间的热传导
            self.apply_inter_battery_heat_transfer()

    def _update_coolant_inlet_temps(self):
        """组间冷却液温度链式传递"""
        for group in range(self.num_groups):
            group_batteries = self.batteries[group*self.num_batteries_per_group : (group+1)*self.num_batteries_per_group]
            
            # 获取组出口温度
            last_battery = group_batteries[-1]
            outlet_temp = last_battery.coolant_temp_accumulator
            
            # 下一组入口温度（循环冷却时需重置）
            next_group = (group + 1) % self.num_groups
            self.coolant_temp_chain[next_group] = outlet_temp
            
            # 更新组内所有电池入口温度
            for battery in group_batteries:
                battery.inlet_temp = self.coolant_temp_chain[group]

    def apply_inter_battery_heat_transfer(self):
        """应用组内电池之间的前后方向热传导"""
        for group in range(self.num_groups):
            group_batteries = self.batteries[group*self.num_batteries_per_group : (group+1)*self.num_batteries_per_group]
            
            # 二维排列假设：每行13个电池，按前后方向排列
            for row in range(len(group_batteries)-1):
                front = group_batteries[row]
                rear = group_batteries[row+1]
                
                # 前电池后表面与后电池前表面传导
                front_surface = front.temperature[..., -2]  # 后表面
                rear_surface = rear.temperature[..., 1]    # 前表面
                
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

def test_muti_battery():
    # 创建4组电池，每组13个
    muti_battery = MutiBattery(num_batteries_per_group=13, num_groups=4)

    t_seconds = 10  # 每次运行的模拟时间
    steps = 999999999  # 运行多少次迭代

    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")

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
            
            print(f"Battery {i} (Group {group_id}): Core={core_temp:.2f}K, Top={top_temp:.2f}K, Bottom={bottom_temp:.2f}K, " 
                  f"Current={current:.2f}A, Flow={flow_rate:.2f}m/s, Inlet={inlet_temp:.2f}K")

        # 等待用户输入
        user_input = input("Press Enter to continue, or 'b' to modify battery parameters: ").strip()

        if user_input.lower() == 'b':
            # 允许用户修改组级别的参数
            for group in range(muti_battery.num_groups):
                try:
                    print(f"--- Group {group + 1} Parameters ---")
                    current = float(input(f"Enter output current for all batteries in Group {group + 1} (A): "))
                    flow_rate = float(input(f"Enter cooling flow rate for all batteries in Group {group + 1} (m/s): "))
                    inlet_temp = float(input(f"Enter inlet cooling temperature for all batteries in Group {group + 1} (K): "))
                    
                    # 更新该组所有电池的参数
                    start_idx = group * muti_battery.num_batteries_per_group
                    end_idx = start_idx + muti_battery.num_batteries_per_group
                    for i in range(start_idx, end_idx):
                        battery = muti_battery.batteries[i]
                        battery.current = current
                        battery.flow_rate = flow_rate
                        battery.inlet_temp = inlet_temp
                except ValueError:
                    print("Invalid input, using previous values.")

        elif user_input == '':
            continue

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
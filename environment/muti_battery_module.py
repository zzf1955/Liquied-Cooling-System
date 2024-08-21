from single_battery_module import SingleBattery

class MutiBattery:
    def __init__(self, num_batteries, **kwargs):
        self.num_batteries = num_batteries
        self.batteries = [SingleBattery(**kwargs) for _ in range(num_batteries)]

    def run(self, t_seconds):
        num_steps = int(t_seconds / self.batteries[0].dt)
        for _ in range(num_steps):
            for battery in self.batteries:
                battery.update_heat_generation()
                battery.temperature = battery.update_temperature_distribution()
                battery.apply_cooling()
            
            self.apply_inter_battery_heat_transfer()

    def apply_inter_battery_heat_transfer(self):
        # 遍历相邻电池进行热交换
        for i in range(self.num_batteries - 1):
            left_battery = self.batteries[i]
            right_battery = self.batteries[i + 1]
            
            # 获取两个电池相邻侧面的温度
            left_side_temp = left_battery.temperature[:, -2, 1:-1]
            right_side_temp = right_battery.temperature[:, 1, 1:-1]
            
            # 计算温度差和热量交换
            temp_diff = left_side_temp - right_side_temp
            heat_transfer = (left_battery.thermal_conductivity * temp_diff) / left_battery.cell_length * left_battery.dt
            
            # 更新左右相邻电池的温度
            left_battery.temperature[:, -2, 1:-1] -= heat_transfer / (left_battery.density * left_battery.specific_heat)
            right_battery.temperature[:, 1, 1:-1] += heat_transfer / (right_battery.density * right_battery.specific_heat)

    def get_all_core_temperatures(self):
        return [battery.get_core_temperature() for battery in self.batteries]

    def get_all_top_surface_average_temperatures(self):
        return [battery.get_top_surface_average_temperature() for battery in self.batteries]

def test_muti_battery():
    num_batteries = 4  # 假设系统中有3个电池
    muti_battery = MutiBattery(num_batteries, side_length=0.1, cell_length=0.01)

    t_seconds = 10  # 每次运行的模拟时间
    steps = 999999999  # 运行多少次迭代

    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")

        # 运行模拟
        muti_battery.run(t_seconds)

        # 显示每个电池的核心温度和上表面平均温度
        for i, battery in enumerate(muti_battery.batteries):
            core_temp = battery.get_core_temperature()
            top_temp = battery.get_top_surface_average_temperature()
            current = battery.current
            flow_rate = battery.flow_rate
            inlet_temp = battery.inlet_temp
            print(f"Battery {i + 1}: Core Temp = {core_temp:.2f} K, Top Surface Avg Temp = {top_temp:.2f} K, "
                f"Current = {current:.2f} A, Flow Rate = {flow_rate:.2f} m/s, Inlet Temp = {inlet_temp:.2f} K")

        # 等待用户输入
        user_input = input("Press Enter to continue, or 'b' to modify battery parameters: ").strip()

        if user_input.lower() == 'b':
            for i, battery in enumerate(muti_battery.batteries):
                try:
                    current = float(input(f"Enter new output current for Battery {i + 1} (A): "))
                    flow_rate = float(input(f"Enter new cooling flow rate for Battery {i + 1} (m/s): "))
                    inlet_temp = float(input(f"Enter new inlet cooling temperature for Battery {i + 1} (K): "))
                    
                    # 更新电池参数
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
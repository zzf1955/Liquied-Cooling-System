import sys
sys.path.insert(0, '/mnt/g/github_project/Liquied-Cooling-System')

import numpy as np
import matplotlib.pyplot as plt
from BatteryEnv.multi_battery_module import MultiBattery

def test_serial_thermal_gradient():
    # 1. 初始化 4组 * 13个 = 52个电池的系统
    mb = MultiBattery(num_batteries_per_group=13, num_groups=4)
    
    # 2. 设置实验工况
    current = 40.0      # 高电流生热
    flow_rate = 2.0     # 2 m/s 流速
    inlet_temp = 290.0  # 初始冷水 17°C
    
    # 为所有电池设置相同的电流
    for i in range(mb.num_groups):
        mb.set_group_current(i, current)
        
    # 应用控制：统一流速，设置总入口温度
    mb.set_group_controls(flow_rate, inlet_temp)

    # 3. 运行长达 300 秒的模拟（每个 step 5s，共 60 步）
    simulation_steps = 60
    for _ in range(simulation_steps):
        mb.run(t_seconds=5)

    # 4. 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # 图 A：52个电池在模拟结束时的核心温度分布
    final_temps = [b.get_core_temperature() for b in mb.batteries]
    ax1.plot(range(52), final_temps, 'o-', markersize=4, label='Core Temp')
    ax1.axhline(inlet_temp, color='r', linestyle='--', label='Initial Coolant Temp')
    ax1.set_title("Temperature Gradient across 52 Batteries (End of 300s)")
    ax1.set_xlabel("Battery Index (0 to 51)")
    ax1.set_ylabel("Temperature (K)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图 B：代表性电池的时间演化曲线 (第1, 13, 26, 52个)
    sample_indices = [0, 12, 25, 51]
    for idx in sample_indices:
        ax2.plot(mb.time_steps, mb.temperature_history[idx], label=f'Battery {idx}')
    
    ax2.set_title("Thermal Evolution of Representative Batteries")
    ax2.set_xlabel("Total Time (s)")
    ax2.set_ylabel("Core Temperature (K)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_serial_thermal_gradient()
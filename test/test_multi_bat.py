import sys
sys.path.insert(0, '/mnt/g/github_project/Liquied-Cooling-System')

import numpy as np
import matplotlib.pyplot as plt
import copy
from BatteryEnv.multi_battery_module import MutiBattery

def run_comparison_simulation(duration_steps=500):
    # 1. 初始化两个完全相同的系统
    # 减少电池数量以便清晰观察传导效果 (1组5个)
    sys_bugged = MutiBattery(num_batteries_per_group=5, num_groups=1)
    sys_fixed = MutiBattery(num_batteries_per_group=5, num_groups=1)

    # 2. 定义错误逻辑函数 (模拟修改前)
    def bugged_transfer(self):
        for group in range(self.num_groups):
            group_batteries = self.batteries[group*self.num_batteries_per_group : (group+1)*self.num_batteries_per_group]
            for row in range(len(group_batteries)-1):
                front, rear = group_batteries[row], group_batteries[row+1]
                temp_diff = front.temperature[..., -2] - rear.temperature[..., 1]
                # 错误：分母缺少 cell_length
                heat_transfer = front.thermal_conductivity * temp_diff / front.cell_length * front.dt
                front.temperature[..., -2] -= heat_transfer / (front.density * front.specific_heat)
                rear.temperature[..., 1] += heat_transfer / (rear.density * rear.specific_heat)

    # 3. 实验设置：让第一个电池（Battery 0）产生剧烈热量，关闭液冷以纯看传导
    for sys in [sys_bugged, sys_fixed]:
        for b in sys.batteries:
            b.current = 0 
            b.flow_rate = 0 # 关闭冷却，纯观察热传导
        sys.batteries[0].current = 80.0 # 只有第一个电池在发热

    # 记录数据
    results_bugged = []
    results_fixed = []

    print("正在模拟错误逻辑 (Bugged)...")
    # 临时替换方法
    original_method = MutiBattery.apply_inter_battery_heat_transfer
    MutiBattery.apply_inter_battery_heat_transfer = bugged_transfer
    for _ in range(duration_steps):
        sys_bugged.run(t_seconds=1)
        results_bugged.append(sys_bugged.get_all_core_temperatures()[:5])

    print("正在模拟修正逻辑 (Fixed)...")
    # 恢复原有的修正后方法
    MutiBattery.apply_inter_battery_heat_transfer = original_method
    for _ in range(duration_steps):
        sys_fixed.run(t_seconds=1)
        results_fixed.append(sys_fixed.get_all_core_temperatures()[:5])

    # 4. 绘图对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    time_axis = np.arange(duration_steps)
    results_bugged = np.array(results_bugged)
    results_fixed = np.array(results_fixed)

    for i in range(5):
        ax1.plot(time_axis, results_bugged[:, i], label=f'Battery {i}')
        ax2.plot(time_axis, results_fixed[:, i], label=f'Battery {i}')

    ax1.set_title("Before Fix: Bugged Heat Transfer\n(100x Slower than Reality)", fontsize=12)
    ax2.set_title("After Fix: Physical Heat Transfer\n(Correct Scaling)", fontsize=12)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel("Simulation Steps (seconds)")
        ax.set_ylabel("Core Temperature (K)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison_simulation()
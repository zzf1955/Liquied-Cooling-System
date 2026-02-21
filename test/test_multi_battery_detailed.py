"""
详细测试多电池物理模型
展示：1. 不同流速对各电池组温度的影响
     2. 不同入口温度对各电池组温度的影响
     3. 各电池的核心温度、顶部温度、底部温度
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

from BatteryEnv.multi_battery_module import MutiBattery
import matplotlib.pyplot as plt


def run_experiment(inlet_temp, flow_rate, current=30.0, duration=60):
    """运行一次实验，返回各电池的温度数据"""
    mb = MutiBattery(num_batteries_per_group=13, num_groups=4, env_temp=300)

    for battery in mb.batteries:
        battery.current = current
        battery.flow_rate = flow_rate
    mb.batteries[0].inlet_temp = inlet_temp

    mb.run(t_seconds=duration)

    return mb


def analyze_temperatures(mb):
    """分析并打印温度数据"""
    total = mb.total_batteries

    core_temps = mb.get_all_core_temperatures()
    top_temps = mb.get_all_top_surface_average_temperatures()
    bottom_temps = mb.get_all_bottom_surface_average_temperatures()

    print(f"\n{'='*80}")
    print(f"{'电池序号':^8} | {'核心温度':^10} | {'顶部温度':^10} | {'底部温度':^10} | {'入口温度':^10}")
    print(f"{'-'*80}")

    for i in range(total):
        # 获取入口温度历史记录
        inlet = mb.coolant_history[i][-1] if mb.coolant_history[i] else 0
        group = i // 13 + 1
        print(f"电池{i+1:2d}(组{group}) | {core_temps[i]:10.2f} | {top_temps[i]:10.2f} | {bottom_temps[i]:10.2f} | {inlet:10.2f}")

    print(f"{'-'*80}")

    # 按组统计
    print(f"\n{'组':^6} | {'平均核心':^10} | {'平均顶部':^10} | {'平均底部':^10} | {'核心标准差':^10}")
    print(f"{'-'*60}")
    for g in range(4):
        start = g * 13
        end = start + 13
        avg_core = np.mean(core_temps[start:end])
        avg_top = np.mean(top_temps[start:end])
        avg_bottom = np.mean(bottom_temps[start:end])
        std_core = np.std(core_temps[start:end])
        print(f"组{g+1}   | {avg_core:10.2f} | {avg_top:10.2f} | {avg_bottom:10.2f} | {std_core:10.2f}")

    return core_temps, top_temps, bottom_temps


def test_different_flow_rates():
    """测试不同流速的影响"""
    print("\n" + "="*80)
    print("实验1: 不同流速对温度的影响")
    print("入口温度=288K, 电流=30A, 持续时间=60s")
    print("="*80)

    results = {}
    for flow_rate in [0, 1, 3, 5]:
        print(f"\n>>> 流速 = {flow_rate} m/s")
        mb = run_experiment(inlet_temp=288, flow_rate=flow_rate, current=30.0, duration=60)
        core_temps, top_temps, bottom_temps = analyze_temperatures(mb)

        # 每组平均
        group_core = []
        for g in range(4):
            start = g * 13
            end = start + 13
            group_core.append(np.mean(core_temps[start:end]))
        results[flow_rate] = group_core

    # 打印对比表
    print("\n" + "="*80)
    print("流速对比总结")
    print("="*80)
    print(f"{'流速':^8} | {'组1':^10} | {'组2':^10} | {'组3':^10} | {'组4':^10} | {'总温差':^10}")
    print(f"{'-'*70}")
    for flow_rate, group_temps in results.items():
        total_diff = group_temps[-1] - group_temps[0]
        print(f"{flow_rate:>4}m/s | {group_temps[0]:10.2f} | {group_temps[1]:10.2f} | {group_temps[2]:10.2f} | {group_temps[3]:10.2f} | {total_diff:10.2f}")

    return results


def test_different_inlet_temps():
    """测试不同入口温度的影响"""
    print("\n" + "="*80)
    print("实验2: 不同入口温度对温度的影响")
    print("流速=3m/s, 电流=30A, 持续时间=60s")
    print("="*80)

    results = {}
    for inlet_temp in [285, 288, 290, 293]:
        print(f"\n>>> 入口温度 = {inlet_temp} K")
        mb = run_experiment(inlet_temp=inlet_temp, flow_rate=3.0, current=30.0, duration=60)
        core_temps, top_temps, bottom_temps = analyze_temperatures(mb)

        # 每组平均
        group_core = []
        for g in range(4):
            start = g * 13
            end = start + 13
            group_core.append(np.mean(core_temps[start:end]))
        results[inlet_temp] = group_core

    # 打印对比表
    print("\n" + "="*80)
    print("入口温度对比总结")
    print("="*80)
    print(f"{'入口温度':^10} | {'组1':^10} | {'组2':^10} | {'组3':^10} | {'组4':^10} | {'总温差':^10}")
    print(f"{'-'*70}")
    for inlet_temp, group_temps in results.items():
        total_diff = group_temps[-1] - group_temps[0]
        print(f"{inlet_temp:>8}K | {group_temps[0]:10.2f} | {group_temps[1]:10.2f} | {group_temps[2]:10.2f} | {group_temps[3]:10.2f} | {total_diff:10.2f}")

    return results


def test_different_currents():
    """测试不同电流的影响"""
    print("\n" + "="*80)
    print("实验3: 不同电流对温度的影响")
    print("入口温度=288K, 流速=3m/s, 持续时间=60s")
    print("="*80)

    results = {}
    for current in [0, 10, 20, 30]:
        print(f"\n>>> 电流 = {current} A")
        mb = run_experiment(inlet_temp=288, flow_rate=3.0, current=current, duration=60)
        core_temps, top_temps, bottom_temps = analyze_temperatures(mb)

        # 每组平均
        group_core = []
        for g in range(4):
            start = g * 13
            end = start + 13
            group_core.append(np.mean(core_temps[start:end]))
        results[current] = group_core

    # 打印对比表
    print("\n" + "="*80)
    print("电流对比总结")
    print("="*80)
    print(f"{'电流':^8} | {'组1':^10} | {'组2':^10} | {'组3':^10} | {'组4':^10} | {'总温差':^10}")
    print(f"{'-'*70}")
    for current, group_temps in results.items():
        total_diff = group_temps[-1] - group_temps[0]
        print(f"{current:>6}A | {group_temps[0]:10.2f} | {group_temps[1]:10.2f} | {group_temps[2]:10.2f} | {group_temps[3]:10.2f} | {total_diff:10.2f}")

    return results


def test_thermal_distribution():
    """测试单个电池内部温度分布"""
    print("\n" + "="*80)
    print("实验4: 单个电池内部温度分布")
    print("="*80)

    mb = run_experiment(inlet_temp=288, flow_rate=3.0, current=30.0, duration=60)

    # 选取几个代表性的电池
    test_batteries = [0, 12, 13, 25, 51]

    for idx in test_batteries:
        battery = mb.batteries[idx]
        core = battery.get_core_temperature()
        top = battery.get_top_surface_average_temperature()
        bottom = battery.get_bottom_surface_average_temperature()

        group = idx // 13 + 1
        local_idx = idx % 13 + 1
        print(f"电池{idx+1} (组{group}第{local_idx}个): 核心={core:.2f}K, 顶部={top:.2f}K, 底部={bottom:.2f}K, 温差(顶-底)={top-bottom:.2f}K")


def visualize_results():
    """可视化不同实验结果"""
    # 实验1: 不同流速
    print("\n正在运行可视化实验...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 不同流速
    flow_results = {}
    for flow_rate in [0, 1, 3, 5]:
        mb = run_experiment(inlet_temp=288, flow_rate=flow_rate, current=30.0, duration=60)
        core_temps = mb.get_all_core_temperatures()
        flow_results[flow_rate] = core_temps

    ax1 = axes[0, 0]
    for flow_rate, temps in flow_results.items():
        ax1.plot(range(1, 53), temps, label=f'flow={flow_rate}m/s', marker='o', markersize=3)
    ax1.set_xlabel('Battery Index')
    ax1.set_ylabel('Core Temperature (K)')
    ax1.set_title('Core Temperature vs Flow Rate')
    ax1.legend()
    ax1.grid(True)

    # 不同入口温度
    temp_results = {}
    for inlet_temp in [285, 288, 290, 293]:
        mb = run_experiment(inlet_temp=inlet_temp, flow_rate=3.0, current=30.0, duration=60)
        core_temps = mb.get_all_core_temperatures()
        temp_results[inlet_temp] = core_temps

    ax2 = axes[0, 1]
    for inlet_temp, temps in temp_results.items():
        ax2.plot(range(1, 53), temps, label=f'inlet={inlet_temp}K', marker='o', markersize=3)
    ax2.set_xlabel('Battery Index')
    ax2.set_ylabel('Core Temperature (K)')
    ax2.set_title('Core Temperature vs Inlet Temperature')
    ax2.legend()
    ax2.grid(True)

    # 不同电流
    current_results = {}
    for current in [0, 10, 20, 30]:
        mb = run_experiment(inlet_temp=288, flow_rate=3.0, current=current, duration=60)
        core_temps = mb.get_all_core_temperatures()
        current_results[current] = core_temps

    ax3 = axes[1, 0]
    for current, temps in current_results.items():
        ax3.plot(range(1, 53), temps, label=f'I={current}A', marker='o', markersize=3)
    ax3.set_xlabel('Battery Index')
    ax3.set_ylabel('Core Temperature (K)')
    ax3.set_title('Core Temperature vs Current')
    ax3.legend()
    ax3.grid(True)

    # 组间温差分析
    ax4 = axes[1, 1]
    mb = run_experiment(inlet_temp=288, flow_rate=3.0, current=30.0, duration=60)
    core_temps = mb.get_all_core_temperatures()
    top_temps = mb.get_all_top_surface_average_temperatures()
    bottom_temps = mb.get_all_bottom_surface_average_temperatures()

    battery_idx = range(1, 53)
    ax4.plot(battery_idx, core_temps, label='Core', marker='o', markersize=3)
    ax4.plot(battery_idx, top_temps, label='Top Surface', marker='s', markersize=3)
    ax4.plot(battery_idx, bottom_temps, label='Bottom Surface', marker='^', markersize=3)

    # 标记组边界
    for g in range(1, 4):
        ax4.axvline(x=g*13+0.5, color='gray', linestyle='--', alpha=0.5)

    ax4.set_xlabel('Battery Index')
    ax4.set_ylabel('Temperature (K)')
    ax4.set_title('Temperature Distribution (Core, Top, Bottom)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('test_multi_battery_visualization.png', dpi=150)
    print("可视化已保存到 test_multi_battery_visualization.png")


if __name__ == "__main__":
    # 运行所有实验
    test_different_flow_rates()
    test_different_inlet_temps()
    test_different_currents()
    test_thermal_distribution()

    # 可视化
    visualize_results()

    print("\n" + "="*80)
    print("所有实验完成!")
    print("="*80)

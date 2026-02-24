"""
详细分析：单个电池的核心/顶部/底部温度
"""
import sys
sys.path.insert(0, '/mnt/g/github_project/Liquied-Cooling-System')

import numpy as np
import matplotlib.pyplot as plt
from BatteryEnv.multi_battery_module import MultiBattery


def analyze_single_battery_detail():
    """详细分析单个电池的温度分布"""
    print("=" * 70)
    print("详细分析：电池内部温度分布")
    print("=" * 70)

    # 场景1: 适度冷却
    mb = MultiBattery(num_batteries_per_group=13, num_groups=4)
    for i in range(4):
        mb.set_group_current(i, 30)
    mb.set_group_controls(0.2, 285)

    for _ in range(500):
        mb.run(t_seconds=1)

    # 分析第一个和最后一个电池
    print("\n场景: 电流=30A, 流速=0.2, 入口=285K")
    print("-" * 50)

    for idx, label in [(0, "第一个电池 (入口)"), (51, "最后一个电池 (出口)")]:
        b = mb.batteries[idx]
        print(f"\n{label}:")
        print(f"  核心温度: {b.get_core_temperature():.2f}K")
        print(f"  顶部温度: {b.get_top_surface_average_temperature():.2f}K")
        print(f"  底部温度: {b.get_bottom_surface_average_temperature():.2f}K")
        print(f"  入口温度: {b.inlet_temp:.2f}K")
        print(f"  核心-底部温差: {b.get_core_temperature() - b.get_bottom_surface_average_temperature():.2f}K")
        print(f"  顶部-底部温差: {b.get_top_surface_average_temperature() - b.get_bottom_surface_average_temperature():.2f}K")

    # 可视化电池内部温度剖面
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. z方向温度剖面 (第一个电池)
    ax1 = axes[0, 0]
    b = mb.batteries[0]
    gx, gy, gz = b.grid_size_x, b.grid_size_y, b.grid_size_z
    center_x, center_y = gx // 2 + 1, gy // 2 + 1
    z_temps = [b.temperature[center_x, center_y, z] for z in range(gz + 2)]
    ax1.plot(range(gz + 2), z_temps, 'b-o', markersize=3)
    ax1.axhline(298, color='g', linestyle='--', alpha=0.5, label='298K')
    ax1.set_xlabel('Z Layer (bottom to top)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Battery 1: Temperature Profile (Z Direction)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 52个电池的核心/顶部/底部温度
    ax2 = axes[0, 1]
    core_temps = [b.get_core_temperature() for b in mb.batteries]
    top_temps = [b.get_top_surface_average_temperature() for b in mb.batteries]
    bottom_temps = [b.get_bottom_surface_average_temperature() for b in mb.batteries]
    ax2.plot(range(52), core_temps, 'b-', label='Core', linewidth=2)
    ax2.plot(range(52), top_temps, 'r-', label='Top', linewidth=2)
    ax2.plot(range(52), bottom_temps, 'g-', label='Bottom', linewidth=2)
    ax2.set_xlabel('Battery Index')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Core/Top/Bottom Temperature (All Batteries)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 温差分析
    ax3 = axes[1, 0]
    core_minus_bottom = [core_temps[i] - bottom_temps[i] for i in range(52)]
    top_minus_bottom = [top_temps[i] - bottom_temps[i] for i in range(52)]
    ax3.plot(range(52), core_minus_bottom, 'b-', label='Core - Bottom', linewidth=2)
    ax3.plot(range(52), top_minus_bottom, 'r-', label='Top - Bottom', linewidth=2)
    ax3.set_xlabel('Battery Index')
    ax3.set_ylabel('Temperature Difference (K)')
    ax3.set_title('Temperature Difference (Top/Bottom vs Core)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 组统计
    ax4 = axes[1, 1]
    group_data = {'Core': [], 'Top': [], 'Bottom': []}
    for g in range(4):
        start = g * 13
        end = start + 13
        group_data['Core'].append(np.mean([mb.batteries[i].get_core_temperature() for i in range(start, end)]))
        group_data['Top'].append(np.mean([mb.batteries[i].get_top_surface_average_temperature() for i in range(start, end)]))
        group_data['Bottom'].append(np.mean([mb.batteries[i].get_bottom_surface_average_temperature() for i in range(start, end)]))

    x = np.arange(4)
    width = 0.25
    ax4.bar(x - width, group_data['Core'], width, label='Core', alpha=0.8)
    ax4.bar(x, group_data['Top'], width, label='Top', alpha=0.8)
    ax4.bar(x + width, group_data['Bottom'], width, label='Bottom', alpha=0.8)
    ax4.set_xlabel('Group')
    ax4.set_ylabel('Temperature (K)')
    ax4.set_title('Group Average: Core/Top/Bottom')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Group {i+1}' for i in range(4)])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('battery_temperature_detail.png', dpi=150)
    print("\n图表已保存: battery_temperature_detail.png")
    plt.show()


def test_response_time():
    """测试动作响应时间"""
    print("\n" + "=" * 70)
    print("测试：动作响应时间")
    print("=" * 70)

    mb = MultiBattery(num_batteries_per_group=13, num_groups=4)
    for i in range(4):
        mb.set_group_current(i, 30)
    mb.set_group_controls(0.2, 285)

    # 稳定
    for _ in range(300):
        mb.run(t_seconds=1)

    stable_temp = mb.batteries[0].get_core_temperature()
    print(f"稳定温度: {stable_temp:.2f}K")

    # 改变动作
    mb.set_group_controls(0.1, 290)  # 降低冷却
    print("\n动作改变: 流速 0.2->0.1, 入口 285K->290K")

    # 记录响应
    temps = [stable_temp]
    for step in range(50):
        mb.run(t_seconds=1)
        temps.append(mb.batteries[0].get_core_temperature())

    print(f"10步后: {temps[10]:.2f}K, 变化: {temps[10]-stable_temp:.2f}K")
    print(f"20步后: {temps[20]:.2f}K, 变化: {temps[20]-stable_temp:.2f}K")
    print(f"30步后: {temps[30]:.2f}K, 变化: {temps[30]-stable_temp:.2f}K")
    print(f"50步后: {temps[50]:.2f}K, 变化: {temps[50]-stable_temp:.2f}K")

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(range(51), temps, 'b-o', markersize=3)
    plt.axhline(stable_temp, color='g', linestyle='--', label='Initial')
    plt.xlabel('Step')
    plt.ylabel('Core Temperature (K)')
    plt.title('Response Time: Temperature Change After Action')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('response_time.png', dpi=150)
    print("\n图表已保存: response_time.png")
    plt.show()


if __name__ == "__main__":
    analyze_single_battery_detail()
    test_response_time()

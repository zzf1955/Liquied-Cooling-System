"""
全面测试和可视化脚本：验证RL环境的可靠性
包含详细的日志输出和物理解释
"""
import sys
sys.path.insert(0, '/mnt/g/github_project/Liquied-Cooling-System')

import numpy as np
import matplotlib.pyplot as plt
from BatteryEnv.multi_battery_module import MultiBattery


def run_scenario(name, mb, steps, description):
    """运行场景并打印详细日志"""
    print("\n" + "=" * 70)
    print(f"场景: {name}")
    print("=" * 70)
    print(f"描述: {description}")
    print("-" * 70)

    # 运行前状态
    print("【运行前】")
    print(f"  第一个电池入口温度: {mb.batteries[0].inlet_temp:.2f}K ({mb.batteries[0].inlet_temp-273:.1f}°C)")
    print(f"  流速: {mb.batteries[0].flow_rate:.2f} m/s")
    print(f"  电流: {mb.batteries[0].current:.1f} A")

    # 记录初始温度
    initial_temps = [b.get_core_temperature() for b in mb.batteries]

    # 运行模拟
    for _ in range(steps):
        mb.run(t_seconds=1)

    # 运行后状态
    print(f"\n【运行 {steps} 秒后】")

    # 各组温度
    group_temps = mb.get_group_average_temperatures()
    print(f"  各组平均温度:")
    for i, t in enumerate(group_temps):
        print(f"    Group {i+1}: {t:.2f}K ({t-273:.1f}°C)")

    # 温差分析
    temp_diff = group_temps[-1] - group_temps[0]
    print(f"\n  【温差分析】")
    print(f"    组间温差(最后一组-第一组): {temp_diff:.2f}K")
    if temp_diff > 3:
        print(f"    ✓ 温差 > 3K，满足要求")
    elif temp_diff > 1.5:
        print(f"    △ 温差中等 (1.5-3K)")
    else:
        print(f"    ○ 温差较小 (<1.5K)")

    # 冷却液分析
    first_inlet = mb.batteries[0].inlet_temp
    last_inlet = mb.batteries[51].inlet_temp
    coolant_rise = last_inlet - first_inlet
    print(f"\n  【冷却液分析】")
    print(f"    入口温度: {first_inlet:.2f}K ({first_inlet-273:.1f}°C)")
    print(f"    出口温度: {last_inlet:.2f}K ({last_inlet-273:.1f}°C)")
    print(f"    温升: {coolant_rise:.2f}K")

    # 温度变化
    final_temps = [b.get_core_temperature() for b in mb.batteries]
    avg_change = np.mean([final_temps[i] - initial_temps[i] for i in range(52)])
    print(f"\n  【温度变化】")
    print(f"    平均温度变化: {avg_change:+.2f}K")

    return mb


def plot_comprehensive(mb, title, filename):
    """综合可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. 52个电池的核心温度
    ax1 = axes[0, 0]
    core_temps = [b.get_core_temperature() for b in mb.batteries]
    ax1.plot(range(52), core_temps, 'b-o', markersize=3)
    ax1.axhline(298, color='g', linestyle='--', alpha=0.5, label='298K target')
    ax1.set_xlabel('Battery Index')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Core Temperature Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 组平均温度
    ax2 = axes[0, 1]
    group_temps = mb.get_group_average_temperatures()
    colors = ['blue', 'green', 'orange', 'red']
    bars = ax2.bar(range(4), group_temps, color=colors, alpha=0.7)
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Group Average Temperature')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f'Group {i+1}' for i in range(4)])
    for i, v in enumerate(group_temps):
        ax2.text(i, v + 0.5, f'{v:.1f}K', ha='center', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 冷却液入口温度
    ax3 = axes[0, 2]
    inlets = [b.inlet_temp for b in mb.batteries]
    ax3.plot(range(52), inlets, 'r-o', markersize=3)
    ax3.set_xlabel('Battery Index')
    ax3.set_ylabel('Temperature (K)')
    ax3.set_title('Coolant Inlet Temperature')
    ax3.grid(True, alpha=0.3)

    # 4. 时间演化 - 每组第一个电池
    ax4 = axes[1, 0]
    indices = [0, 13, 26, 39]
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['Group 1 (First)', 'Group 2', 'Group 3', 'Group 4 (Last)']
    for idx, color, label in zip(indices, colors, labels):
        ax4.plot(mb.time_steps, mb.temperature_history[idx],
                 label=label, color=color, linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Core Temperature (K)')
    ax4.set_title('Temperature Evolution by Group')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 电池温度分布 (核心/顶部/底部)
    ax5 = axes[1, 1]
    core_temps = [b.get_core_temperature() for b in mb.batteries]
    top_temps = [b.get_top_surface_average_temperature() for b in mb.batteries]
    bottom_temps = [b.get_bottom_surface_average_temperature() for b in mb.batteries]
    ax5.plot(range(52), core_temps, 'b-', label='Core', linewidth=2)
    ax5.plot(range(52), top_temps, 'r-', label='Top', linewidth=2)
    ax5.plot(range(52), bottom_temps, 'g-', label='Bottom', linewidth=2)
    ax5.set_xlabel('Battery Index')
    ax5.set_ylabel('Temperature (K)')
    ax5.set_title('Core/Top/Bottom Temperature')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 组间温差随时间变化
    ax6 = axes[1, 2]
    if len(mb.time_steps) > 0:
        temp_diffs = [mb.temperature_history[51][i] - mb.temperature_history[0][i]
                      for i in range(len(mb.time_steps))]
        ax6.plot(mb.time_steps, temp_diffs, 'purple', linewidth=1.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Temperature Difference (K)')
        ax6.set_title('Temp Difference (Last - First Battery)')
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\n[图表已保存: {filename}]")
    plt.show()


# ========== 场景1: 无冷却 ==========
print("\n" + "#" * 70)
print("# RL环境可靠性测试 - 场景1: 无冷却")
print("#" * 70)

mb1 = MultiBattery(num_batteries_per_group=13, num_groups=4)
for i in range(4):
    mb1.set_group_current(i, 30)
# 无冷却
for battery in mb1.batteries:
    battery.flow_rate = 0
mb1.batteries[0].inlet_temp = 288

mb1 = run_scenario(
    "无冷却 (flow_rate = 0)",
    mb1, 300,
    "测试没有冷却时的温度上升情况，模拟冷却系统故障"
)
plot_comprehensive(mb1, "Scenario 1: No Cooling", "scenario1_no_cooling.png")


# ========== 场景2: 冷却不足 ==========
print("\n" + "#" * 70)
print("# RL环境可靠性测试 - 场景2: 冷却不足")
print("#" * 70)

mb2 = MultiBattery(num_batteries_per_group=13, num_groups=4)
for i in range(4):
    mb2.set_group_current(i, 30)
mb2.set_group_controls(0.15, 288)

mb2 = run_scenario(
    "冷却不足 (低流速)",
    mb2, 500,
    "低流速冷却，电池温度较高，温差较大"
)
plot_comprehensive(mb2, "Scenario 2: Insufficient Cooling", "scenario2_low_cooling.png")


# ========== 场景3: 适度冷却 ==========
print("\n" + "#" * 70)
print("# RL环境可靠性测试 - 场景3: 适度冷却")
print("#" * 70)

mb3 = MultiBattery(num_batteries_per_group=13, num_groups=4)
for i in range(4):
    mb3.set_group_current(i, 30)
mb3.set_group_controls(0.25, 288)

mb3 = run_scenario(
    "适度冷却 (流速=0.25)",
    mb3, 500,
    "较高流速，电池温度接近25°C，温差适中"
)
plot_comprehensive(mb3, "Scenario 3: Moderate Cooling", "scenario3_moderate_cooling.png")


# ========== 场景4: 冷却过度 ==========
print("\n" + "#" * 70)
print("# RL环境可靠性测试 - 场景4: 冷却过度")
print("#" * 70)

mb4 = MultiBattery(num_batteries_per_group=13, num_groups=4)
for i in range(4):
    mb4.set_group_current(i, 30)
mb4.set_group_controls(0.4, 288)

mb4 = run_scenario(
    "冷却过度 (高流速+低温入口)",
    mb4, 500,
    "高流速+低温入口，电池温度过低"
)
plot_comprehensive(mb4, "Scenario 4: Over Cooling", "scenario4_over_cooling.png")


# ========== 场景5: 高入口温度 ==========
print("\n" + "#" * 70)
print("# RL环境可靠性测试 - 场景5: 高入口温度")
print("#" * 70)

mb5 = MultiBattery(num_batteries_per_group=13, num_groups=4)
for i in range(4):
    mb5.set_group_current(i, 30)
mb5.set_group_controls(0.25, 295)

mb5 = run_scenario(
    "高入口温度 (入口=22°C)",
    mb5, 500,
    "较高入口温度(22°C)，电池温度偏高"
)
plot_comprehensive(mb5, "Scenario 5: High Inlet Temperature", "scenario5_high_inlet.png")


# ========== 场景6: 电流突变 ==========
print("\n" + "#" * 70)
print("# RL环境可靠性测试 - 场景6: 电流突变")
print("#" * 70)

mb6 = MultiBattery(num_batteries_per_group=13, num_groups=4)
for i in range(4):
    mb6.set_group_current(i, 30)
mb6.set_group_controls(0.25, 288)

# 先稳定
for _ in range(200):
    mb6.run(t_seconds=1)

stable_temp = mb6.batteries[0].get_core_temperature()
print(f"\n【200秒稳定后】")
print(f"  第一组核心温度: {stable_temp:.2f}K ({stable_temp-273:.1f}°C)")

# 突变电流
print(f"\n【电流突变】30A -> 50A")
for i in range(4):
    mb6.set_group_current(i, 50)

for _ in range(300):
    mb6.run(t_seconds=1)

final_temp = mb6.batteries[0].get_core_temperature()
print(f"  300秒后温度: {final_temp:.2f}K ({final_temp-273:.1f}°C)")
print(f"  温度变化: {final_temp - stable_temp:+.2f}K")

plot_comprehensive(mb6, "Scenario 6: Current Surge", "scenario6_current_surge.png")


print("\n" + "#" * 70)
print("# 测试完成")
print("#" * 70)

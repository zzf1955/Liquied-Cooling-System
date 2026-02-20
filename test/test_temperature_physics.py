"""
测试脚本：验证无冷却情况下电池温度变化是否符合物理规律
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

from BatteryEnv.single_battery_module import SingleBattery

def test_no_cooling_temperature_rise():
    """测试无冷却情况下电池温度变化"""
    battery = SingleBattery()

    # 重置电池状态
    battery.reset()
    # 设置恒定电流 30A
    battery.current = 30.0
    # 无冷却：流速=0
    battery.flow_rate = 0.0

    # 记录初始平均温度
    initial_avg_temp = np.mean(battery.temperature[1:-1, 1:-1, 1:-1])
    print(f"初始平均温度: {initial_avg_temp:.2f} K")
    print(f"初始核心温度: {battery.get_core_temperature():.2f} K")
    print(f"初始顶部温度: {battery.get_top_surface_average_temperature():.2f} K")
    print(f"初始底部温度: {battery.get_bottom_surface_average_temperature():.2f} K")
    print()

    # 运行模拟，记录不同时刻的温度
    test_times = [60, 120, 300]  # 1min, 2min, 5min
    current_time = 0

    for target_time in test_times:
        # 运行到目标时间
        battery.run(t_seconds=target_time - current_time)
        current_time = target_time

        # 计算平均温度
        avg_temp = np.mean(battery.temperature[1:-1, 1:-1, 1:-1])
        core_temp = battery.get_core_temperature()
        top_temp = battery.get_top_surface_average_temperature()
        bottom_temp = battery.get_bottom_surface_average_temperature()

        print(f"时间: {current_time}秒 ({current_time/60:.1f}分钟)")
        print(f"  平均温度: {avg_temp:.2f} K")
        print(f"  核心温度: {core_temp:.2f} K")
        print(f"  顶部温度: {top_temp:.2f} K")
        print(f"  底部温度: {bottom_temp:.2f} K")
        print(f"  顶部-底部温差: {top_temp - bottom_temp:.2f} K")
        print()

    # 验证物理规律：底部应该比顶部凉
    print("=" * 50)
    print("物理规律验证:")
    print(f"  顶部温度 > 底部温度? {top_temp > bottom_temp}")

    # 计算理论温升
    print("\n理论计算:")
    Q = 30**2 * 0.18  # 热量产生 W
    V = battery.length * battery.width * battery.height  # 体积 m³
    m = battery.density * V  # 质量 kg
    c = battery.specific_heat  # 比热容

    print(f"  电池质量: {m:.2f} kg")
    print(f"  热量产生: {Q:.2f} W")

    for t in [60, 120, 300]:
        delta_T = Q * t / (m * c)
        theoretical_temp = 300 + delta_T  # 假设初始300K
        print(f"  {t}秒后理论温度: {theoretical_temp:.2f} K")

def test_with_cooling():
    """测试有冷却情况下电池温度变化"""
    print("\n" + "=" * 50)
    print("测试：有冷却情况")
    print("=" * 50)

    battery = SingleBattery()
    battery.reset()
    battery.current = 30.0
    battery.flow_rate = 2.0  # 有冷却
    battery.inlet_temp = 298.0

    initial_avg_temp = np.mean(battery.temperature[1:-1, 1:-1, 1:-1])
    print(f"初始平均温度: {initial_avg_temp:.2f} K")

    # 运行5分钟
    battery.run(t_seconds=300)

    avg_temp = np.mean(battery.temperature[1:-1, 1:-1, 1:-1])
    core_temp = battery.get_core_temperature()
    top_temp = battery.get_top_surface_average_temperature()
    bottom_temp = battery.get_bottom_surface_average_temperature()

    print(f"5分钟后 (有冷却, 流速=2.0 m/s):")
    print(f"  平均温度: {avg_temp:.2f} K")
    print(f"  核心温度: {core_temp:.2f} K")
    print(f"  顶部温度: {top_temp:.2f} K")
    print(f"  底部温度: {bottom_temp:.2f} K")
    print(f"  顶部-底部温差: {top_temp - bottom_temp:.2f} K")

if __name__ == "__main__":
    test_no_cooling_temperature_rise()
    test_with_cooling()


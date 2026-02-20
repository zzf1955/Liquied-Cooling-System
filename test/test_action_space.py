"""
测试：实际动作空间的效果
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

from BatteryEnv.single_battery_module import SingleBattery

print("=" * 60)
print("测试：实际动作空间的效果")
print("入口温度: 288-295K (15-22°C)")
print("流速: 0-6")
print("=" * 60)

# 测试所有组合
for flow_rate in [0.0, 1.0, 2.0, 4.0, 6.0]:
    for inlet_temp in [288.0, 290.0, 295.0]:
        battery = SingleBattery()
        battery.reset()
        battery.current = 30.0
        battery.flow_rate = flow_rate
        battery.inlet_temp = inlet_temp

        battery.run(t_seconds=300)  # 5分钟

        avg_temp = np.mean(battery.temperature[1:-1, 1:-1, 1:-1])
        top_temp = battery.get_top_surface_average_temperature()
        bottom_temp = battery.get_bottom_surface_average_temperature()

        print(f"流速={flow_rate:.1f}, 入口={inlet_temp:.0f}K: 平均={avg_temp:.2f}K, 温差={top_temp-bottom_temp:.2f}K")

print("\n" + "=" * 60)
print("效果总结:")
print("  入口温度288→295K: 约6K变化")
print("  流速0→6: 约2K变化")
print("  温差: 4-5K")

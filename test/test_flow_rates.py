"""
测试：不同流速下的冷却效果
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

from BatteryEnv.single_battery_module import SingleBattery

print("=" * 60)
print("测试：不同流速下的效果 (入口温度=288K)")
print("=" * 60)

for flow_rate in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]:
    battery = SingleBattery()
    battery.reset()
    battery.current = 30.0
    battery.flow_rate = flow_rate
    battery.inlet_temp = 288.0

    battery.run(t_seconds=300)  # 5分钟

    core_temp = battery.get_core_temperature()
    top_temp = battery.get_top_surface_average_temperature()
    bottom_temp = battery.get_bottom_surface_average_temperature()
    avg_temp = np.mean(battery.temperature[1:-1, 1:-1, 1:-1])

    print(f"流速={flow_rate:.1f}: 平均={avg_temp:.2f}, 核心={core_temp:.2f}, 顶部={top_temp:.2f}, 底部={bottom_temp:.2f}, 温差={top_temp-bottom_temp:.2f}")

print("\n" + "=" * 60)
print("目标:")
print("  - 电池温度控制在 25-30°C (298-303K)")
print("  - 顶部-底部温差: 6-8K (长时间)")

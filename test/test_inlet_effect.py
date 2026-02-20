"""
测试：不同入口温度的效果
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

from BatteryEnv.single_battery_module import SingleBattery

print("=" * 60)
print("测试：不同入口温度的效果 (流速=2.0)")
print("入口温度范围: 288-295K (15-22°C)")
print("=" * 60)

for inlet_temp in [288.0, 290.0, 292.0, 295.0]:
    battery = SingleBattery()
    battery.reset()
    battery.current = 30.0
    battery.flow_rate = 2.0
    battery.inlet_temp = inlet_temp

    battery.run(t_seconds=300)  # 5分钟

    core_temp = battery.get_core_temperature()
    top_temp = battery.get_top_surface_average_temperature()
    bottom_temp = battery.get_bottom_surface_average_temperature()
    avg_temp = np.mean(battery.temperature[1:-1, 1:-1, 1:-1])

    print(f"入口={inlet_temp:.0f}K: 平均={avg_temp:.2f}, 核心={core_temp:.2f}, 顶部={top_temp:.2f}, 底部={bottom_temp:.2f}, 温差={top_temp-bottom_temp:.2f}")

print("\n" + "=" * 60)
print("测试：同时改变流速和入口温度")
print("=" * 60)

for flow_rate in [2.0, 4.0, 6.0]:
    for inlet_temp in [280.0, 288.0, 295.0]:
        battery = SingleBattery()
        battery.reset()
        battery.current = 30.0
        battery.flow_rate = flow_rate
        battery.inlet_temp = inlet_temp

        battery.run(t_seconds=300)

        avg_temp = np.mean(battery.temperature[1:-1, 1:-1, 1:-1])
        top_temp = battery.get_top_surface_average_temperature()
        bottom_temp = battery.get_bottom_surface_average_temperature()

        print(f"流速={flow_rate}, 入口={inlet_temp:.0f}: 平均={avg_temp:.2f}, 温差={top_temp-bottom_temp:.2f}")

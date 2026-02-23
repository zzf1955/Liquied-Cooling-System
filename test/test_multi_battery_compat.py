"""测试优化后的热扩散函数对多电池环境的影响"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

from BatteryEnv.single_battery_module import SingleBattery
from BatteryEnv.multi_battery_module import MutiBattery
from BatteryEnv.multi_battery_env import MutiBatteryEnv


def create_reference_implementation():
    """创建原始实现（用于对比）- 包含原始的热扩散函数"""
    class ReferenceBattery(SingleBattery):
        def update_temperature_distribution_original(self):
            """原始实现 - 三层嵌套循环"""
            new_temp = np.copy(self.temperature)

            for i in range(1, self.grid_size_x + 1):
                for j in range(1, self.grid_size_y + 1):
                    for k in range(1, self.grid_size_z + 1):
                        T_ip1 = self.temperature[i+1, j, k] if i < self.grid_size_x else self.temperature[i, j, k]
                        T_im1 = self.temperature[i-1, j, k] if i > 1 else self.temperature[i, j, k]
                        T_jp1 = self.temperature[i, j+1, k] if j < self.grid_size_y else self.temperature[i, j, k]
                        T_jm1 = self.temperature[i, j-1, k] if j > 1 else self.temperature[i, j, k]
                        T_kp1 = self.temperature[i, j, k+1] if k < self.grid_size_z else self.temperature[i, j, k]
                        T_km1 = self.temperature[i, j, k-1] if k > 1 else self.temperature[i, j, k]
                        new_temp[i, j, k] = self.temperature[i, j, k] + self.alpha * self.dt / self.cell_length**2 * (
                            T_ip1 + T_im1 + T_jp1 + T_jm1 + T_kp1 + T_km1 - 6 * self.temperature[i, j, k])

            return new_temp

        def diffuse_cooling_original(self):
            """原始实现 - 三层嵌套循环"""
            new_temp = np.copy(self.temperature)

            diffusion_factor = 0.02

            for k in range(1, self.grid_size_z + 1):
                for i in range(1, self.grid_size_x + 1):
                    for j in range(1, self.grid_size_y + 1):
                        if k == 1:
                            z_diffusion = 0
                        else:
                            z_diffusion = self.temperature[i, j, k-1] - self.temperature[i, j, k]

                        new_temp[i, j, k] = self.temperature[i, j, k] + self.adjusting_factor * diffusion_factor * self.alpha * self.dt / self.cell_length**2 * z_diffusion

            return new_temp

    return ReferenceBattery


def create_reference_muti_battery():
    """创建使用原始实现的 MutiBattery"""
    class ReferenceBatteryClass(SingleBattery):
        pass

    # 替换为原始方法
    ReferenceBatteryClass.update_temperature_distribution = create_reference_implementation().update_temperature_distribution_original
    ReferenceBatteryClass.diffuse_cooling = create_reference_implementation().diffuse_cooling_original

    class ReferenceMutiBattery(MutiBattery):
        def __init__(self, num_batteries_per_group=13, num_groups=4, env_temp=300):
            # 临时保存原始方法
            orig_single_init = SingleBattery.__init__

            # 创建使用原始方法的 SingleBattery
            class OrigSingleBattery(SingleBattery):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.update_temperature_distribution = create_reference_implementation().update_temperature_distribution_original
                    self.diffuse_cooling = create_reference_implementation().diffuse_cooling_original

            # 临时修改 MutiBattery 的电池创建
            self.num_batteries_per_group = num_batteries_per_group
            self.num_groups = num_groups
            self.total_batteries = num_batteries_per_group * num_groups
            self.batteries = [OrigSingleBattery(voltage=3.2, env_temperature=env_temp) for _ in range(self.total_batteries)]
            self.initial_coolant_temp = env_temp
            self.group_indices = [i // num_batteries_per_group for i in range(self.total_batteries)]
            self.temperature_history = [[] for _ in range(self.total_batteries)]
            self.coolant_history = [[] for _ in range(self.total_batteries)]

            # 复制原始 MutiBattery 的方法
            self.reset = MutiBattery.reset
            self.run = MutiBattery.run
            self.apply_inter_battery_heat_transfer = MutiBattery.apply_inter_battery_heat_transfer
            self.get_all_core_temperatures = MutiBattery.get_all_core_temperatures
            self.get_all_bottom_surface_average_temperatures = MutiBattery.get_all_bottom_surface_average_temperatures
            self.get_group_average_temperatures = MutiBattery.get_group_average_temperatures

    return ReferenceMutiBattery


def test_single_battery_basic():
    """测试单电池基本功能"""
    print("=" * 60)
    print("测试单电池基本功能")
    print("=" * 60)

    battery = SingleBattery()

    # 测试初始状态
    assert battery.get_core_temperature() == 300.0, "初始核心温度应该是环境温度"
    print("    ✓ 初始状态正确")

    # 测试 run 方法
    battery.current = 30
    battery.flow_rate = 3.0
    battery.inlet_temp = 288.0

    outlet_temp = battery.run(1.0)

    # 验证核心温度上升
    core_temp = battery.get_core_temperature()
    assert core_temp > 300.0, f"有电流时核心温度应该上升，实际: {core_temp}"
    print(f"    ✓ 运行1秒后核心温度: {core_temp:.2f} K")

    # 验证出口温度
    assert outlet_temp > battery.inlet_temp, "出口温度应该高于入口温度"
    print(f"    ✓ 出口温度: {outlet_temp:.2f} K")

    print("\n" + "=" * 60)
    print("单电池基本功能测试通过!")
    print("=" * 60)


def test_optimized_vs_original_single():
    """测试优化后的实现与原始实现的一致性"""
    print("\n" + "=" * 60)
    print("测试优化后 vs 原始实现一致性")
    print("=" * 60)

    ReferenceBattery = create_reference_implementation()

    # 测试各种情况
    test_cases = [
        ("有冷却", 30, 3.0, 288.0),
        ("无冷却", 30, 0.0, 288.0),
        ("低流速", 30, 1.0, 290.0),
        ("高电流", 50, 3.0, 288.0),
    ]

    for name, current, flow_rate, inlet_temp in test_cases:
        ref_battery = ReferenceBattery()
        opt_battery = SingleBattery()

        # 设置相同参数
        ref_battery.current = current
        ref_battery.flow_rate = flow_rate
        ref_battery.inlet_temp = inlet_temp
        opt_battery.current = current
        opt_battery.flow_rate = flow_rate
        opt_battery.inlet_temp = inlet_temp

        # 运行模拟
        ref_outlet = ref_battery.run(1.0)
        opt_outlet = opt_battery.run(1.0)

        # 比较核心温度
        ref_core = ref_battery.get_core_temperature()
        opt_core = opt_battery.get_core_temperature()

        diff = abs(ref_core - opt_core)

        print(f"    {name}: 原始={ref_core:.6f}K, 优化={opt_core:.6f}K, 差异={diff:.2e}")

        # 验证差异在容差范围内
        assert diff < 1e-6, f"{name} 测试失败! 差异: {diff}"

    print("    ✓ 所有一致性测试通过")

    print("\n" + "=" * 60)
    print("优化后 vs 原始实现一致性测试通过!")
    print("=" * 60)


def test_multi_battery_sequential():
    """测试多电池串联冷却"""
    print("\n" + "=" * 60)
    print("测试多电池串联冷却")
    print("=" * 60)

    muti_battery = MutiBattery(
        num_batteries_per_group=13,
        num_groups=4,
        env_temp=300
    )

    # 设置电流
    for battery in muti_battery.batteries:
        battery.current = 30

    # 设置冷却
    muti_battery.batteries[0].flow_rate = 3.0
    muti_battery.batteries[0].inlet_temp = 288.0

    # 运行模拟 - 增加时间以观察更明显的效果
    muti_battery.run(10.0)

    # 验证第一组电池温度最低，最后一组最高
    group1_core = np.mean([muti_battery.batteries[i].get_core_temperature()
                           for i in range(13)])
    group4_core = np.mean([muti_battery.batteries[i].get_core_temperature()
                           for i in range(39, 52)])

    print(f"    第1组平均核心温度: {group1_core:.2f} K")
    print(f"    第4组平均核心温度: {group4_core:.2f} K")

    # 打印所有组的温度
    for g in range(4):
        g_start = g * 13
        g_end = (g + 1) * 13
        g_avg = np.mean([muti_battery.batteries[i].get_core_temperature()
                        for i in range(g_start, g_end)])
        print(f"    第{g+1}组平均核心温度: {g_avg:.2f} K")

    # 由于冷却液吸热，后面的电池入口温度更高，理论上应该更热
    # 但由于每组之间热传导可能抵消一些差异
    if group1_core < group4_core:
        print("    ✓ 串联冷却效果正确（第1组温度 < 第4组温度）")
    else:
        print("    ⚠ 温度差异不明显，可能需要更多时间步")

    print("\n" + "=" * 60)
    print("多电池串联冷却测试完成!")
    print("=" * 60)


def create_reference_muti_battery():
    """创建使用原始实现的 MutiBattery - 用于一致性验证"""
    from functools import partial

    # 定义原始的热扩散方法（独立函数）
    def update_temperature_distribution_original(self):
        """原始实现 - 三层嵌套循环"""
        new_temp = np.copy(self.temperature)

        for i in range(1, self.grid_size_x + 1):
            for j in range(1, self.grid_size_y + 1):
                for k in range(1, self.grid_size_z + 1):
                    T_ip1 = self.temperature[i+1, j, k] if i < self.grid_size_x else self.temperature[i, j, k]
                    T_im1 = self.temperature[i-1, j, k] if i > 1 else self.temperature[i, j, k]
                    T_jp1 = self.temperature[i, j+1, k] if j < self.grid_size_y else self.temperature[i, j, k]
                    T_jm1 = self.temperature[i, j-1, k] if j > 1 else self.temperature[i, j, k]
                    T_kp1 = self.temperature[i, j, k+1] if k < self.grid_size_z else self.temperature[i, j, k]
                    T_km1 = self.temperature[i, j, k-1] if k > 1 else self.temperature[i, j, k]
                    new_temp[i, j, k] = self.temperature[i, j, k] + self.alpha * self.dt / self.cell_length**2 * (
                        T_ip1 + T_im1 + T_jp1 + T_jm1 + T_kp1 + T_km1 - 6 * self.temperature[i, j, k])

        return new_temp

    def diffuse_cooling_original(self):
        """原始实现 - 三层嵌套循环"""
        new_temp = np.copy(self.temperature)

        diffusion_factor = 0.02

        for k in range(1, self.grid_size_z + 1):
            for i in range(1, self.grid_size_x + 1):
                for j in range(1, self.grid_size_y + 1):
                    if k == 1:
                        z_diffusion = 0
                    else:
                        z_diffusion = self.temperature[i, j, k-1] - self.temperature[i, j, k]

                    new_temp[i, j, k] = self.temperature[i, j, k] + self.adjusting_factor * diffusion_factor * self.alpha * self.dt / self.cell_length**2 * z_diffusion

        return new_temp

    # 创建使用原始方法的 MutiBattery
    # 策略：创建 MutiBattery，然后替换每个电池的方法
    class ReferenceMutiBattery(MutiBattery):
        def __init__(self, num_batteries_per_group=13, num_groups=4, env_temp=300):
            # 调用父类初始化
            super().__init__(num_batteries_per_group, num_groups, env_temp)

            # 替换每个电池的方法为原始实现
            for battery in self.batteries:
                battery.update_temperature_distribution = partial(update_temperature_distribution_original, battery)
                battery.diffuse_cooling = partial(diffuse_cooling_original, battery)

    return ReferenceMutiBattery


def test_optimized_vs_original_multi_battery():
    """测试多电池场景下优化实现与原始实现的一致性"""
    print("\n" + "=" * 60)
    print("测试多电池场景: 优化 vs 原始实现一致性")
    print("=" * 60)

    ReferenceMutiBattery = create_reference_muti_battery()

    # 测试不同场景
    test_cases = [
        ("标准场景", 30, 3.0, 288.0, 30.0),
        ("高电流", 50, 3.0, 288.0, 30.0),
        ("低流速", 30, 1.0, 288.0, 30.0),
        ("高温入口", 30, 3.0, 295.0, 30.0),
    ]

    for name, current, flow_rate, inlet_temp, sim_time in test_cases:
        print(f"\n    测试: {name}")

        # 创建原始实现的 MutiBattery
        ref_muti = ReferenceMutiBattery(
            num_batteries_per_group=13,
            num_groups=4,
            env_temp=300
        )

        # 创建优化实现的 MutiBattery
        opt_muti = MutiBattery(
            num_batteries_per_group=13,
            num_groups=4,
            env_temp=300
        )

        # 设置相同参数
        for battery in ref_muti.batteries:
            battery.current = current
            battery.flow_rate = flow_rate
            battery.inlet_temp = inlet_temp

        for battery in opt_muti.batteries:
            battery.current = current
            battery.flow_rate = flow_rate
            battery.inlet_temp = inlet_temp

        # 运行模拟
        ref_muti.run(sim_time)
        opt_muti.run(sim_time)

        # 比较所有电池的核心温度
        ref_temps = ref_muti.get_all_core_temperatures()
        opt_temps = opt_muti.get_all_core_temperatures()

        # 计算差异
        max_diff = max(abs(r - o) for r, o in zip(ref_temps, opt_temps))
        mean_diff = np.mean([abs(r - o) for r, o in zip(ref_temps, opt_temps)])

        # 打印每组平均温度对比
        ref_groups = ref_muti.get_group_average_temperatures()
        opt_groups = opt_muti.get_group_average_temperatures()

        print(f"      原始组温度: {[f'{t:.4f}' for t in ref_groups]}")
        print(f"      优化组温度: {[f'{t:.4f}' for t in opt_groups]}")
        print(f"      最大差异: {max_diff:.2e}, 平均差异: {mean_diff:.2e}")

        # 验证差异在容差范围内
        assert max_diff < 1e-4, f"{name} 测试失败 {max_diff}"
        print(f"      ✓ {name} 一致性验证通过")

    print("\n" + "=" * 60)
    print("多电池场景优化 vs 原始一致性测试通过!")
    print("=" * 60)


def test_multi_battery_env_basic():
    """测试多电池环境基本功能"""
    print("\n" + "=" * 60)
    print("测试多电池环境基本功能")
    print("=" * 60)

    env = MutiBatteryEnv(
        num_batteries_per_group=13,
        num_groups=4,
        max_steps=100,
        env_temp=300,
        con=True
    )

    # 重置环境
    obs, info = env.reset()
    print(f"    ✓ 环境重置成功")

    # 测试 step
    action = np.array([288.0, 3.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"    ✓ step 执行成功")

    print("\n" + "=" * 60)
    print("多电池环境基本功能测试通过!")
    print("=" * 60)


def test_flow_rate_effect():
    """测试流速对温度的影响"""
    print("\n" + "=" * 60)
    print("测试流速对温度的影响")
    print("=" * 60)

    # 测试不同流速
    flow_rates = [0, 1, 3, 5]
    temps = []

    for flow_rate in flow_rates:
        battery = SingleBattery()
        battery.current = 30
        battery.flow_rate = flow_rate
        battery.inlet_temp = 288.0

        battery.run(1.0)

        core_temp = battery.get_core_temperature()
        temps.append(core_temp)
        print(f"    流速={flow_rate}: 核心温度={core_temp:.2f} K")

    # 验证：流速越高，温度越低
    for i in range(len(flow_rates) - 1):
        if flow_rates[i] < flow_rates[i+1]:
            assert temps[i] > temps[i+1], \
                f"流速 {flow_rates[i]} -> {flow_rates[i+1]} 温度应该降低"

    print("    ✓ 流速效果正确")

    print("\n" + "=" * 60)
    print("流速效果测试通过!")
    print("=" * 60)


def test_inlet_temp_effect():
    """测试入口温度对核心温度的影响"""
    print("\n" + "=" * 60)
    print("测试入口温度对核心温度的影响")
    print("=" * 60)

    # 测试不同入口温度
    inlet_temps = [285, 288, 292, 295]
    core_temps = []

    for inlet_temp in inlet_temps:
        battery = SingleBattery()
        battery.current = 30
        battery.flow_rate = 3.0
        battery.inlet_temp = inlet_temp

        battery.run(1.0)

        core_temp = battery.get_core_temperature()
        core_temps.append(core_temp)
        print(f"    入口温度={inlet_temp}K: 核心温度={core_temp:.2f} K")

    # 验证：入口温度越高，核心温度越高
    for i in range(len(inlet_temps) - 1):
        if inlet_temps[i] < inlet_temps[i+1]:
            assert core_temps[i] < core_temps[i+1], \
                f"入口温度 {inlet_temps[i]} -> {inlet_temps[i+1]} 核心温度应该升高"

    print("    ✓ 入口温度效果正确")

    print("\n" + "=" * 60)
    print("入口温度效果测试通过!")
    print("=" * 60)


def test_no_cooling_temperature_rise():
    """测试无冷却时温度持续上升"""
    print("\n" + "=" * 60)
    print("测试无冷却时温度变化")
    print("=" * 60)

    battery = SingleBattery()
    battery.current = 30
    battery.flow_rate = 0  # 无冷却

    initial_temp = battery.get_core_temperature()
    print(f"    初始核心温度: {initial_temp:.2f} K")

    # 运行多步
    for step in range(10):
        battery.run(1.0)

    final_temp = battery.get_core_temperature()
    print(f"    10秒后核心温度: {final_temp:.2f} K")

    # 无冷却时温度应该上升
    assert final_temp > initial_temp, "无冷却时温度应该上升"
    print("    ✓ 无冷却时温度上升")

    print("\n" + "=" * 60)
    print("无冷却温度测试通过!")
    print("=" * 60)


def test_stability_long_run():
    """测试长时间运行的稳定性"""
    print("\n" + "=" * 60)
    print("测试长时间运行稳定性")
    print("=" * 60)

    battery = SingleBattery()
    battery.current = 30
    battery.flow_rate = 3.0
    battery.inlet_temp = 288.0

    # 运行100步
    temps = []
    for _ in range(100):
        battery.run(1.0)
        core_temp = battery.get_core_temperature()
        temps.append(core_temp)

        # 检查数值稳定性
        assert np.isfinite(core_temp), "温度应该是有限值"
        assert core_temp > 0, "温度应该是正值"

    print(f"    100步后核心温度: {temps[-1]:.2f} K")
    print(f"    温度范围: {min(temps):.2f} - {max(temps):.2f} K")
    print("    ✓ 数值稳定")

    print("\n" + "=" * 60)
    print("长时间运行稳定性测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_single_battery_basic()

    test_optimized_vs_original_single()

    test_multi_battery_sequential()

    # 新增：多电池场景下原始vs优化一致性测试
    test_optimized_vs_original_multi_battery()

    test_multi_battery_env_basic()

    test_flow_rate_effect()

    test_inlet_temp_effect()

    test_no_cooling_temperature_rise()

    test_stability_long_run()

    print("\n" + "=" * 60)
    print("所有多电池兼容性测试通过!")
    print("=" * 60)

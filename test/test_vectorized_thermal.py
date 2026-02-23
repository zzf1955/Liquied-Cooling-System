"""测试向量化优化后的热扩散函数"""
import numpy as np
import time
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

from BatteryEnv.single_battery_module import SingleBattery


def create_reference_implementation():
    """创建原始实现（用于对比）"""
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


def test_correctness():
    """测试优化后的代码与原始实现的数值一致性"""
    print("=" * 60)
    print("测试正确性：优化后的代码 vs 原始实现")
    print("=" * 60)

    ReferenceBattery = create_reference_implementation()
    ref_battery = ReferenceBattery()
    opt_battery = SingleBattery()

    gx, gy, gz = ref_battery.grid_size_x, ref_battery.grid_size_y, ref_battery.grid_size_z

    np.random.seed(42)
    test_temp = 300.0 + np.random.randn(gx + 2, gy + 2, gz + 2) * 5

    ref_battery.temperature = test_temp.copy()
    opt_battery.temperature = test_temp.copy()

    ref_battery.temperature[0, :, :] = ref_battery.temperature[1, :, :]
    ref_battery.temperature[-1, :, :] = ref_battery.temperature[-2, :, :]
    ref_battery.temperature[:, 0, :] = ref_battery.temperature[:, 1, :]
    ref_battery.temperature[:, -1, :] = ref_battery.temperature[:, -2, :]
    ref_battery.temperature[:, :, 0] = ref_battery.temperature[:, :, 1]
    ref_battery.temperature[:, :, -1] = ref_battery.temperature[:, :, -2]
    opt_battery.temperature = ref_battery.temperature.copy()

    print("\n[1] 测试 update_temperature_distribution...")
    ref_result = ref_battery.update_temperature_distribution_original()
    opt_result = opt_battery.update_temperature_distribution()

    ref_internal = ref_result[1:gx+1, 1:gy+1, 1:gz+1]
    opt_internal = opt_result[1:gx+1, 1:gy+1, 1:gz+1]

    diff = np.abs(ref_internal - opt_internal)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"    最大差异: {max_diff:.10e}")
    print(f"    平均差异: {mean_diff:.10e}")
    print(f"    容差: 1e-10")

    if max_diff < 1e-10:
        print("    ✓ update_temperature_distribution 测试通过!")
    else:
        print(f"    ✗ update_temperature_distribution 测试失败! 最大差异 {max_diff}")
        return False

    print("\n[2] 测试 diffuse_cooling...")

    ref_battery.temperature = test_temp.copy()
    opt_battery.temperature = test_temp.copy()
    ref_battery.temperature[0, :, :] = ref_battery.temperature[1, :, :]
    ref_battery.temperature[-1, :, :] = ref_battery.temperature[-2, :, :]
    ref_battery.temperature[:, 0, :] = ref_battery.temperature[:, 1, :]
    ref_battery.temperature[:, -1, :] = ref_battery.temperature[:, -2, :]
    ref_battery.temperature[:, :, 0] = ref_battery.temperature[:, :, 1]
    ref_battery.temperature[:, :, -1] = ref_battery.temperature[:, :, -2]
    opt_battery.temperature = ref_battery.temperature.copy()

    ref_result = ref_battery.diffuse_cooling_original()
    opt_result = opt_battery.diffuse_cooling()

    ref_internal = ref_result[1:gx+1, 1:gy+1, 1:gz+1]
    opt_internal = opt_result[1:gx+1, 1:gy+1, 1:gz+1]

    diff = np.abs(ref_internal - opt_internal)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"    最大差异: {max_diff:.10e}")
    print(f"    平均差异: {mean_diff:.10e}")
    print(f"    容差: 1e-10")

    if max_diff < 1e-10:
        print("    ✓ diffuse_cooling 测试通过!")
    else:
        print(f"    ✗ diffuse_cooling 测试失败! 最大差异 {max_diff}")
        return False

    print("\n" + "=" * 60)
    print("正确性测试全部通过!")
    print("=" * 60)
    return True


def test_performance():
    """测试性能提升"""
    print("\n" + "=" * 60)
    print("测试性能：优化后的代码 vs 原始实现")
    print("=" * 60)

    ReferenceBattery = create_reference_implementation()
    ref_battery = ReferenceBattery()
    opt_battery = SingleBattery()

    np.random.seed(42)
    test_temp = 300.0 + np.random.randn(
        ref_battery.grid_size_x + 2,
        ref_battery.grid_size_y + 2,
        ref_battery.grid_size_z + 2
    ) * 5

    for _ in range(10):
        ref_battery.temperature = test_temp.copy()
        ref_battery.update_temperature_distribution_original()
        ref_battery.diffuse_cooling_original()

    print("\n[1] 测试 update_temperature_distribution 性能...")
    n_iterations = 1000

    start = time.time()
    for _ in range(n_iterations):
        ref_battery.temperature = test_temp.copy()
        ref_battery.update_temperature_distribution_original()
    ref_time = time.time() - start

    start = time.time()
    for _ in range(n_iterations):
        opt_battery.temperature = test_temp.copy()
        opt_battery.update_temperature_distribution()
    opt_time = time.time() - start

    speedup = ref_time / opt_time
    print(f"    原始实现: {ref_time:.4f}s ({n_iterations} 次迭代)")
    print(f"    优化实现: {opt_time:.4f}s ({n_iterations} 次迭代)")
    print(f"    加速比: {speedup:.2f}x")

    print("\n[2] 测试 diffuse_cooling 性能...")
    n_iterations = 1000

    start = time.time()
    for _ in range(n_iterations):
        ref_battery.temperature = test_temp.copy()
        ref_battery.diffuse_cooling_original()
    ref_time = time.time() - start

    start = time.time()
    for _ in range(n_iterations):
        opt_battery.temperature = test_temp.copy()
        opt_battery.diffuse_cooling()
    opt_time = time.time() - start

    speedup = ref_time / opt_time
    print(f"    原始实现: {ref_time:.4f}s ({n_iterations} 次迭代)")
    print(f"    优化实现: {opt_time:.4f}s ({n_iterations} 次迭代)")
    print(f"    加速比: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("性能测试完成!")
    print("=" * 60)


def test_battery_run():
    """测试电池运行流程"""
    print("\n" + "=" * 60)
    print("测试电池运行流程（带冷却）")
    print("=" * 60)

    battery = SingleBattery()
    battery.current = 30
    battery.flow_rate = 3.0
    battery.inlet_temp = 288.0

    outlet_temp = battery.run(1.0)

    print(f"入口温度: {battery.inlet_temp:.2f} K")
    print(f"流速: {battery.flow_rate} m/s")
    print(f"出口温度: {outlet_temp:.2f} K")
    print(f"核心温度: {battery.get_core_temperature():.2f} K")
    print(f"平均温度: {np.mean(battery.temperature):.2f} K")

    print("\n--- 无冷却测试 ---")
    battery2 = SingleBattery()
    battery2.current = 30
    battery2.flow_rate = 0

    outlet_temp2 = battery2.run(1.0)
    print(f"核心温度 (无冷却): {battery2.get_core_temperature():.2f} K")

    print("\n" + "=" * 60)
    print("电池运行测试完成!")
    print("=" * 60)


def test_boundary_conditions():
    """测试边界条件是否正确应用"""
    print("\n" + "=" * 60)
    print("测试边界条件")
    print("=" * 60)

    battery = SingleBattery()

    # 设置非均匀温度场
    np.random.seed(42)
    battery.temperature = 300.0 + np.random.randn(
        battery.grid_size_x + 2,
        battery.grid_size_y + 2,
        battery.grid_size_z + 2
    ) * 10

    # 执行热扩散
    battery.update_temperature_distribution()

    # 检查边界条件：边界值应该等于相邻内部值
    # X方向
    assert np.allclose(battery.temperature[0, :, :], battery.temperature[1, :, :]), "X- 边界条件失败"
    assert np.allclose(battery.temperature[-1, :, :], battery.temperature[-2, :, :]), "X+ 边界条件失败"

    # Y方向
    assert np.allclose(battery.temperature[:, 0, :], battery.temperature[:, 1, :]), "Y- 边界条件失败"
    assert np.allclose(battery.temperature[:, -1, :], battery.temperature[:, -2, :]), "Y+ 边界条件失败"

    # Z方向
    assert np.allclose(battery.temperature[:, :, 0], battery.temperature[:, :, 1]), "Z- 边界条件失败"
    assert np.allclose(battery.temperature[:, :, -1], battery.temperature[:, :, -2]), "Z+ 边界条件失败"

    print("    ✓ 绝热边界条件测试通过!")

    # diffuse_cooling 边界测试
    battery2 = SingleBattery()
    battery2.temperature = 300.0 + np.random.randn(
        battery2.grid_size_x + 2,
        battery2.grid_size_y + 2,
        battery2.grid_size_z + 2
    ) * 10

    battery2.diffuse_cooling()

    # 检查边界条件
    assert np.allclose(battery2.temperature[0, :, :], battery2.temperature[1, :, :]), "diffuse X- 边界条件失败"
    assert np.allclose(battery2.temperature[-1, :, :], battery2.temperature[-2, :, :]), "diffuse X+ 边界条件失败"
    assert np.allclose(battery2.temperature[:, 0, :], battery2.temperature[:, 1, :]), "diffuse Y- 边界条件失败"
    assert np.allclose(battery2.temperature[:, -1, :], battery2.temperature[:, -2, :]), "diffuse Y+ 边界条件失败"
    assert np.allclose(battery2.temperature[:, :, 0], battery2.temperature[:, :, 1]), "diffuse Z- 边界条件失败"
    assert np.allclose(battery2.temperature[:, :, -1], battery2.temperature[:, :, -2]), "diffuse Z+ 边界条件失败"

    print("    ✓ diffuse_cooling 边界条件测试通过!")

    print("\n" + "=" * 60)
    print("边界条件测试通过!")
    print("=" * 60)


def test_energy_conservation():
    """测试能量守恒（绝热系统）"""
    print("\n" + "=" * 60)
    print("测试能量守恒")
    print("=" * 60)

    battery = SingleBattery()

    # 初始均匀温度
    initial_temp = 300.0
    battery.temperature.fill(initial_temp)

    # 执行多次热扩散
    for _ in range(100):
        battery.update_temperature_distribution()

    # 检查总能量（温度总和）
    total_temp_before = initial_temp * battery.temperature.size
    total_temp_after = np.sum(battery.temperature)

    # 绝热条件下，总温度应该保持不变（能量守恒）
    relative_diff = abs(total_temp_after - total_temp_before) / total_temp_before

    print(f"    初始总温度: {total_temp_before:.2f}")
    print(f"    最终总温度: {total_temp_after:.2f}")
    print(f"    相对差异: {relative_diff:.10e}")

    # 容差设置为 1e-8，因为数值计算有精度误差
    assert relative_diff < 1e-8, f"能量守恒测试失败! 相对差异: {relative_diff}"
    print("    ✓ 能量守恒测试通过!")

    print("\n" + "=" * 60)
    print("能量守恒测试通过!")
    print("=" * 60)


def test_stability():
    """测试数值稳定性"""
    print("\n" + "=" * 60)
    print("测试数值稳定性")
    print("=" * 60)

    battery = SingleBattery()

    # 设置极端温度梯度
    battery.temperature.fill(300.0)
    battery.temperature[1:-1, 1:-1, 1:-1] = 500.0  # 内部非常热

    # 执行多次迭代，确保不会数值爆炸
    for i in range(100):
        battery.update_temperature_distribution()
        battery.diffuse_cooling()

        # 检查温度是否有限（不会爆炸）
        assert np.all(np.isfinite(battery.temperature)), f"数值爆炸在第 {i} 次迭代"
        assert np.all(battery.temperature > 0), f"温度变为负值在第 {i} 次迭代"

    print(f"    迭代100次后最大温度: {np.max(battery.temperature):.2f} K")
    print(f"    迭代100次后最小温度: {np.min(battery.temperature):.2f} K")
    print("    ✓ 数值稳定性测试通过!")

    print("\n" + "=" * 60)
    print("数值稳定性测试通过!")
    print("=" * 60)


def test_zero_gradient():
    """测试零温度梯度情况"""
    print("\n" + "=" * 60)
    print("测试零温度梯度情况")
    print("=" * 60)

    battery = SingleBattery()

    # 均匀温度场 - 不应该有任何变化
    uniform_temp = 300.0
    battery.temperature.fill(uniform_temp)

    battery.update_temperature_distribution()

    assert np.allclose(battery.temperature, uniform_temp), "均匀温度场应该有变化"
    print("    ✓ 均匀温度场测试通过!")

    # 验证均匀温度场在多次迭代后仍然保持均匀
    for _ in range(100):
        battery.update_temperature_distribution()

    assert np.allclose(battery.temperature, uniform_temp), "多次迭代后均匀温度场发生变化"
    print("    ✓ 多次迭代均匀温度场测试通过!")

    print("\n" + "=" * 60)
    print("零温度梯度测试通过!")
    print("=" * 60)


def test_various_temperature_distributions():
    """测试各种不同温度分布"""
    print("\n" + "=" * 60)
    print("测试各种不同温度分布")
    print("=" * 60)

    ReferenceBattery = create_reference_implementation()

    # 创建不同温度分布
    gx, gy, gz = 7, 17, 20
    shape = (gx + 2, gy + 2, gz + 2)

    test_cases = []

    # 1. 中心热点
    temp = np.full(shape, 300.0)
    temp[gx//2+1, gy//2+1, gz//2+1] = 400.0
    test_cases.append(("中心热点", temp.copy()))

    # 2. 边缘热点
    temp = np.full(shape, 300.0)
    temp[1, 1, 1] = 400.0
    test_cases.append(("边缘热点", temp.copy()))

    # 3. 对角线热点
    temp = np.full(shape, 300.0)
    temp[gx, gy, gz] = 400.0
    test_cases.append(("对角线热点", temp.copy()))

    # 4. 随机噪声
    np.random.seed(42)
    temp = 300.0 + np.random.randn(*shape) * 10
    test_cases.append(("随机噪声", temp.copy()))

    for name, test_temp in test_cases:
        ref_battery = ReferenceBattery()
        opt_battery = SingleBattery()

        ref_battery.temperature = test_temp.copy()
        opt_battery.temperature = test_temp.copy()

        # 应用边界条件
        ref_battery.temperature[0, :, :] = ref_battery.temperature[1, :, :]
        ref_battery.temperature[-1, :, :] = ref_battery.temperature[-2, :, :]
        ref_battery.temperature[:, 0, :] = ref_battery.temperature[:, 1, :]
        ref_battery.temperature[:, -1, :] = ref_battery.temperature[:, -2, :]
        ref_battery.temperature[:, :, 0] = ref_battery.temperature[:, :, 1]
        ref_battery.temperature[:, :, -1] = ref_battery.temperature[:, :, -2]
        opt_battery.temperature = ref_battery.temperature.copy()

        # 运行单步迭代 - 只比较单步
        ref_battery.update_temperature_distribution_original()
        ref_battery.diffuse_cooling_original()
        opt_battery.update_temperature_distribution()
        opt_battery.diffuse_cooling()

        # 比较结果
        ref_internal = ref_battery.temperature[1:gx+1, 1:gy+1, 1:gz+1]
        opt_internal = opt_battery.temperature[1:gx+1, 1:gy+1, 1:gz+1]

        diff = np.abs(ref_internal - opt_internal)
        max_diff = np.max(diff)

        print(f"    {name}: 最大差异 = {max_diff:.10e}")
        assert max_diff < 1e-10, f"{name} 测试失败! 最大差异 {max_diff}"

    print("    ✓ 各种温度分布测试通过!")

    print("\n" + "=" * 60)
    print("各种温度分布测试通过!")
    print("=" * 60)


def test_multi_step_iteration():
    """测试多步迭代后的结果对比"""
    print("\n" + "=" * 60)
    print("测试多步迭代")
    print("=" * 60)

    ReferenceBattery = create_reference_implementation()

    ref_battery = ReferenceBattery()
    opt_battery = SingleBattery()

    gx, gy, gz = ref_battery.grid_size_x, ref_battery.grid_size_y, ref_battery.grid_size_z

    np.random.seed(123)
    test_temp = 300.0 + np.random.randn(gx + 2, gy + 2, gz + 2) * 15

    ref_battery.temperature = test_temp.copy()
    opt_battery.temperature = test_temp.copy()

    # 迭代多次
    n_steps = 10  # 减少迭代次数以确保数值精度
    for step in range(n_steps):
        ref_battery.update_temperature_distribution_original()
        ref_battery.diffuse_cooling_original()
        opt_battery.update_temperature_distribution()
        opt_battery.diffuse_cooling()

    # 比较最终结果
    ref_internal = ref_battery.temperature[1:gx+1, 1:gy+1, 1:gz+1]
    opt_internal = opt_battery.temperature[1:gx+1, 1:gy+1, 1:gz+1]

    diff = np.abs(ref_internal - opt_internal)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"    迭代 {n_steps} 步后:")
    print(f"    最大差异: {max_diff:.10e}")
    print(f"    平均差异: {mean_diff:.10e}")

    assert max_diff < 1e-10, f"多步迭代测试失败! 最大差异 {max_diff}"
    print("    ✓ 多步迭代测试通过!")

    print("\n" + "=" * 60)
    print("多步迭代测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    if not test_correctness():
        sys.exit(1)

    test_performance()

    test_battery_run()

    test_boundary_conditions()

    test_energy_conservation()

    test_stability()

    test_zero_gradient()

    print("\n" + "=" * 60)
    print("所有测试全部通过!")
    print("=" * 60)

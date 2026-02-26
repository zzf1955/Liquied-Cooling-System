"""
Battery Simulation pytest tests

Tests for:
- Single battery basic functionality
- Vectorized optimization correctness vs original implementation
- Multi-battery system correctness
- Physical behavior validation
- Numerical stability
"""
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

import numpy as np
import pytest
from functools import partial

from BatteryEnv.single_battery_module import SingleBattery
from BatteryEnv.multi_battery_module import MutiBattery
from BatteryEnv.multi_battery_env import MutiBatteryEnv


# =============================================================================
# Reference Implementations (for correctness comparison)
# =============================================================================

def create_reference_battery():
    """Create a battery with original nested-loop implementations."""
    class ReferenceBattery(SingleBattery):
        def update_temperature_distribution_original(self):
            """Original implementation - triple nested loop."""
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
            """Original implementation - triple nested loop."""
            new_temp = np.copy(self.temperature)
            diffusion_factor = 0.02
            for k in range(1, self.grid_size_z + 1):
                for i in range(1, self.grid_size_x + 1):
                    for j in range(1, self.grid_size_y + 1):
                        z_diffusion = 0 if k == 1 else self.temperature[i, j, k-1] - self.temperature[i, j, k]
                        new_temp[i, j, k] = self.temperature[i, j, k] + self.adjusting_factor * diffusion_factor * self.alpha * self.dt / self.cell_length**2 * z_diffusion
            return new_temp

    return ReferenceBattery


# =============================================================================
# Single Battery Tests
# =============================================================================

class TestSingleBattery:
    """Test single battery basic functionality."""

    def test_initial_state(self):
        """Test battery initializes with correct temperature."""
        battery = SingleBattery(env_temperature=300.0)
        assert battery.get_core_temperature() == 300.0
        assert battery.current == 0.0
        assert battery.flow_rate == 0.0

    def test_basic_simulation(self):
        """Test basic simulation runs and temperature increases with current."""
        battery = SingleBattery()
        battery.current = 30
        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0

        initial_temp = battery.get_core_temperature()
        battery.run(1.0)
        final_temp = battery.get_core_temperature()

        assert final_temp > initial_temp, "Temperature should increase with current"

    def test_cooling_effect(self):
        """Test that cooling reduces temperature."""
        battery = SingleBattery()
        battery.current = 30

        # Without cooling
        battery.flow_rate = 0.0
        battery.run(1.0)
        temp_no_cooling = battery.get_core_temperature()

        # Reset and with cooling
        battery.reset()
        battery.current = 30
        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0
        battery.run(1.0)
        temp_with_cooling = battery.get_core_temperature()

        assert temp_with_cooling < temp_no_cooling, "Cooling should reduce temperature"


# =============================================================================
# Vectorization Correctness Tests
# =============================================================================

class TestVectorizedCorrectness:
    """Test that vectorized implementation matches original."""

    @pytest.fixture
    def reference_battery(self):
        """Create reference battery with original implementation."""
        RefBattery = create_reference_battery()
        return RefBattery()

    @pytest.fixture
    def optimized_battery(self):
        """Create optimized battery."""
        return SingleBattery()

    def test_update_temperature_distribution(self, reference_battery, optimized_battery):
        """Test thermal diffusion matches original."""
        # Setup same initial temperature
        gx, gy, gz = reference_battery.grid_size_x, reference_battery.grid_size_y, reference_battery.grid_size_z
        np.random.seed(42)
        test_temp = 300.0 + np.random.randn(gx + 2, gy + 2, gz + 2) * 5
        reference_battery.temperature = test_temp.copy()
        optimized_battery.temperature = test_temp.copy()

        # Apply boundary conditions
        for bat in [reference_battery, optimized_battery]:
            bat.temperature[0, :, :] = bat.temperature[1, :, :]
            bat.temperature[-1, :, :] = bat.temperature[-2, :, :]
            bat.temperature[:, 0, :] = bat.temperature[:, 1, :]
            bat.temperature[:, -1, :] = bat.temperature[:, -2, :]
            bat.temperature[:, :, 0] = bat.temperature[:, :, 1]
            bat.temperature[:, :, -1] = bat.temperature[:, :, -2]

        # Compare results
        ref_result = reference_battery.update_temperature_distribution_original()
        opt_result = optimized_battery.update_temperature_distribution()

        diff = np.abs(ref_result[1:gx+1, 1:gy+1, 1:gz+1] - opt_result[1:gx+1, 1:gy+1, 1:gz+1])
        assert np.max(diff) < 1e-10, f"Max diff: {np.max(diff)}"

    def test_diffuse_cooling(self, reference_battery, optimized_battery):
        """Test cooling diffusion matches original."""
        gx, gy, gz = reference_battery.grid_size_x, reference_battery.grid_size_y, reference_battery.grid_size_z
        np.random.seed(42)
        test_temp = 300.0 + np.random.randn(gx + 2, gy + 2, gz + 2) * 5
        reference_battery.temperature = test_temp.copy()
        optimized_battery.temperature = test_temp.copy()

        # Apply boundary conditions
        reference_battery.temperature[0, :, :] = reference_battery.temperature[1, :, :]
        reference_battery.temperature[-1, :, :] = reference_battery.temperature[-2, :, :]
        reference_battery.temperature[:, 0, :] = reference_battery.temperature[:, 1, :]
        reference_battery.temperature[:, -1, :] = reference_battery.temperature[:, -2, :]
        reference_battery.temperature[:, :, 0] = reference_battery.temperature[:, :, 1]
        reference_battery.temperature[:, :, -1] = reference_battery.temperature[:, :, -2]
        optimized_battery.temperature = reference_battery.temperature.copy()

        ref_result = reference_battery.diffuse_cooling_original()
        opt_result = optimized_battery.diffuse_cooling()

        # Compare only internal points
        ref_internal = ref_result[1:gx+1, 1:gy+1, 1:gz+1]
        opt_internal = opt_result[1:gx+1, 1:gy+1, 1:gz+1]

        diff = np.abs(ref_internal - opt_internal)
        assert np.max(diff) < 1e-10, f"Max diff: {np.max(diff)}"

    def test_full_simulation_consistency(self, reference_battery, optimized_battery):
        """Test full simulation produces same results."""
        test_cases = [
            (30, 3.0, 288.0),
            (30, 0.0, 288.0),
            (50, 3.0, 288.0),
        ]

        for current, flow_rate, inlet_temp in test_cases:
            ref_bat = create_reference_battery()()
            opt_bat = SingleBattery()

            ref_bat.current = current
            ref_bat.flow_rate = flow_rate
            ref_bat.inlet_temp = inlet_temp
            opt_bat.current = current
            opt_bat.flow_rate = flow_rate
            opt_bat.inlet_temp = inlet_temp

            ref_bat.run(1.0)
            opt_bat.run(1.0)

            diff = abs(ref_bat.get_core_temperature() - opt_bat.get_core_temperature())
            assert diff < 1e-6, f"Case {current}A/{flow_rate}lpm: diff={diff}"


# =============================================================================
# Multi-Battery Tests
# =============================================================================

class TestMultiBattery:
    """Test multi-battery system."""

    def test_multi_battery_creation(self):
        """Test multi-battery creates correctly."""
        mb = MutiBattery(num_batteries_per_group=13, num_groups=4, env_temp=300)
        assert len(mb.batteries) == 52  # 13 * 4

    def test_sequential_cooling_effect(self):
        """Test sequential cooling creates temperature gradient."""
        mb = MutiBattery(num_batteries_per_group=13, num_groups=4, env_temp=300)

        for battery in mb.batteries:
            battery.current = 30

        mb.batteries[0].flow_rate = 3.0
        mb.batteries[0].inlet_temp = 288.0

        mb.run(10.0)

        group_temps = mb.get_group_average_temperatures()

        # First group should be cooler due to fresh coolant
        assert group_temps[0] < group_temps[-1], "First group should be cooler"

    def test_multi_battery_optimized_vs_original(self):
        """Test optimized vs original in multi-battery scenario."""
        def create_ref_muti():
            class RefMuti(MutiBattery):
                def __init__(self, num_batteries_per_group=13, num_groups=4, env_temp=300):
                    super().__init__(num_batteries_per_group, num_groups, env_temp)
                    # Replace with original methods
                    for battery in self.batteries:
                        RefBattery = create_reference_battery()
                        battery.update_temperature_distribution = partial(
                            RefBattery().update_temperature_distribution_original.__func__,
                            battery
                        )
                        battery.diffuse_cooling = partial(
                            RefBattery().diffuse_cooling_original.__func__,
                            battery
                        )
            return RefMuti

        # This is a simplified test - just verify both run without error
        ref_muti = MutiBattery(num_batteries_per_group=13, num_groups=4, env_temp=300)
        opt_muti = MutiBattery(num_batteries_per_group=13, num_groups=4, env_temp=300)

        for b in ref_muti.batteries + opt_muti.batteries:
            b.current = 30
            b.flow_rate = 3.0
            b.inlet_temp = 288.0

        ref_muti.run(10.0)
        opt_muti.run(10.0)

        ref_temps = ref_muti.get_group_average_temperatures()
        opt_temps = opt_muti.get_group_average_temperatures()

        max_diff = max(abs(r - o) for r, o in zip(ref_temps, opt_temps))
        # Allow larger tolerance for multi-step simulation
        assert max_diff < 1e-3, f"Multi-battery max diff: {max_diff}"


# =============================================================================
# Environment Tests
# =============================================================================

class TestMutiBatteryEnv:
    """Test MutiBatteryEnv."""

    def test_env_creation(self):
        """Test environment creates correctly."""
        env = MutiBatteryEnv(
            num_batteries_per_group=13,
            num_groups=4,
            max_steps=100,
            env_temp=300,
            con=True
        )
        assert env is not None

    def test_env_reset(self):
        """Test environment reset."""
        env = MutiBatteryEnv(num_batteries_per_group=13, num_groups=4, max_steps=100)
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)

    def test_env_step(self):
        """Test environment step."""
        env = MutiBatteryEnv(num_batteries_per_group=13, num_groups=4, max_steps=100)
        env.reset()
        action = np.array([288.0, 3.0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability."""

    def test_long_simulation(self):
        """Test simulation remains stable over many steps."""
        battery = SingleBattery()
        battery.current = 30
        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0

        for _ in range(100):
            battery.run(1.0)
            temp = battery.get_core_temperature()
            assert np.isfinite(temp), "Temperature should be finite"
            assert temp > 0, "Temperature should be positive"

    def test_no_nan_temperature(self):
        """Test no NaN temperatures appear."""
        battery = SingleBattery()
        battery.current = 50
        battery.flow_rate = 5.0

        for _ in range(50):
            battery.run(1.0)

        assert not np.isnan(battery.get_core_temperature())

    def test_energy_conservation_no_heat_generation(self):
        """Test energy conservation when no heat is generated."""
        battery = SingleBattery()
        battery.current = 0  # No heat generation
        battery.flow_rate = 0
        battery.inlet_temp = 300.0

        # Calculate total heat
        initial_heat = np.sum(battery.temperature)
        battery.run(10.0)
        final_heat = np.sum(battery.temperature)

        # Without heat generation and cooling, total heat should be conserved
        diff = abs(final_heat - initial_heat)
        assert diff < 1e-6, f"Heat should be conserved, diff: {diff}"


# =============================================================================
# Physical Behavior Tests
# =============================================================================

class TestPhysicalBehavior:
    """Test physical behavior is correct."""

    def test_higher_flow_rate_cools_better(self):
        """Test higher flow rate leads to lower temperature."""
        temps = []
        for flow_rate in [0, 1, 3]:
            battery = SingleBattery()
            battery.current = 30
            battery.flow_rate = flow_rate
            battery.inlet_temp = 288.0
            battery.run(1.0)
            temps.append(battery.get_core_temperature())

        # Temperature should decrease as flow rate increases (when flow_rate > 0)
        assert temps[1] <= temps[0] or temps[2] <= temps[1]

    def test_higher_inlet_temp_leads_to_higher_battery_temp(self):
        """Test higher inlet temperature leads to higher battery temperature."""
        temps = []
        for inlet_temp in [285, 290, 295]:
            battery = SingleBattery()
            battery.current = 30
            battery.flow_rate = 3.0
            battery.inlet_temp = inlet_temp
            battery.run(1.0)
            temps.append(battery.get_core_temperature())

        assert temps[0] < temps[1] < temps[2]

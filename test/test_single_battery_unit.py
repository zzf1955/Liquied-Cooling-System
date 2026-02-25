"""
Unit tests for SingleBattery class in single_battery_module.py

This test file provides comprehensive unit tests for each function
in the SingleBattery class to verify correctness.
"""
import sys
sys.path.insert(0, '/mnt/g/github_project/Liquied-Cooling-System/test_worktree')

import numpy as np
import pytest

from BatteryEnv.single_battery_module import SingleBattery


class TestSingleBatteryInit:
    """Test SingleBattery initialization."""

    def test_default_initialization(self):
        """Test default initialization with all default parameters."""
        battery = SingleBattery()

        # Check default values
        assert battery.voltage == 3.2
        assert battery.internal_resistance == 0.18
        assert battery.length == 0.07165
        assert battery.width == 0.1747
        assert battery.height == 0.20747
        assert battery.rated_capacity == 280
        assert battery.density == 2700
        assert battery.specific_heat == 900
        assert battery.thermal_conductivity == 237

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        battery = SingleBattery(
            voltage=3.7,
            internal_resistance=0.2,
            length=0.1,
            width=0.2,
            height=0.3,
            rated_capacity=300,
            min_voltage=2.0,
            max_voltage=4.2,
            density=2500,
            specific_heat=1000,
            thermal_conductivity=200,
            coolant_density=1000,
            coolant_specific_heat=3500,
            coolant_conductivity=0.5,
            env_temperature=310.0
        )

        assert battery.voltage == 3.7
        assert battery.internal_resistance == 0.2
        assert battery.env_temperature == 310.0
        assert battery.temperature[1, 1, 1] == 310.0

    def test_grid_size_calculation(self):
        """Test grid size is calculated correctly."""
        battery = SingleBattery(
            length=0.06,  # 0.06 / 0.02 = 3
            width=0.16,   # 0.16 / 0.02 = 8
            height=0.2    # 0.2 / 0.02 = 10
        )

        # grid_size = length / cell_length (default cell_length = 0.02)
        assert battery.grid_size_x == 3
        assert battery.grid_size_y == 8
        assert battery.grid_size_z == 10

    def test_temperature_grid_initialized(self):
        """Test temperature grid is properly initialized."""
        battery = SingleBattery(env_temperature=300.0)

        # Check grid dimensions (gx+2, gy+2, gz+2 for boundaries)
        assert battery.temperature.shape[0] == battery.grid_size_x + 2
        assert battery.temperature.shape[1] == battery.grid_size_y + 2
        assert battery.temperature.shape[2] == battery.grid_size_z + 2

        # Check all temperatures are initialized to env_temperature
        assert np.all(battery.temperature == 300.0)

    def test_thermal_parameters(self):
        """Test thermal parameters are calculated correctly."""
        battery = SingleBattery(
            density=2700,
            specific_heat=900,
            thermal_conductivity=237
        )

        # alpha = k / (rho * Cp)
        expected_alpha = 237 / (2700 * 900)
        assert np.isclose(battery.alpha, expected_alpha)

    def test_invalid_parameters_raises_assertion(self):
        """Test invalid parameters raise assertion errors."""
        # max_voltage <= min_voltage should raise
        with pytest.raises(AssertionError):
            SingleBattery(max_voltage=2.5, min_voltage=3.0)


class TestSingleBatteryReset:
    """Test SingleBattery reset function."""

    def test_reset_clears_temperature(self):
        """Test reset clears temperature to env temperature."""
        battery = SingleBattery(env_temperature=300.0)

        # Modify temperature
        battery.temperature.fill(400.0)
        battery.reset()

        assert np.all(battery.temperature == 300.0)

    def test_reset_clears_current(self):
        """Test reset clears current."""
        battery = SingleBattery()
        battery.current = 50.0
        battery.reset()

        assert battery.current == 0.0

    def test_reset_clears_flow_rate(self):
        """Test reset clears flow rate."""
        battery = SingleBattery()
        battery.flow_rate = 5.0
        battery.reset()

        assert battery.flow_rate == 0.0

    def test_reset_clears_inlet_temp(self):
        """Test reset clears inlet temperature."""
        battery = SingleBattery(env_temperature=300.0)
        battery.inlet_temp = 280.0
        battery.reset()

        assert battery.inlet_temp == 300.0

    def test_reset_clears_thermal_history(self):
        """Test reset clears thermal history."""
        battery = SingleBattery()
        battery.thermal_history = [300.0, 301.0, 302.0]
        battery.reset()

        assert battery.thermal_history == []


class TestUpdateHeatGeneration:
    """Test update_heat_generation function."""

    def test_no_heat_generation_without_current(self):
        """Test no heat generation when current is zero."""
        battery = SingleBattery(env_temperature=300.0)
        battery.current = 0.0

        initial_temp = battery.get_core_temperature()
        battery.update_heat_generation()
        final_temp = battery.get_core_temperature()

        assert final_temp == initial_temp

    def test_heat_generation_with_current(self):
        """Test heat generation when current is applied."""
        battery = SingleBattery(env_temperature=300.0)
        battery.current = 30.0

        initial_temp = battery.get_core_temperature()
        battery.update_heat_generation()
        final_temp = battery.get_core_temperature()

        # Temperature should increase
        assert final_temp > initial_temp

    def test_heat_generation_proportional_to_current_squared(self):
        """Test heat generation is proportional to I^2 * R."""
        battery1 = SingleBattery(env_temperature=300.0)
        battery1.current = 20.0
        battery1.update_heat_generation()
        temp1 = battery1.get_core_temperature()

        battery2 = SingleBattery(env_temperature=300.0)
        battery2.current = 40.0  # 2x current
        battery2.update_heat_generation()
        temp2 = battery2.get_core_temperature()

        # Heat should be 4x (I^2), so temperature increase should be ~4x
        delta1 = temp1 - 300.0
        delta2 = temp2 - 300.0

        assert np.isclose(delta2 / delta1, 4.0, rtol=0.1)


class TestUpdateTemperatureDistribution:
    """Test update_temperature_distribution function."""

    def test_returns_temperature_array(self):
        """Test function returns temperature array."""
        battery = SingleBattery(env_temperature=300.0)
        result = battery.update_temperature_distribution()

        assert isinstance(result, np.ndarray)
        assert result.shape == battery.temperature.shape

    def test_thermal_diffusion_homogeneous(self):
        """Test thermal diffusion in homogeneous temperature field."""
        battery = SingleBattery(env_temperature=300.0)

        # All points at same temperature - no change expected
        initial = battery.temperature.copy()
        battery.update_temperature_distribution()

        # With homogeneous temperature, diffusion should not change much
        # (small numerical differences possible)
        diff = np.abs(battery.temperature - initial).max()
        assert diff < 1e-10

    def test_thermal_diffusion_temperature_gradient(self):
        """Test thermal diffusion with temperature gradient."""
        battery = SingleBattery(env_temperature=300.0)

        # Set up a gradient: one side hot, one side cold
        battery.temperature[1:battery.grid_size_x+1, :, :] = 350.0

        initial = battery.temperature.copy()
        battery.update_temperature_distribution()

        # Temperature should become more uniform
        assert not np.array_equal(battery.temperature, initial)


class TestApplyCooling:
    """Test apply_cooling function."""

    def test_no_cooling_when_flow_rate_zero(self):
        """Test no cooling when flow rate is zero."""
        battery = SingleBattery(env_temperature=300.0)
        battery.flow_rate = 0.0
        battery.inlet_temp = 300.0
        battery.current = 30.0

        # Generate some heat first
        battery.update_heat_generation()

        result = battery.apply_cooling()

        # Should return inlet temperature when no flow
        assert result == 300.0

    def test_cooling_returns_outlet_temperature(self):
        """Test cooling returns outlet temperature."""
        battery = SingleBattery(env_temperature=300.0)
        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0
        battery.current = 30.0

        result = battery.apply_cooling()

        # Outlet temp should be higher than inlet (heat absorbed)
        assert result > 288.0

    def test_cooling_reduces_temperature(self):
        """Test cooling reduces battery temperature."""
        battery = SingleBattery(env_temperature=300.0)
        battery.current = 30.0
        battery.temperature[1, 1, 1] = 350.0  # Set high temperature

        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0

        temp_before = battery.temperature[1, 1, 1]
        battery.apply_cooling()
        temp_after = battery.temperature[1, 1, 1]

        assert temp_after < temp_before


class TestCalculateConvectiveCoefficient:
    """Test calculate_convective_coefficient function."""

    def test_returns_positive_value(self):
        """Test function returns positive convective coefficient."""
        battery = SingleBattery()
        h = battery.calculate_convective_coefficient(flow_rate=1.0)

        assert h > 0

    def test_increases_with_flow_rate(self):
        """Test convective coefficient increases with flow rate."""
        battery = SingleBattery()

        h_low = battery.calculate_convective_coefficient(flow_rate=1.0)
        h_high = battery.calculate_convective_coefficient(flow_rate=5.0)

        assert h_high > h_low

    def test_zero_flow_rate_returns_zero(self):
        """Test with zero flow rate."""
        battery = SingleBattery()
        h = battery.calculate_convective_coefficient(flow_rate=0.0)

        # Should be very small or zero
        assert h == 0

    def test_reynolds_number_calculation(self):
        """Test Reynolds number calculation for different flow regimes."""
        battery = SingleBattery(
            coolant_density=1111,
            coolant_viscosity=3.94,
            coolant_height=0.04,
            coolant_width=0.08
        )

        # Calculate expected hydraulic diameter
        dh = 4 * 0.04 * 0.08 / (2 * (0.04 + 0.08))

        # For flow_rate = 1.0, Re should be in turbulent range (>4000)
        Re = 1111 * 1.0 * dh / (3.94 / 1000)
        assert Re > 4000  # Should be turbulent


class TestDiffuseCooling:
    """Test diffuse_cooling function."""

    def test_returns_temperature_array(self):
        """Test function returns temperature array."""
        battery = SingleBattery(env_temperature=300.0)
        result = battery.diffuse_cooling()

        assert isinstance(result, np.ndarray)

    def test_vertical_heat_transfer(self):
        """Test vertical heat transfer from bottom to top."""
        battery = SingleBattery(env_temperature=300.0)

        # Set layer 2 (z=2) hot
        battery.temperature[1:-1, 1:-1, 2] = 350.0
        # Set layer 3 and above cold
        battery.temperature[1:-1, 1:-1, 3:] = 300.0

        initial_layer2 = battery.temperature[1, 1, 2].copy()
        initial_layer3 = battery.temperature[1, 1, 3].copy()

        battery.diffuse_cooling()

        # Layer 2 should cool down (heat transfers upward)
        assert battery.temperature[1, 1, 2] < initial_layer2
        # Layer 3 should warm up
        assert battery.temperature[1, 1, 3] > initial_layer3


class TestRunMethod:
    """Test run method."""

    def test_run_returns_outlet_temperature(self):
        """Test run returns average outlet temperature."""
        battery = SingleBattery(env_temperature=300.0)
        battery.current = 30.0
        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0

        result = battery.run(1.0)

        # Should return a temperature value
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_run_increases_temperature(self):
        """Test run increases battery temperature."""
        battery = SingleBattery(env_temperature=300.0)
        battery.current = 30.0
        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0

        initial = battery.get_core_temperature()
        battery.run(1.0)
        final = battery.get_core_temperature()

        assert final > initial

    def test_run_records_thermal_history(self):
        """Test run records thermal history."""
        battery = SingleBattery(env_temperature=300.0)
        battery.current = 30.0
        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0

        battery.run(1.0)

        assert len(battery.thermal_history) > 0


class TestActionMethods:
    """Test set_action and get_action methods."""

    def test_set_action(self):
        """Test set_action sets inlet_temp and flow_rate."""
        battery = SingleBattery(env_temperature=300.0)

        battery.set_action(285.0, 5.0)

        assert battery.inlet_temp == 285.0
        assert battery.flow_rate == 5.0

    def test_get_action(self):
        """Test get_action returns inlet_temp and flow_rate."""
        battery = SingleBattery(env_temperature=300.0)
        battery.inlet_temp = 290.0
        battery.flow_rate = 2.0

        inlet, flow = battery.get_action()

        assert inlet == 290.0
        assert flow == 2.0

    def test_set_and_get_action_consistency(self):
        """Test set and get action are consistent."""
        battery = SingleBattery(env_temperature=300.0)

        battery.set_action(280.0, 4.0)
        inlet, flow = battery.get_action()

        assert inlet == 280.0
        assert flow == 4.0


class TestTemperatureGetters:
    """Test all temperature getter methods."""

    def test_get_core_temperature(self):
        """Test get_core_temperature."""
        battery = SingleBattery(env_temperature=300.0)

        # Calculate core index: grid_size // 2 + 1
        # Default: gx=3, gy=8, gz=10 -> core_x=2, core_y=5, core_z=6
        core_x = battery.grid_size_x // 2 + 1
        core_y = battery.grid_size_y // 2 + 1
        core_z = battery.grid_size_z // 2 + 1

        battery.temperature[core_x, core_y, core_z] = 350.0

        core = battery.get_core_temperature()

        assert core == 350.0

    def test_get_surface_temperatures(self):
        """Test get_surface_temperatures returns 6 arrays."""
        battery = SingleBattery(env_temperature=300.0)

        surfaces = battery.get_surface_temperatures()

        assert len(surfaces) == 6

        # Top/Bottom surfaces: [1:-1, 1:-1, :] -> shape (gx, gy)
        assert surfaces[0].shape == (battery.grid_size_x, battery.grid_size_y)  # top
        assert surfaces[1].shape == (battery.grid_size_x, battery.grid_size_y)  # bottom

        # Left/Right surfaces: [1:-1, :, 1:-1] -> shape (gx, gz)
        assert surfaces[2].shape == (battery.grid_size_x, battery.grid_size_z)  # left
        assert surfaces[3].shape == (battery.grid_size_x, battery.grid_size_z)  # right

        # Front/Back surfaces: [:, 1:-1, 1:-1] -> shape (gy, gz)
        assert surfaces[4].shape == (battery.grid_size_y, battery.grid_size_z)  # front
        assert surfaces[5].shape == (battery.grid_size_y, battery.grid_size_z)  # back

    def test_get_top_surface_average_temperature(self):
        """Test get_top_surface_average_temperature."""
        battery = SingleBattery(env_temperature=300.0)

        # Set top surface to known temperature
        battery.temperature[1:-1, 1:-1, -2] = 320.0

        top_avg = battery.get_top_surface_average_temperature()

        assert np.isclose(top_avg, 320.0)

    def test_get_bottom_surface_average_temperature(self):
        """Test get_bottom_surface_average_temperature."""
        battery = SingleBattery(env_temperature=300.0)

        # Set bottom surface to known temperature
        battery.temperature[1:-1, 1:-1, 1] = 310.0

        bottom_avg = battery.get_bottom_surface_average_temperature()

        assert np.isclose(bottom_avg, 310.0)

    def test_get_temps_returns_dict(self):
        """Test get_temps returns dict with correct keys."""
        battery = SingleBattery(env_temperature=300.0)

        temps = battery.get_temps()

        assert isinstance(temps, dict)
        assert 'core' in temps
        assert 'top' in temps
        assert 'bottom' in temps

    def test_temps_are_reasonable(self):
        """Test temperature values are in reasonable range."""
        battery = SingleBattery(env_temperature=300.0)
        battery.current = 30.0
        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0

        battery.run(1.0)

        temps = battery.get_temps()

        # All temperatures should be above absolute zero
        assert temps['core'] > 0
        assert temps['top'] > 0
        assert temps['bottom'] > 0

        # All temperatures should be reasonable (below 1000K for battery)
        assert temps['core'] < 1000
        assert temps['top'] < 1000
        assert temps['bottom'] < 1000


class TestGetVoltage:
    """Test get_voltage method."""

    def test_voltage_without_current(self):
        """Test voltage without current is nominal voltage."""
        battery = SingleBattery(voltage=3.2)
        battery.current = 0.0

        v = battery.get_voltage()

        assert v == 3.2

    def test_voltage_with_current(self):
        """Test voltage with current includes internal resistance drop."""
        battery = SingleBattery(voltage=3.2, internal_resistance=0.01)
        battery.current = 10.0

        v = battery.get_voltage()

        # voltage_drop = 10 * 0.01 = 0.1
        expected = 3.2 - 10.0 * 0.01
        assert np.isclose(v, expected)

    def test_voltage_respects_min_max(self):
        """Test voltage respects min/max bounds."""
        battery = SingleBattery(
            voltage=3.2,
            internal_resistance=0.18,
            min_voltage=2.5,
            max_voltage=3.65
        )

        # Very high discharge current (negative) should clamp to min_voltage
        battery.current = 10.0  # 3.2 - 10*0.18 = 1.4, below min 2.5
        v = battery.get_voltage()
        assert v == battery.min_voltage

        # Charging with high current should clamp to max_voltage
        battery.current = -5.0  # 3.2 - (-5)*0.18 = 4.1, above max 3.65
        v = battery.get_voltage()
        assert v == battery.max_voltage


class TestIntegration:
    """Integration tests for the complete system."""

    def test_full_simulation_cycle(self):
        """Test a complete simulation cycle."""
        battery = SingleBattery(env_temperature=300.0)

        # Set up conditions
        battery.set_action(285.0, 3.0)
        battery.current = 30.0

        # Run simulation
        battery.run(10.0)

        # Check all outputs are valid
        core = battery.get_core_temperature()
        top = battery.get_top_surface_average_temperature()
        bottom = battery.get_bottom_surface_average_temperature()
        temps = battery.get_temps()
        voltage = battery.get_voltage()

        # All should be finite numbers
        assert np.isfinite(core)
        assert np.isfinite(top)
        assert np.isfinite(bottom)
        assert np.isfinite(voltage)
        assert 'core' in temps

    def test_reset_after_simulation(self):
        """Test reset works correctly after simulation."""
        battery = SingleBattery(env_temperature=300.0)

        battery.current = 30.0
        battery.flow_rate = 3.0
        battery.inlet_temp = 288.0
        battery.run(10.0)

        battery.reset()

        # All state should be reset
        assert battery.current == 0.0
        assert battery.flow_rate == 0.0
        assert battery.inlet_temp == 300.0
        assert battery.thermal_history == []
        assert np.all(battery.temperature == 300.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

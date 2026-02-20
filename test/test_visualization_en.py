"""
Visualization: Battery temperature over time
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

from BatteryEnv.single_battery_module import SingleBattery
import matplotlib.pyplot as plt

def test_temperature_over_time():
    """Test temperature changes under different conditions"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test 1: No cooling vs with cooling
    print("Test 1: No cooling vs with cooling")
    cases = [
        {"flow_rate": 0.0, "inlet_temp": 298, "label": "No cooling (flow=0)"},
        {"flow_rate": 2.0, "inlet_temp": 288, "label": "Cooling (flow=2, inlet=288K)"},
        {"flow_rate": 4.0, "inlet_temp": 288, "label": "Strong cooling (flow=4, inlet=288K)"},
        {"flow_rate": 6.0, "inlet_temp": 288, "label": "Max cooling (flow=6, inlet=288K)"},
    ]

    time_points = []
    core_temps = []
    top_temps = []
    bottom_temps = []
    avg_temps = []

    for case in cases:
        battery = SingleBattery()
        battery.reset()
        battery.current = 30.0
        battery.flow_rate = case["flow_rate"]
        battery.inlet_temp = case["inlet_temp"]

        core_history = []
        top_history = []
        bottom_history = []
        avg_history = []

        # Run 60 seconds, record every second
        for t in range(0, 60, 1):
            battery.run(t_seconds=1)
            core_history.append(battery.get_core_temperature())
            top_history.append(battery.get_top_surface_average_temperature())
            bottom_history.append(battery.get_bottom_surface_average_temperature())
            avg_history.append(np.mean(battery.temperature[1:-1, 1:-1, 1:-1]))

        time_points = list(range(60))
        core_temps.append(core_history)
        top_temps.append(top_history)
        bottom_temps.append(bottom_history)
        avg_temps.append(avg_history)

    # Plot average temperature
    ax1 = axes[0, 0]
    for i, case in enumerate(cases):
        ax1.plot(time_points, avg_temps[i], label=case["label"], linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Average Temperature (K)', fontsize=12)
    ax1.set_title('Average Temperature vs Time', fontsize=14)
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, 60)

    # Plot core temperature
    ax2 = axes[0, 1]
    for i, case in enumerate(cases):
        ax2.plot(time_points, core_temps[i], label=case["label"], linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Core Temperature (K)', fontsize=12)
    ax2.set_title('Core Temperature vs Time', fontsize=14)
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 60)

    # Test 2: Different inlet temperatures
    print("Test 2: Different inlet temperatures")
    inlet_cases = [
        {"flow_rate": 2.0, "inlet_temp": 288, "label": "Inlet 288K (15C)"},
        {"flow_rate": 2.0, "inlet_temp": 290, "label": "Inlet 290K (17C)"},
        {"flow_rate": 2.0, "inlet_temp": 295, "label": "Inlet 295K (22C)"},
    ]

    top_temps2 = []
    bottom_temps2 = []

    for case in inlet_cases:
        battery = SingleBattery()
        battery.reset()
        battery.current = 30.0
        battery.flow_rate = case["flow_rate"]
        battery.inlet_temp = case["inlet_temp"]

        top_history = []
        bottom_history = []

        for t in range(0, 60, 1):
            battery.run(t_seconds=1)
            top_history.append(battery.get_top_surface_average_temperature())
            bottom_history.append(battery.get_bottom_surface_average_temperature())

        top_temps2.append(top_history)
        bottom_temps2.append(bottom_history)

    # Plot top/bottom temperature for different inlet temps
    ax3 = axes[1, 0]
    for i, case in enumerate(inlet_cases):
        ax3.plot(time_points, top_temps2[i], label=f'{case["label"]} Top', linestyle='-', linewidth=2)
        ax3.plot(time_points, bottom_temps2[i], label=f'{case["label"]} Bottom', linestyle='--', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Temperature (K)', fontsize=12)
    ax3.set_title('Top/Bottom Temperature vs Time', fontsize=14)
    ax3.legend(fontsize=8)
    ax3.grid(True)
    ax3.set_xlim(0, 60)

    # Test 3: Temperature difference
    ax4 = axes[1, 1]
    for i, case in enumerate(cases):
        diff = [top_temps[i][t] - bottom_temps[i][t] for t in range(len(time_points))]
        ax4.plot(time_points, diff, label=case["label"], linewidth=2)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Temperature Difference (K)', fontsize=12)
    ax4.set_title('Top - Bottom Temperature Difference', fontsize=14)
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim(0, 60)

    plt.tight_layout()
    plt.savefig('temperature_over_time.png', dpi=150)
    print("Saved to temperature_over_time.png")
    plt.close()

def test_layer_temperature():
    """Test battery layer temperature distribution"""

    print("\nTest: Battery layer temperature distribution")

    battery = SingleBattery()
    battery.reset()
    battery.current = 30.0
    battery.flow_rate = 2.0
    battery.inlet_temp = 288.0

    # Run for 5 minutes then check layer temperatures
    battery.run(t_seconds=300)

    # Plot layer temperatures
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Layer average temperature
    layers = range(1, battery.grid_size_z + 2)
    layer_temps = [np.mean(battery.temperature[1:-1, 1:-1, z]) for z in layers]

    ax1 = axes[0]
    ax1.plot(layers, layer_temps, 'b-o', linewidth=2, markersize=4)
    ax1.axhline(y=battery.inlet_temp, color='r', linestyle='--', label=f'Inlet Temp ({battery.inlet_temp}K)')
    ax1.set_xlabel('Z Layer', fontsize=12)
    ax1.set_ylabel('Average Temperature (K)', fontsize=12)
    ax1.set_title('Z-Direction Temperature Distribution (after 5min)', fontsize=14)
    ax1.legend()
    ax1.grid(True)
    ax1.axvline(x=1, color='g', linestyle=':', alpha=0.7, label='Bottom')
    ax1.axvline(x=battery.grid_size_z, color='orange', linestyle=':', alpha=0.7, label='Top')

    # Temperature distribution at different times
    times = [60, 180, 300]
    ax2 = axes[1]

    for t in times:
        battery = SingleBattery()
        battery.reset()
        battery.current = 30.0
        battery.flow_rate = 2.0
        battery.inlet_temp = 288.0
        battery.run(t_seconds=t)

        layer_temps = [np.mean(battery.temperature[1:-1, 1:-1, z]) for z in layers]
        ax2.plot(layers, layer_temps, label=f'{t}s', linewidth=2)

    ax2.set_xlabel('Z Layer', fontsize=12)
    ax2.set_ylabel('Average Temperature (K)', fontsize=12)
    ax2.set_title('Z-Direction Temperature at Different Times', fontsize=14)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('layer_temperature.png', dpi=150)
    print("Saved to layer_temperature.png")
    plt.close()

if __name__ == "__main__":
    test_temperature_over_time()
    test_layer_temperature()
    print("Done!")

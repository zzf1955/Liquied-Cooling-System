"""
Visualization: Long-term battery temperature stability
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/data/hejiakai/Liquied-Cooling-System')

from BatteryEnv.single_battery_module import SingleBattery
import matplotlib.pyplot as plt

def test_long_term_stability():
    """Test long-term temperature stability (2 hours)"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test: 2 hours with different cooling conditions
    print("Test: 2 hours simulation")
    cases = [
        {"flow_rate": 0.0, "inlet_temp": 298, "label": "No cooling (flow=0)"},
        {"flow_rate": 2.0, "inlet_temp": 288, "label": "Cooling (flow=2, inlet=288K)"},
        {"flow_rate": 4.0, "inlet_temp": 288, "label": "Strong cooling (flow=4, inlet=288K)"},
        {"flow_rate": 6.0, "inlet_temp": 288, "label": "Max cooling (flow=6, inlet=288K)"},
    ]

    # Record every 60 seconds for 2 hours = 7200 seconds
    record_times = list(range(0, 7200, 60))  # Every 60 seconds
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

        # Run 2 hours, record every 60 seconds
        for t in range(0, 7200, 60):
            battery.run(t_seconds=60)
            core_history.append(battery.get_core_temperature())
            top_history.append(battery.get_top_surface_average_temperature())
            bottom_history.append(battery.get_bottom_surface_average_temperature())
            avg_history.append(np.mean(battery.temperature[1:-1, 1:-1, 1:-1]))

        time_points = record_times
        core_temps.append(core_history)
        top_temps.append(top_history)
        bottom_temps.append(bottom_history)
        avg_temps.append(avg_history)

    # Convert to hours
    time_hours = [t/3600 for t in time_points]

    # Plot average temperature
    ax1 = axes[0, 0]
    for i, case in enumerate(cases):
        ax1.plot(time_hours, avg_temps[i], label=case["label"], linewidth=2)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Average Temperature (K)', fontsize=12)
    ax1.set_title('Average Temperature vs Time (2 hours)', fontsize=14)
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, 2)

    # Plot core temperature
    ax2 = axes[0, 1]
    for i, case in enumerate(cases):
        ax2.plot(time_hours, core_temps[i], label=case["label"], linewidth=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Core Temperature (K)', fontsize=12)
    ax2.set_title('Core Temperature vs Time (2 hours)', fontsize=14)
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 2)

    # Plot top/bottom temperature
    ax3 = axes[1, 0]
    for i, case in enumerate(cases):
        ax3.plot(time_hours, top_temps[i], label=f'{case["label"]} Top', linestyle='-', linewidth=2)
        ax3.plot(time_hours, bottom_temps[i], label=f'{case["label"]} Bottom', linestyle='--', linewidth=2)
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_ylabel('Temperature (K)', fontsize=12)
    ax3.set_title('Top/Bottom Temperature vs Time', fontsize=14)
    ax3.legend(fontsize=7)
    ax3.grid(True)
    ax3.set_xlim(0, 2)

    # Plot temperature difference
    ax4 = axes[1, 1]
    for i, case in enumerate(cases):
        diff = [top_temps[i][t] - bottom_temps[i][t] for t in range(len(time_points))]
        ax4.plot(time_hours, diff, label=case["label"], linewidth=2)
    ax4.set_xlabel('Time (hours)', fontsize=12)
    ax4.set_ylabel('Temperature Difference (K)', fontsize=12)
    ax4.set_title('Top - Bottom Temperature Difference', fontsize=14)
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim(0, 2)

    plt.tight_layout()
    plt.savefig('temperature_long_term.png', dpi=150)
    print("Saved to temperature_long_term.png")
    plt.close()

    # Print final values
    print("\nFinal values (after 2 hours):")
    for i, case in enumerate(cases):
        print(f"{case['label']}:")
        print(f"  Average: {avg_temps[i][-1]:.2f} K")
        print(f"  Core: {core_temps[i][-1]:.2f} K")
        print(f"  Top: {top_temps[i][-1]:.2f} K")
        print(f"  Bottom: {bottom_temps[i][-1]:.2f} K")
        print(f"  Temp diff: {top_temps[i][-1] - bottom_temps[i][-1]:.2f} K")

def test_different_inlet_temps():
    """Test different inlet temperatures over long term"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cases = [
        {"flow_rate": 2.0, "inlet_temp": 288, "label": "Inlet 288K (15C)"},
        {"flow_rate": 2.0, "inlet_temp": 290, "label": "Inlet 290K (17C)"},
        {"flow_rate": 2.0, "inlet_temp": 295, "label": "Inlet 295K (22C)"},
    ]

    record_times = list(range(0, 7200, 60))
    time_hours = [t/3600 for t in record_times]

    avg_temps = []
    temp_diffs = []

    for case in cases:
        battery = SingleBattery()
        battery.reset()
        battery.current = 30.0
        battery.flow_rate = case["flow_rate"]
        battery.inlet_temp = case["inlet_temp"]

        avg_history = []
        diff_history = []

        for t in range(0, 7200, 60):
            battery.run(t_seconds=60)
            avg_history.append(np.mean(battery.temperature[1:-1, 1:-1, 1:-1]))
            top = battery.get_top_surface_average_temperature()
            bottom = battery.get_bottom_surface_average_temperature()
            diff_history.append(top - bottom)

        avg_temps.append(avg_history)
        temp_diffs.append(diff_history)

    # Plot average temperature
    ax1 = axes[0]
    for i, case in enumerate(cases):
        ax1.plot(time_hours, avg_temps[i], label=case["label"], linewidth=2)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Average Temperature (K)', fontsize=12)
    ax1.set_title('Inlet Temperature Effect on Average Temp', fontsize=14)
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, 2)

    # Plot temperature difference
    ax2 = axes[1]
    for i, case in enumerate(cases):
        ax2.plot(time_hours, temp_diffs[i], label=case["label"], linewidth=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Temperature Difference (K)', fontsize=12)
    ax2.set_title('Top - Bottom Temperature Difference', fontsize=14)
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 2)

    plt.tight_layout()
    plt.savefig('temperature_inlet_effect.png', dpi=150)
    print("Saved to temperature_inlet_effect.png")
    plt.close()

if __name__ == "__main__":
    test_long_term_stability()
    test_different_inlet_temps()
    print("Done!")

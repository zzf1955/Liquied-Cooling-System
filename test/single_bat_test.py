import sys
sys.path.insert(0, '/mnt/g/github_project/Liquied-Cooling-System')

import numpy as np
import matplotlib.pyplot as plt
from BatteryEnv.single_battery_module import SingleBattery
import time

def run_simulation(battery, duration, current, flow, inlet_t):
    """运行模拟并记录关键数据"""
    battery.reset()
    battery.current = current
    battery.flow_rate = flow
    battery.inlet_temp = inlet_t
    
    data = {
        'time': [],
        'core': [],
        'top': [],
        'bottom': [],
        'outlet': []
    }
    
    num_steps = int(duration / battery.dt)
    # 抽样记录，防止数据量过大
    record_interval = max(1, int(1.0 / battery.dt)) 
    
    for i in range(num_steps):
        battery.update_heat_generation()
        battery.update_temperature_distribution()
        out_t = battery.apply_cooling()
        battery.diffuse_cooling()
        
        if i % record_interval == 0:
            data['time'].append(i * battery.dt)
            data['core'].append(battery.get_core_temperature())
            data['top'].append(battery.get_top_surface_average_temperature())
            data['bottom'].append(battery.get_bottom_surface_average_temperature())
            data['outlet'].append(out_t)
            
    return data

def test_battery_performance():
    print("开始电池热管理物理测试...")
    duration = 600  # 模拟 10 分钟 (600秒)
    current = 30.0  # 初始电流 30A
    
    # --- 1. 核心、底部、顶部温差测试 (有冷却) ---
    print("正在测试内部温差分布...")
    bat_gradient = SingleBattery()
    data_gradient = run_simulation(bat_gradient, duration, current, flow=2.0, inlet_t=290)
    
    # --- 2. 有无冷却对比测试 ---
    print("正在对比有无冷却效果...")
    bat_no_cool = SingleBattery()
    data_no_cool = run_simulation(bat_no_cool, duration, current, flow=0.0, inlet_t=300)
    
    # --- 3. 流速敏感性测试 ---
    print("正在进行流速敏感性测试...")
    flows = [0.5, 2, 5]
    flow_results = {}
    for f in flows:
        bat = SingleBattery()
        flow_results[f] = run_simulation(bat, duration, current, flow=f, inlet_t=290)
        
    # --- 4. 入口温度敏感性测试 ---
    print("正在进行入口温度敏感性测试...")
    temps = [285, 290, 295]
    temp_results = {}
    for t in temps:
        bat = SingleBattery()
        temp_results[t] = run_simulation(bat, duration, current, flow=2.0, inlet_t=t)

    # ================= 可视化 =================
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: 内部温差 (Gradient)
    axs[0, 0].plot(data_gradient['time'], data_gradient['core'], label='Core Temp', linewidth=2)
    axs[0, 0].plot(data_gradient['time'], data_gradient['top'], label='Top Surface', linestyle='--')
    axs[0, 0].plot(data_gradient['time'], data_gradient['bottom'], label='Bottom Surface (Cooling side)', linestyle=':')
    axs[0, 0].set_title("Internal Temperature Distribution (Flow=2m/s, Inlet=290K)")
    axs[0, 0].set_ylabel("Temp (K)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: 有无冷却对比
    axs[0, 1].plot(data_no_cool['time'], data_no_cool['core'], label='No Cooling (0 m/s)', color='red')
    axs[0, 1].plot(data_gradient['time'], data_gradient['core'], label='With Cooling (2 m/s)', color='green')
    axs[0, 1].set_title("Cooling vs. No Cooling (Core Temperature @ 30A)")
    axs[0, 1].set_ylabel("Temp (K)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: 流速影响
    for f, d in flow_results.items():
        axs[1, 0].plot(d['time'], d['core'], label=f'Flow: {f} m/s')
    axs[1, 0].set_title("Sensitivity: Coolant Flow Rate")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Core Temp (K)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: 入口温度影响
    for t, d in temp_results.items():
        axs[1, 1].plot(d['time'], d['core'], label=f'Inlet: {t} K')
    axs[1, 1].set_title("Sensitivity: Coolant Inlet Temperature")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Core Temp (K)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('battery_physics_test.png')
    print("测试完成！图像已保存为 battery_physics_test.png")
    plt.show()

def run_comprehensive_tests():
    print("开始执行全维度物理验证测试...")
    duration = 800  # 模拟时长
    bat = SingleBattery()

    # --- Case 1: 内部温差分析 (30A, 2m/s, 290K) ---
    print("正在运行 Case 1: 内部温差分析...")
    bat.reset()
    bat.current, bat.flow_rate, bat.inlet_temp = 30.0, 2.0, 290.0
    res1 = {'t':[], 'core':[], 'top':[], 'bottom':[]}
    for i in range(int(duration/bat.dt)):
        bat.update_heat_generation(); bat.update_temperature_distribution(); bat.apply_cooling()
        if i % 100 == 0:
            ts = bat.get_temps()
            res1['t'].append(i*bat.dt); res1['core'].append(ts['core'])
            res1['top'].append(ts['top']); res1['bottom'].append(ts['bottom'])

    # --- Case 2: 冷却 vs 无冷却对比 (30A) ---
    print("正在运行 Case 2: 冷却对比...")
    bat.reset()
    bat.current, bat.flow_rate = 30.0, 0.0 # 无冷却
    res2_none = []
    for i in range(int(duration/bat.dt)):
        bat.update_heat_generation(); bat.update_temperature_distribution(); bat.apply_cooling()
        if i % 100 == 0: res2_none.append(bat.get_core_temp_only())

    # --- Case 3 & 4: 流速与入口温度敏感性 ---
    print("正在运行 Case 3 & 4: 敏感性分析...")
    flow_tests = [0.5, 2, 5]
    inlet_tests = [285, 290, 295]
    res_flow = {}; res_inlet = {}
    
    for f in flow_tests:
        bat.reset(); bat.current, bat.flow_rate, bat.inlet_temp = 30.0, f, 290.0
        tmp = []
        for _ in range(int(duration/bat.dt)):
            bat.update_heat_generation(); bat.update_temperature_distribution(); bat.apply_cooling()
            tmp.append(bat.get_core_temp_only())
        res_flow[f] = tmp[::100]

    for it in inlet_tests:
        bat.reset(); bat.current, bat.flow_rate, bat.inlet_temp = 30.0, 2.0, it
        tmp = []
        for _ in range(int(duration/bat.dt)):
            bat.update_heat_generation(); bat.update_temperature_distribution(); bat.apply_cooling()
            tmp.append(bat.get_core_temp_only())
        res_inlet[it] = tmp[::100]

    # --- Case 5: 能量守恒验证 ---
    print("正在运行 Case 5: 能量守恒验证...")
    bat.reset()
    bat.current, bat.flow_rate, bat.inlet_temp = 50.0, 2.0, 290.0
    p_gen, p_diss, p_delta_E, t_eb = [], [], [], []
    vol_total = (bat.grid_size_x*bat.grid_size_y*bat.grid_size_z) * (bat.cell_length**3)
    for i in range(int(300/bat.dt)):
        E_start = np.sum(bat.temperature[1:-1,1:-1,1:-1]) * bat.density * (bat.cell_length**3) * bat.specific_heat
        pow_g = bat.update_heat_generation()
        bat.update_temperature_distribution()
        bat.apply_cooling()
        E_end = np.sum(bat.temperature[1:-1,1:-1,1:-1]) * bat.density * (bat.cell_length**3) * bat.specific_heat
        if i % 50 == 0:
            t_eb.append(i*bat.dt); p_gen.append(pow_g); p_diss.append(bat.last_step_heat/bat.dt)
            p_delta_E.append((E_end - E_start)/bat.dt)

    # ================= 可视化汇总 =================
    plt.figure(figsize=(20, 12))
    
    # 子图1: 内部温差
    plt.subplot(2, 3, 1)
    plt.plot(res1['t'], res1['core'], label='Core (Hottest)')
    plt.plot(res1['t'], res1['top'], label='Top Surface')
    plt.plot(res1['t'], res1['bottom'], label='Bottom (Cooling)')
    plt.title("1. Internal Temp Distribution (30A, 2m/s)")
    plt.ylabel("Temp (K)"); plt.legend(); plt.grid(True)

    # 子图2: 冷却对比
    plt.subplot(2, 3, 2)
    plt.plot(res1['t'], res2_none, label='No Cooling (0m/s)', color='red')
    plt.plot(res1['t'], res1['core'], label='With Cooling (2m/s)', color='green')
    plt.title("2. Cooling vs No-Cooling Impact")
    plt.ylabel("Core Temp (K)"); plt.legend(); plt.grid(True)

    # 子图3: 流速影响
    plt.subplot(2, 3, 3)
    for f, data in res_flow.items(): plt.plot(res1['t'], data, label=f'{f} m/s')
    plt.title("3. Sensitivity: Flow Rate")
    plt.legend(); plt.grid(True)

    # 子图4: 入口温度影响
    plt.subplot(2, 3, 4)
    for it, data in res_inlet.items(): plt.plot(res1['t'], data, label=f'{it} K')
    plt.title("4. Sensitivity: Inlet Temp")
    plt.xlabel("Time (s)"); plt.legend(); plt.grid(True)

    # 子图5: 能量平衡
    plt.subplot(2, 3, 5)
    plt.plot(t_eb, p_gen, label='Power In (Generation)')
    plt.plot(t_eb, p_diss, label='Power Out (Cooling)')
    plt.plot(t_eb, p_delta_E, '--', label='Energy Change Rate (dE/dt)')
    plt.title("5. Energy Balance (P_in = P_out + dE/dt)")
    plt.xlabel("Time (s)"); plt.ylabel("Power (W)"); plt.legend(); plt.grid(True)

    # 子图6: 截面云图
    plt.subplot(2, 3, 6)
    slice_yz = bat.temperature[bat.grid_size_x//2+1, 1:-1, 1:-1].T
    plt.imshow(slice_yz, extent=[0, 0.1747, 0, 0.20747], origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(label='K'); plt.title("6. Final Y-Z Temp Slice (Mid-X)")
    plt.xlabel("Width (m)"); plt.ylabel("Height (m)")

    plt.tight_layout()
    plt.show()

# 辅助函数
SingleBattery.get_core_temp_only = lambda self: self.temperature[self.grid_size_x//2+1, self.grid_size_y//2+1, self.grid_size_z//2+1]
    

if __name__ == "__main__":
    # test_battery_performance()
    run_comprehensive_tests()
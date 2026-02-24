import numpy as np
from loguru import logger

class SingleBattery:
    def __init__(self, 
                 # 真实电池参数
                 voltage=3.2,              # 标称电压 (V)
                 internal_resistance=0.18,  # 内阻 (Ω)
                 length=0.07165,           # 长度 (m) (转换自71.65mm)
                 width=0.1747,             # 宽度 (m) (转换自174.70mm)
                 height=0.20747,           # 高度 (m) (转换自207.47mm)
                 rated_capacity=280,       # 额定容量 (Ah)
                 min_voltage=2.5,          # 最小工作电压 (V)
                 max_voltage=3.65,         # 最大工作电压 (V)
                 
                 # 电池材料参数
                 density=2700,             # 密度 (kg/m^3)
                 specific_heat=900,        # 比热容 (J/kg·K)
                 thermal_conductivity=237, # 热导率 (W/m·K)
                 
                 # 冷却液参数
                 coolant_density=1111,     # 冷却液密度 (kg/m^3)
                 coolant_specific_heat=3465, # 冷却液比热容 (J/kg·K)
                 coolant_conductivity=0.503, # 冷却液导热系数 (W/m·K)
                 coolant_viscosity=3.94,   # 冷却液黏度参数
                 coolant_width = 0.08,      # 冷却液宽度 (m)
                 coolant_height = 0.04,     # 冷却液高度 (m)
                 
                 # 环境参数
                 env_temperature=300.0,    # 环境温度 (K)
                ):
        
        # 调整系数（用来调整电池冷却温度的传递的传递)
        # 增加此系数以加快热传导，使动作效果在约10步内可见
        self.adjusting_factor = 3.0

        # 电池尺寸参数
        self.length, self.width, self.height = length, width, height
        self.cell_length = 0.02            # 单元格长度 (m)
        
        # 电网格参数
        self.grid_size_x = int(self.length / self.cell_length)
        self.grid_size_y = int(self.width / self.cell_length)
        self.grid_size_z = int(self.height / self.cell_length)

        # 添加参数合理性检查
        assert max_voltage > min_voltage, "电压范围无效"
        assert self.cell_length < min(length, width, height), "网格尺寸过大"
        
        # 电池电气参数
        self.voltage = voltage
        self.internal_resistance = internal_resistance
        self.rated_capacity = rated_capacity
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.current = 0                   # 初始电流为0
        
        # 电池热学参数
        self.density = density
        self.specific_heat = specific_heat
        self.thermal_conductivity = thermal_conductivity
        
        # 冷却液参数
        self.coolant_density = coolant_density
        self.coolant_specific_heat = coolant_specific_heat
        self.coolant_conductivity = coolant_conductivity
        self.flow_rate = 0.0
        self.inlet_temp = env_temperature  # 液冷温度初始化为环境温度
        self.coolant_viscosity = coolant_viscosity
        self.coolant_kinematic_viscosity = coolant_viscosity / coolant_density # 运动粘度 (m2/s)
        self.coolant_width = coolant_width
        self.coolant_height = coolant_height
        
        # 环境参数
        self.env_temperature = env_temperature
        
        # 初始化温度网格 (考虑实际尺寸)
        self.temperature = np.full((self.grid_size_x + 2, self.grid_size_y + 2, self.grid_size_z + 2), env_temperature,dtype=np.float64)
        
        # 热扩散率和时间步长
        self.alpha = self.thermal_conductivity / (self.density * self.specific_heat)
        self.dt = self.cell_length**2 / (6 * self.alpha) * 0.95 # 稳定性条件
        
        self.thermal_history = []  # 添加温度记录
        
        self.last_step_heat = 0.0  # 记录上一步吸收的热量(J)  # TODO
    
    def reset(self):
        """重置电池状态"""
        self.temperature.fill(self.env_temperature)
        self.current = 0.0
        self.flow_rate = 0.0
        self.inlet_temp = self.env_temperature
        self.thermal_history = []
        
    def update_heat_generation(self):
        # 电池中心位置
        center_x = self.grid_size_x // 2 + 1
        center_y = self.grid_size_y // 2 + 1
        center_z = self.grid_size_z // 2 + 1
        
        # 热量产生: Q = I^2 * R * dt
        heat_generation = self.current**2 * self.internal_resistance * self.dt
        
        # 将热量转化为温升: ΔT = Q / (rho * V * Cp)
        volume = self.cell_length ** 3
        self.temperature[center_x, center_y, center_z] += heat_generation / (self.density * volume * self.specific_heat)
        
        # logger.debug(f"电池中心温度: {self.temperature[center_x, center_y, center_z]}")
        
    def update_temperature_distribution(self):
        """
        更新电池内部温度分布（热扩散）
        简化版本，标准热扩散方程，绝热边界

        向量化实现版本：使用 NumPy 广播机制进行三维热扩散计算
        """
        T = self.temperature
        gx, gy, gz = self.grid_size_x, self.grid_size_y, self.grid_size_z

        # 先应用绝热边界条件，确保边界值等于相邻内部值
        # 这样在计算拉普拉斯算子时能正确使用绝热边界
        T[0, :, :] = T[1, :, :]
        T[-1, :, :] = T[-2, :, :]
        T[:, 0, :] = T[:, 1, :]
        T[:, -1, :] = T[:, -2, :]
        T[:, :, 0] = T[:, :, 1]
        T[:, :, -1] = T[:, :, -2]

        # 计算拉普拉斯算子 (Standard 7-point stencil for 3D)
        # T[x, y, z] 的变化取决于上下左右前后的温差
        # 注意：原始代码中 i 范围是 1 到 gx，对应 numpy 索引 1 到 gx+1
        # 所以 T[1:gx+2] 对应内部格点，T[2:gx+2] 对应 i+1，T[:gx+1] 对应 i-1
        delta_T = (
            T[2:gx+2, 1:gy+1, 1:gz+1] + T[:gx, 1:gy+1, 1:gz+1] +  # X 方向
            T[1:gx+1, 2:gy+2, 1:gz+1] + T[1:gx+1, :gy, 1:gz+1] +  # Y 方向
            T[1:gx+1, 1:gy+1, 2:gz+2] + T[1:gx+1, 1:gy+1, :gz] -  # Z 方向
            6 * T[1:gx+1, 1:gy+1, 1:gz+1]
        )

        # 更新温度网格（仅更新内部点）
        self.temperature[1:gx+1, 1:gy+1, 1:gz+1] += self.alpha * self.dt / self.cell_length**2 * delta_T

        # 再次应用绝热边界条件
        self.temperature[0, :, :] = self.temperature[1, :, :]
        self.temperature[-1, :, :] = self.temperature[-2, :, :]
        self.temperature[:, 0, :] = self.temperature[:, 1, :]
        self.temperature[:, -1, :] = self.temperature[:, -2, :]
        self.temperature[:, :, 0] = self.temperature[:, :, 1]
        self.temperature[:, :, -1] = self.temperature[:, :, -2]

        return self.temperature
    
    def apply_cooling(self):
        """
        应用底部液冷散热
        返回: 冷却液在本个 dt 步长结束时的出口温度 (K)
        """
        # 如果没有启动液冷，那么此时没有热量被吸收
        if self.flow_rate <= 0:
            self.last_step_heat = 0.0
            return self.inlet_temp

        # 计算对流换热系数 (与流速相关)
        h = self.calculate_convective_coefficient(self.flow_rate)

        # 增强冷却效果的系数 - 适度增加
        cooling_boost = 10.0

        # 1. 确定参与换热的层（底部层）
        bottom_layer_indices = (slice(1, self.grid_size_x+1), slice(1, self.grid_size_y+1), 1)
        layer2_indices = (slice(1, self.grid_size_x+1), slice(1, self.grid_size_y+1), 2)

        # 2. 计算本步热交换能量 Q (单位: J)
        # 公式: Q = h * A * (T_surface - T_inlet) * dt * cooling_boost
        heat_exchange = h * self.cell_length**2 * (self.temperature[bottom_layer_indices] - self.inlet_temp) * self.dt * cooling_boost

        # 同时也考虑第二层的一定换热（模拟对流穿透效应）
        heat_exchange_layer2 = heat_exchange * 0.5
        
        # 3. 更新电池内部温度 (ΔT = Q / (rho * V * Cp))
        volume = self.cell_length ** 3
        self.temperature[bottom_layer_indices] -= heat_exchange / (self.density * volume * self.specific_heat)
        self.temperature[layer2_indices] -= heat_exchange_layer2 / (self.density * volume * self.specific_heat)
        
        # 4. 计算本步总吸热量，并计算出口温度
        self.last_step_heat = np.sum(heat_exchange) + np.sum(heat_exchange_layer2)
        
        pipe_area = self.coolant_height * self.coolant_width  # 管道横截面积 (m^2)
        mass_this_step = self.coolant_density * self.flow_rate * pipe_area * self.dt
                
        if mass_this_step > 0:
            return self.inlet_temp + self.last_step_heat / (mass_this_step * self.coolant_specific_heat)
        
        return self.inlet_temp
    
    def calculate_convective_coefficient(self, flow_rate):
        """
        采用标准管内强制对流换热关联式，自动区分层流 / 过渡 / 湍流。
        粘度 self.coolant_viscosity Pa · s。
        """
        # ── 物性、几何 ───────────────────────────────
        mu  = self.coolant_viscosity / 1000       # Pa·s 
        rho  = self.coolant_density                   # kg·m⁻³
        k  = self.coolant_conductivity              # W·m⁻¹·K⁻¹
        cp = self.coolant_specific_heat             # J·kg⁻¹·K⁻¹

        dh = 4 * self.coolant_height * self.coolant_width / (2 * (self.coolant_height + self.coolant_width))  # 水力直径 (m)

        # ── 无量纲数 ────────────────────────────────
        Re = rho * flow_rate * dh / mu
        Pr = mu * cp / k

        # ── 努塞尔数 ────────────────────────────────
        if Re < 2300:  # 层流，恒通量条件
            Nu = 1.86 * (Re * Pr * dh / self.coolant_width) ** (1/3)
        elif Re < 4000:  # 过渡区，线性插值
            Nu_lam = 1.86 * (Re * Pr * dh / self.coolant_width) ** (1/3)
            Nu_turb = 0.023 * Re**0.8 * Pr**0.3
            f = (Re - 2300) / (4000 - 2300)
            Nu = (1 - f) * Nu_lam + f * Nu_turb
        else:  # 湍流 Dittus-Boelter
            Nu = 0.023 * Re**0.8 * Pr**0.4

        h = Nu * k / dh   # W·m⁻²·K⁻¹
        
        # logger.debug(f"Re={Re:.1e}, Pr={Pr:.1e}, Nu={Nu:.1f}, h={h:.1f} W/m²K")

        return h
    
    def diffuse_cooling(self):
        """
        将底部冷却后的温度扩散到整个三维电池。
        扩散是从底面 z=1 向上扩散，z方向为主。
        使用绝热边界条件 - 热量不流出电池。

        向量化实现版本
        修改：减小 diffusion_factor 以增加温差
        """
        # 扩散因子 - 减小以增加电池内部垂直温差
        diffusion_factor = 0.005

        T = self.temperature
        gx, gy, gz = self.grid_size_x, self.grid_size_y, self.grid_size_z

        # 计算 z 方向温差 (仅内部格点)
        # z_diffusion[i,j,k] = T[i,j,k] - T[i,j,k+1] 表示热量从下往上传递
        # 对应原代码: z_diffusion = T[i,j,k-1] - T[i,j,k]
        z_diffusion = T[1:gx+1, 1:gy+1, 1:gz] - T[1:gx+1, 1:gy+1, 2:gz+1]

        # 更新温度 (从第2层开始，第1层是绝热边界不更新)
        # 注意：原代码 k 范围是 1 到 gz，对应 z 索引 1 到 gz
        # 这里我们只更新内部格点，边界层由后续绝热边界处理
        self.temperature[1:gx+1, 1:gy+1, 2:gz+1] += (
            self.adjusting_factor * diffusion_factor * self.alpha * self.dt / self.cell_length**2 * z_diffusion
        )

        # 绝热边界条件处理
        # 底部 (k=1): 热量不流出 - 已经在上面跳过更新
        # 其他边界: 使用相邻内部层的值
        self.temperature[0, :, :] = self.temperature[1, :, :]
        self.temperature[-1, :, :] = self.temperature[-2, :, :]
        self.temperature[:, 0, :] = self.temperature[:, 1, :]
        self.temperature[:, -1, :] = self.temperature[:, -2, :]
        self.temperature[:, :, 0] = self.temperature[:, :, 1]
        self.temperature[:, :, -1] = self.temperature[:, :, -2]

        return self.temperature
    
    def run(self, t_seconds):
        """单体测试运行接口"""
        num_steps = int(t_seconds / self.dt)
        
        # 累积吸热量
        cumulative_q = 0.0
        
        for _ in range(num_steps):
            self.update_heat_generation()
            self.update_temperature_distribution()
            out_t = self.apply_cooling()
            cumulative_q += self.last_step_heat
            
            # 只有当有冷却液流动时才启用垂直方向的热扩散
            if self.flow_rate > 0:
                self.diffuse_cooling()
            
            self.thermal_history.append(self.get_core_temperature())
        
        # 返回测试期间的平均出口温升供参考
        total_mass = self.coolant_density * self.flow_rate * (self.coolant_height * self.coolant_width) * t_seconds
        
        return self.inlet_temp + cumulative_q / (total_mass * self.coolant_specific_heat) if total_mass > 0 else self.inlet_temp
    
    def set_action(self, tmp, flow_rate):
        self.inlet_temp = tmp
        self.flow_rate = flow_rate

    def get_action(self):
        return self.inlet_temp, self.flow_rate

    def get_surface_temperatures(self):
        return (self.temperature[1:-1, 1:-1, -2],  # 顶部
                self.temperature[1:-1, 1:-1, 1],   # 底部
                self.temperature[1:-1, 1, 1:-1],   # 左侧
                self.temperature[1:-1, -2, 1:-1],  # 右侧
                self.temperature[1, 1:-1, 1:-1],   # 前面
                self.temperature[-2, 1:-1, 1:-1])  # 后面

    def get_core_temperature(self):
        # 计算核心位置的索引
        core_x = self.grid_size_x // 2 + 1
        core_y = self.grid_size_y // 2 + 1
        core_z = self.grid_size_z // 2 + 1
        
        # 获取核心温度
        core_temperature = self.temperature[core_x, core_y, core_z]
        return core_temperature

    def get_top_surface_average_temperature(self):
        # 获取顶面温度分布
        top_surface_temperatures = self.get_surface_temperatures()[0]
        # 计算平均温度
        average_temperature = np.mean(top_surface_temperatures)
        return average_temperature

    def get_bottom_surface_average_temperature(self):
        # 获取底面温度分布
        bottom_surface_temperatures = self.get_surface_temperatures()[1]
        # 计算平均温度
        average_temperature = np.mean(bottom_surface_temperatures)
        return average_temperature
    
    def get_temps(self):
        return {
            'core': self.temperature[self.grid_size_x//2+1, self.grid_size_y//2+1, self.grid_size_z//2+1],
            'top': np.mean(self.temperature[1:-1, 1:-1, -2]),
            'bottom': np.mean(self.temperature[1:-1, 1:-1, 1])
        }

    def get_voltage(self):
        """计算电池当前电压"""
        # 考虑内阻压降
        voltage_drop = self.current * self.internal_resistance
        current_voltage = self.voltage - voltage_drop
        
        # 确保电压在允许范围内
        current_voltage = max(self.min_voltage, min(self.max_voltage, current_voltage))
        
        return current_voltage
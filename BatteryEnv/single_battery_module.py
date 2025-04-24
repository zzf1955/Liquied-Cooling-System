import numpy as np

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
                 cooling_flow_direction='x',  # 冷却液流动方向
                 
                 # 环境参数
                 env_temperature=298.0,    # 环境温度 (K)
                 flow_rate=0.1,            # 冷却液流速 (m/s)
                 inlet_temp=294            # 冷却液入口温度 (K)
                ):
        
        # 电池尺寸参数
        self.length = length
        self.width = width
        self.height = height
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
        self.flow_rate = flow_rate
        self.inlet_temp = inlet_temp
        self.coolant_viscosity = coolant_viscosity
        self.coolant_kinematic_viscosity = coolant_viscosity / coolant_density # 运动粘度 (m2/s)
        
        # 环境参数
        self.env_temperature = env_temperature
        
        # 初始化温度网格 (考虑实际尺寸)
        self.temperature = np.full((self.grid_size_x + 2, self.grid_size_y + 2, self.grid_size_z + 2), env_temperature)
        
        # 热扩散率和时间步长
        self.alpha = self.thermal_conductivity / (self.density * self.specific_heat)
        self.dt = self.cell_length**2 / (6 * self.alpha)  # 稳定性条件
        
        self.thermal_history = []  # 添加温度记录
        self.coolant_temp_accumulator = inlet_temp  # 温度累积器

        # 对流换热系数 (基于冷却液参数计算)
        # self.convective_heat_transfer_coefficient = 500  # 初始值

    def reset(self):
        # 将温度网格重置为环境温度
        self.temperature.fill(self.env_temperature)
        self.current = 0

    def update_heat_generation(self):
        # 电池中心位置
        center_x = self.grid_size_x // 2 + 1
        center_y = self.grid_size_y // 2 + 1
        center_z = self.grid_size_z // 2 + 1
        
        # 热量产生: I^2 * R
        heat_generation = self.current**2 * self.internal_resistance
        
        # 更新中心温度
        self.temperature[center_x, center_y, center_z] += heat_generation / (self.density * self.specific_heat * self.cell_length**3) * self.dt

    def run(self, t_seconds):
        num_steps = int(t_seconds / self.dt)
        for _ in range(num_steps):
            self.update_heat_generation()
            self.temperature = self.update_temperature_distribution()
            self.apply_cooling()
            self.thermal_history.append(self.get_core_temperature())

    def update_temperature_distribution(self):
        new_temp = np.copy(self.temperature)
        
        # 更新内部温度分布
        for i in range(1, self.grid_size_x + 1):
            for j in range(1, self.grid_size_y + 1):
                for k in range(1, self.grid_size_z + 1):
                    new_temp[i, j, k] = self.temperature[i, j, k] + self.alpha * self.dt / self.cell_length**2 * (
                        self.temperature[i+1, j, k] + self.temperature[i-1, j, k] +
                        self.temperature[i, j+1, k] + self.temperature[i, j-1, k] +
                        self.temperature[i, j, k+1] + self.temperature[i, j, k-1] -
                        6 * self.temperature[i, j, k])
        
        # 设置边界单元温度
        # 沿 x 轴的边界
        new_temp[0, :, :] = new_temp[1, :, :]
        new_temp[self.grid_size_x + 1, :, :] = new_temp[self.grid_size_x, :, :]
        
        # 沿 y 轴的边界
        new_temp[:, 0, :] = new_temp[:, 1, :]
        new_temp[:, self.grid_size_y + 1, :] = new_temp[:, self.grid_size_y, :]
        
        # 沿 z 轴的边界
        new_temp[:, :, 0] = new_temp[:, :, 1]
        new_temp[:, :, self.grid_size_z + 1] = new_temp[:, :, self.grid_size_z]

        return new_temp

    def apply_cooling(self):
        effective_inlet_temp = self.coolant_temp_accumulator
        # 计算对流换热系数 (与流速相关)
        h = self.calculate_convective_coefficient(self.flow_rate)
        
        # 底部冷却处理
        bottom_layer_indices = (slice(1, self.grid_size_x+1), slice(1, self.grid_size_y+1), 1)
        
        # 计算热交换
        # heat_exchange = h * self.cell_length**2 * (self.temperature[bottom_layer_indices] - self.inlet_temp) * self.dt
        # self.temperature[bottom_layer_indices] -= heat_exchange / (self.density * self.cell_length**3 * self.specific_heat)
        heat_exchange = h * self.cell_length**2 * (self.temperature[bottom_layer_indices] - effective_inlet_temp) * self.dt
        self.temperature[bottom_layer_indices] -= heat_exchange / (self.density * self.cell_length**3 * self.specific_heat)

        # 更新累积温度（假设每个单元升温0.5%）
        self.coolant_temp_accumulator = effective_inlet_temp + 0.005 * (
            np.mean(self.temperature[bottom_layer_indices]) - effective_inlet_temp
        )

    def calculate_convective_coefficient(self, flow_rate):
        # 基于流速和冷却液参数计算对流换热系数
        # 简化的关系: h = base_coefficient + coefficient * flow_rate
        # base_coefficient = 200  # 基础换热系数
        # coefficient = 300       # 流速影响系数
        
        # return base_coefficient + coefficient * flow_rate

        # 改进对流换热系数
        """基于Dittus-Boelter公式计算h值"""
        length = 0.08
        width = 0.04
        dh = 4 * length * width / (2 * (length + width))  # 水力直径(m)，假设流道为矩形
        
        # 雷诺数
        Re = (flow_rate * dh) / self.coolant_kinematic_viscosity
        
        # 普朗特数
        Pr = (self.coolant_viscosity * self.coolant_specific_heat) / self.coolant_conductivity
        
        # 努塞尔数 (湍流条件: Re > 4000)
        Nu = 0.023 * (Re ** 0.8) * (Pr ** 0.4) if Re > 4000 else 4.36  # 层流值
        
        return Nu * self.coolant_conductivity / dh

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

    def get_voltage(self):
        """计算电池当前电压"""
        # 考虑内阻压降
        voltage_drop = self.current * self.internal_resistance
        current_voltage = self.voltage - voltage_drop
        
        # 确保电压在允许范围内
        current_voltage = max(self.min_voltage, min(self.max_voltage, current_voltage))
        
        return current_voltage

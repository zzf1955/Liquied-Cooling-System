import numpy as np

class SingleBattery:
    def __init__(self, side_length=0.1, cell_length=0.01, density=2700, specific_heat=900, thermal_conductivity=237, resistance=1, env_temperature=298.0, flow_rate=0.1, inlet_temp=300):
        self.side_length = side_length  # 立方体边长 (m)
        self.cell_length = cell_length  # 单元格长度 (m)
        self.grid_size = int(self.side_length / self.cell_length)  # 根据边长和单元格长度计算网格数量
        self.density = density  # 密度 (kg/m^3)
        self.specific_heat = specific_heat  # 比热容 (J/kg·K)
        self.thermal_conductivity = thermal_conductivity  # 热导率 (W/m·K)
        self.resistance = resistance  # 内阻 (Ω)
        self.flow_rate = flow_rate      # 冷却液流速 (m/s)
        self.inlet_temp = inlet_temp    # 冷却液进入温度 (K)

        # 初始化温度网格
        self.temperature = np.full((self.grid_size + 2, self.grid_size + 2, self.grid_size + 2), env_temperature)

        # 热扩散率和时间步长
        self.alpha = self.thermal_conductivity / (self.density * self.specific_heat)
        self.dt = self.cell_length**2 / (6 * self.alpha)  # 稳定性条件

        # 对流换热系数 (可通过实验或经验公式确定)
        self.convective_heat_transfer_coefficient = 500  # 假设值 (W/m^2·K)
        self.top_layer_indices = (slice(1, self.grid_size+1), slice(1, self.grid_size+1), -2)
        self.current = 0
        self.voltage = 5

    def update_heat_generation(self):
        center_index = self.grid_size // 2 + 1
        heat_generation = self.current**2 * self.resistance
        self.temperature[center_index, center_index, center_index] += heat_generation / (self.density * self.specific_heat * self.cell_length**3) * self.dt

    def run(self, t_seconds):
        num_steps = int(t_seconds / self.dt)
        for _ in range(num_steps):
            self.update_heat_generation()
            self.temperature = self.update_temperature_distribution()
            self.apply_cooling()

    def update_temperature_distribution(self):
        new_temp = np.copy(self.temperature)
        for i in range(1, self.grid_size + 1):
            for j in range(1, self.grid_size + 1):
                for k in range(1, self.grid_size + 1):
                    new_temp[i, j, k] = self.temperature[i, j, k] + self.alpha * self.dt / self.cell_length**2 * (
                        self.temperature[i+1, j, k] + self.temperature[i-1, j, k] +
                        self.temperature[i, j+1, k] + self.temperature[i, j-1, k] +
                        self.temperature[i, j, k+1] + self.temperature[i, j, k-1] -
                        6 * self.temperature[i, j, k])
        # 设置边界单元温度与相邻内部单元相同
        # 沿 x 轴的边界
        new_temp[0, :, :] = new_temp[1, :, :]
        new_temp[self.grid_size + 1, :, :] = new_temp[self.grid_size, :, :]
        # 沿 y 轴的边界
        new_temp[:, 0, :] = new_temp[:, 1, :]
        new_temp[:, self.grid_size + 1, :] = new_temp[:, self.grid_size, :]
        # 沿 z 轴的边界
        new_temp[:, :, 0] = new_temp[:, :, 1]
        new_temp[:, :, self.grid_size + 1] = new_temp[:, :, self.grid_size]

        return new_temp

    def apply_cooling(self):
        # 对流换热系数可能与流速有关
        h = self.calculate_convective_coefficient(self.flow_rate)  # 新方法来计算h
        # 仅对顶面进行冷却处理
        top_layer_indices = (slice(1, self.grid_size+1), slice(1, self.grid_size+1), -2)
        heat_exchange = h * self.cell_length**2 * (self.temperature[top_layer_indices] - self.inlet_temp) * self.dt
        self.temperature[top_layer_indices] -= heat_exchange / (self.density * self.cell_length**3 * self.specific_heat)
        self.temperature[top_layer_indices]

    def calculate_convective_coefficient(self, flow_rate):
        # 通过某种方法，如基于流速的经验公式计算对流换热系数
        return 500 + 500 * flow_rate  # 这只是一个示例，具体应根据实际情况进行调整

    def set_action(self,tmp,flow_rate):
        self.inlet_temp = tmp
        self.flow_rate = flow_rate

    def get_action(self):
        return self.inlet_temp,self.flow_rate

    def get_surface_temperatures(self):
        return (self.temperature[1:-1, 1:-1, -2],  # Top
                self.temperature[1:-1, 1:-1, 1],    # Bottom
                self.temperature[1:-1, 1, 1:-1],    # Left
                self.temperature[1:-1, -2, 1:-1],   # Right
                self.temperature[1, 1:-1, 1:-1],    # Front
                self.temperature[-2, 1:-1, 1:-1])   # Back

    def get_core_temperature(self):
        # 计算核心位置的索引
        core_index = self.grid_size // 2 + 1
        # 获取核心温度
        core_temperature = self.temperature[core_index, core_index, core_index]
        return core_temperature

    def get_top_surface_average_temperature(self):
        # 获取顶面温度分布
        top_surface_temperatures = self.get_surface_temperatures()[0]
        # 计算平均温度
        average_temperature = np.mean(top_surface_temperatures)
        return average_temperature


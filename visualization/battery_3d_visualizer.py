"""
电池3D可视化模块

使用Plotly生成交互式3D温度分布图，支持：
- 单电池3D温度分布可视化
- 时间序列动画
- 多电池热力图
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple
import os


class BatteryVisualizer:
    """电池3D可视化类"""

    def __init__(self, battery_system):
        """
        初始化可视化器

        参数:
            battery_system: MutiBattery 实例
        """
        self.battery_system = battery_system
        # 颜色映射范围（温度单位：K）
        self.temp_min = 280  # 最低温度 (K)
        self.temp_max = 320  # 最高温度 (K)

    def _get_temperature_color_scale(self) -> List[Tuple[float, str]]:
        """获取温度颜色映射"""
        return [
            (0.0, '#313695'),   # 深蓝 - 低温
            (0.25, '#4575b4'),  # 蓝色
            (0.5, '#74add1'),   # 浅蓝
            (0.75, '#fdae61'),  # 橙色
            (1.0, '#a50026')    # 深红 - 高温
        ]

    def _get_color_scale_for_temp(self) -> List[str]:
        """获取Plotly颜色刻度"""
        return 'RdYlBu_r'  # 红-黄-蓝（反转）

    def plot_3d_temperature(self,
                           battery_idx: int,
                           save_path: str = "battery_temp_3d.html",
                           title: Optional[str] = None) -> go.Figure:
        """
        绘制单个电池的3D温度分布

        参数:
            battery_idx: 电池索引
            save_path: HTML文件保存路径
            title: 图表标题

        返回:
            Plotly Figure对象
        """
        battery = self.battery_system.batteries[battery_idx]
        temp = battery.temperature

        # 获取网格坐标
        x = np.arange(temp.shape[0]) * battery.cell_length * 1000  # 转换为mm
        y = np.arange(temp.shape[1]) * battery.cell_length * 1000
        z = np.arange(temp.shape[2]) * battery.cell_length * 1000

        # 创建网格
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # 获取温度数据用于颜色映射
        temp_flat = temp.flatten()

        # 创建3D散点图（采样以提高性能）
        # 对数据进行下采样以提高渲染速度
        step = 2
        X_sampled = X[::step, ::step, ::step].flatten()
        Y_sampled = Y[::step, ::step, ::step].flatten()
        Z_sampled = Z[::step, ::step, ::step].flatten()
        temp_sampled = temp[::step, ::step, ::step].flatten()

        # 创建图形
        fig = go.Figure(data=[go.Scatter3d(
            x=X_sampled,
            y=Y_sampled,
            z=Z_sampled,
            mode='markers',
            marker=dict(
                size=3,
                color=temp_sampled,
                colorscale=self._get_color_scale_for_temp(),
                cmin=self.temp_min,
                cmax=self.temp_max,
                colorbar=dict(
                    title=dict(text='Temperature (K)', side='right')
                ),
                opacity=0.8
            ),
            hovertemplate='x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>z: %{z:.1f} mm<br>Temp: %{marker.color:.1f} K<extra></extra>'
        )])

        # 设置布局
        group_id = battery_idx // self.battery_system.num_batteries_per_group
        battery_in_group = battery_idx % self.battery_system.num_batteries_per_group

        fig.update_layout(
            title=title or f'Battery {battery_idx} (Group {group_id}, Cell {battery_in_group}) 3D Temperature Distribution',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # 保存为HTML
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.write_html(save_path)
        print(f"3D温度分布图已保存至: {save_path}")

        return fig

    def plot_3d_cross_section(self,
                             battery_idx: int,
                             plane: str = 'xy',
                             z_slice: Optional[int] = None,
                             save_path: str = "battery_cross_section.html",
                             title: Optional[str] = None) -> go.Figure:
        """
        绘制电池3D切面温度分布

        参数:
            battery_idx: 电池索引
            plane: 切面类型 ('xy', 'xz', 'yz')
            z_slice: 切面位置索引
            save_path: HTML文件保存路径
            title: 图表标题

        返回:
            Plotly Figure对象
        """
        battery = self.battery_system.batteries[battery_idx]
        temp = battery.temperature
        cell_length = battery.cell_length * 1000  # mm

        if plane == 'xy':
            if z_slice is None:
                z_slice = temp.shape[2] // 2
            data = temp[:, :, z_slice]
            x = np.arange(temp.shape[0]) * cell_length
            y = np.arange(temp.shape[1]) * cell_length
            x_label, y_label = 'X (mm)', 'Y (mm)'
        elif plane == 'xz':
            if z_slice is None:
                z_slice = temp.shape[1] // 2
            data = temp[:, z_slice, :]
            x = np.arange(temp.shape[0]) * cell_length
            y = np.arange(temp.shape[2]) * cell_length
            x_label, y_label = 'X (mm)', 'Z (mm)'
        else:  # yz
            if z_slice is None:
                z_slice = temp.shape[0] // 2
            data = temp[z_slice, :, :]
            x = np.arange(temp.shape[1]) * cell_length
            y = np.arange(temp.shape[2]) * cell_length
            x_label, y_label = 'Y (mm)', 'Z (mm)'

        fig = go.Figure(data=go.Heatmap(
            x=x,
            y=y,
            z=data,
            colorscale=self._get_color_scale_for_temp(),
            cmin=self.temp_min,
            cmax=self.temp_max,
            colorbar=dict(title='Temperature (K)')
        ))

        fig.update_layout(
            title=title or f'Battery {battery_idx} {plane.upper()} Cross Section (z={z_slice})',
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=800,
            height=600
        )

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.write_html(save_path)
        print(f"切面温度分布图已保存至: {save_path}")

        return fig

    def plot_all_batteries_heatmap(self,
                                   save_path: str = "all_batteries_heatmap.html",
                                   temperature_type: str = 'core') -> go.Figure:
        """
        绘制所有电池的温度热力图

        参数:
            save_path: HTML文件保存路径
            temperature_type: 温度类型 ('core', 'top', 'bottom')

        返回:
            Plotly Figure对象
        """
        num_groups = self.battery_system.num_groups
        num_per_group = self.battery_system.num_batteries_per_group

        # 收集所有电池的温度
        temps = np.zeros((num_groups, num_per_group))

        for i in range(self.battery_system.total_batteries):
            group = i // num_per_group
            idx = i % num_per_group
            battery = self.battery_system.batteries[i]

            if temperature_type == 'core':
                temps[group, idx] = battery.get_core_temperature()
            elif temperature_type == 'top':
                temps[group, idx] = battery.get_top_surface_average_temperature()
            else:  # bottom
                temps[group, idx] = battery.get_bottom_surface_average_temperature()

        # 创建热力图
        fig = go.Figure()

        for group in range(num_groups):
            fig.add_trace(go.Heatmap(
                z=[temps[group, :]],
                x=[f'Cell {i}' for i in range(num_per_group)],
                y=[f'Group {group}'],
                colorscale=self._get_color_scale_for_temp(),
                zmin=self.temp_min,
                zmax=self.temp_max,
                colorbar=dict(title=dict(text='Temp (K)'), len=0.4, y=0.5 - group * 0.25)
            ))

        fig.update_layout(
            title=f'All Batteries {temperature_type.capitalize()} Temperature Heatmap',
            xaxis_title='Battery Index',
            yaxis_title='Group',
            height=300 + num_groups * 80
        )

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.write_html(save_path)
        print(f"所有电池热力图已保存至: {save_path}")

        return fig

    def plot_temperature_history_animation(self,
                                           battery_indices: Optional[List[int]] = None,
                                           save_path: str = "battery_animation.html") -> go.Figure:
        """
        绘制电池温度随时间变化的动画

        参数:
            battery_indices: 要绘制的电池索引列表
            save_path: HTML文件保存路径

        返回:
            Plotly Figure对象（包含动画帧）
        """
        if battery_indices is None:
            battery_indices = [0, self.battery_system.num_batteries_per_group,
                            2 * self.battery_system.num_batteries_per_group,
                            3 * self.battery_system.num_batteries_per_group]

        time_steps = self.battery_system.time_steps
        if not time_steps:
            print("没有温度历史数据可绘制")
            return None

        # 收集每个电池的温度历史
        temp_histories = []
        for idx in battery_indices:
            if idx < len(self.battery_system.temperature_history):
                temp_histories.append(self.battery_system.temperature_history[idx])
            else:
                temp_histories.append([])

        # 创建帧
        frames = []
        # 每10步取一帧以减少文件大小
        step = max(1, len(time_steps) // 50)

        for t in range(0, len(time_steps), step):
            frame_data = []
            for i, temp_history in enumerate(temp_histories):
                if t < len(temp_history):
                    frame_data.append(go.Scatter(
                        x=[time_steps[t]],
                        y=[temp_history[t]],
                        mode='markers+lines',
                        name=f'Battery {battery_indices[i]}',
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))

            frames.append(go.Frame(
                data=frame_data,
                name=str(t),
                traces=list(range(len(battery_indices)))
            ))

        # 创建初始图形
        fig = go.Figure(
            data=[go.Scatter(
                x=time_steps[:min(10, len(time_steps))],
                y=temp_histories[i][:min(10, len(temp_histories[i]))],
                mode='markers+lines',
                name=f'Battery {battery_indices[i]}',
                line=dict(width=2),
                marker=dict(size=8)
            ) for i in range(len(battery_indices))],
            frames=frames
        )

        # 添加滑块和播放按钮
        fig.update_layout(
            title='Battery Temperature History Animation',
            xaxis_title='Time (s)',
            yaxis_title='Temperature (K)',
            yaxis_range=[self.temp_min - 5, self.temp_max + 5],
            width=900,
            height=600,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': -0.1,
                'x': 0.1,
                'xanchor': 'right',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 15},
                    'prefix': 'Time step: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 50},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [{
                    'method': 'animate',
                    'args': [[str(t)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f'{time_steps[t]:.1f}s'
                } for t in range(0, len(time_steps), step)]
            }]
        )

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.write_html(save_path)
        print(f"温度历史动画已保存至: {save_path}")

        return fig

    def plot_cooling_analysis(self,
                             battery_idx: int = 0,
                             save_path: str = "cooling_analysis.html") -> go.Figure:
        """
        绘制冷却分析图：核心、底部、顶部温度随时间变化

        参数:
            battery_idx: 电池索引
            save_path: HTML文件保存路径

        返回:
            Plotly Figure对象
        """
        battery = self.battery_system.batteries[battery_idx]
        time_steps = self.battery_system.time_steps

        if not time_steps:
            print("没有时间数据可绘制")
            return None

        # 从thermal_history获取数据（如果存在）
        core_temps = self.battery_system.temperature_history[battery_idx] if battery_idx < len(self.battery_system.temperature_history) else []
        coolant_temps = self.battery_system.coolant_history[battery_idx] if battery_idx < len(self.battery_system.coolant_history) else []

        # 计算顶部和底部温度（使用最后的值或计算平均值）
        # 这里简化处理，使用实际可用的数据
        fig = go.Figure()

        if core_temps:
            fig.add_trace(go.Scatter(
                x=time_steps[:len(core_temps)],
                y=core_temps,
                mode='lines',
                name='Core Temperature',
                line=dict(color='red', width=2)
            ))

        if coolant_temps:
            fig.add_trace(go.Scatter(
                x=time_steps[:len(coolant_temps)],
                y=coolant_temps,
                mode='lines',
                name='Coolant Temperature',
                line=dict(color='blue', width=2)
            ))

        # 添加温度边界线
        fig.add_hline(y=self.temp_min, line_dash="dash", line_color="green", annotation_text="Min Safe Temp")
        fig.add_hline(y=self.temp_max, line_dash="dash", line_color="orange", annotation_text="Max Safe Temp")

        fig.update_layout(
            title=f'Battery {battery_idx} Cooling Analysis',
            xaxis_title='Time (s)',
            yaxis_title='Temperature (K)',
            width=900,
            height=500,
            hovermode='x unified'
        )

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.write_html(save_path)
        print(f"冷却分析图已保存至: {save_path}")

        return fig


def create_quick_visualization(battery_system,
                              output_dir: str = "./visualization_output"):
    """
    快速生成所有可视化

    参数:
        battery_system: MutiBattery 实例
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    visualizer = BatteryVisualizer(battery_system)

    # 绘制第一组第一个电池的3D温度分布
    visualizer.plot_3d_temperature(
        battery_idx=0,
        save_path=os.path.join(output_dir, "battery_0_3d_temp.html")
    )

    # 绘制所有电池的热力图
    visualizer.plot_all_batteries_heatmap(
        save_path=os.path.join(output_dir, "all_batteries_heatmap.html")
    )

    # 如果有历史数据，绘制动画
    if battery_system.time_steps:
        visualizer.plot_temperature_history_animation(
            save_path=os.path.join(output_dir, "battery_animation.html")
        )

        visualizer.plot_cooling_analysis(
            save_path=os.path.join(output_dir, "cooling_analysis.html")
        )

    print(f"\n所有可视化已保存至: {output_dir}/")


if __name__ == "__main__":
    # 测试可视化功能
    from BatteryEnv.muti_battery_module import MutiBattery

    # 创建测试电池系统
    battery_system = MutiBattery(num_batteries_per_group=13, num_groups=4)

    # 运行一些模拟步骤
    for step in range(100):
        for battery in battery_system.batteries:
            battery.current = 30.0
            battery.set_action(293, 4)

        battery_system.run(t_seconds=1)

    # 生成可视化
    create_quick_visualization(battery_system, output_dir="./visualization_output")

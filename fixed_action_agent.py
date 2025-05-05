# agent.py
import numpy as np
import pandas as pd
import os # 导入 os 模块用于检查文件是否存在

class CsvReplayAgent:
    """
    一个从 CSV 文件读取预定义动作序列并按顺序回放的 Agent。
    """
    def __init__(self, csv_filepath: str, action_space_dim: int, num_groups: int):
        """
        初始化 CsvReplayAgent。

        Args:
            csv_filepath (str): 包含动作数据的 CSV 文件路径。
                                文件应包含至少 action_space_dim 列，
                                每一行代表一个时间步的完整动作向量。
            action_space_dim (int): 要从 CSV 读取的列数（即动作向量的维度）。
                                    对于 MutiBatteryEnv，这通常是 num_groups * 2。
            num_groups (int): 环境中的电池组数量。用于将读取的动作向量
                              重塑为 (num_groups, 2) 的形状。

        Raises:
            FileNotFoundError: 如果指定的 CSV 文件不存在。
            ValueError: 如果 action_space_dim 与 num_groups * 2 不匹配，
                        或者 CSV 文件中的列数不足。
            Exception: 读取或处理 CSV 文件时发生其他错误。
        """
        self.csv_filepath = csv_filepath
        self.action_space_dim = action_space_dim
        self.num_groups = num_groups
        self._actions = None
        self._current_step = 0
        self._total_steps = 0

        if action_space_dim != num_groups * 2:
            raise ValueError(f"action_space_dim ({action_space_dim}) 必须等于 "
                             f"num_groups * 2 ({num_groups * 2})")

        self._load_actions()

    def _load_actions(self):
        """从 CSV 文件加载动作数据。"""
        if not os.path.exists(self.csv_filepath):
            raise FileNotFoundError(f"错误：找不到 CSV 文件 '{self.csv_filepath}'")

        try:
            # 读取 CSV 文件，只选择前 action_space_dim 列
            # 假设 CSV 没有表头，如果你的 CSV 有表头，设置 header=0
            df = pd.read_csv(self.csv_filepath, header=None, usecols=range(self.action_space_dim))

            # 将数据转换为 NumPy 数组，确保类型为 float32
            self._actions = df.values.astype(np.float32)
            self._total_steps = len(self._actions)

            if self._total_steps == 0:
                print(f"警告：CSV 文件 '{self.csv_filepath}' 为空或不包含有效数据。")
                self._actions = np.empty((0, self.action_space_dim), dtype=np.float32) # 创建空数组以防后续出错

            print(f"成功从 '{self.csv_filepath}' 加载了 {self._total_steps} 个动作步骤。")

        except pd.errors.EmptyDataError:
             print(f"警告：CSV 文件 '{self.csv_filepath}' 为空。")
             self._actions = np.empty((0, self.action_space_dim), dtype=np.float32)
             self._total_steps = 0
        except ValueError as e:
             # 检查是否是因为列数不足引起的错误
             if "Usecols do not match columns" in str(e) or "Index out of bounds" in str(e):
                 raise ValueError(f"错误：CSV 文件 '{self.csv_filepath}' 的列数少于指定的 "
                                  f"action_space_dim ({self.action_space_dim})。") from e
             else:
                 raise Exception(f"读取 CSV 文件 '{self.csv_filepath}' 时发生值错误: {e}") from e
        except Exception as e:
            raise Exception(f"加载和处理 CSV 文件 '{self.csv_filepath}' 时出错: {e}") from e

    def act(self, state):
        """
        返回预定义动作序列中的下一个动作。
        如果到达序列末尾，则从头开始循环。

        Args:
            state: 当前环境状态（此 Agent 不使用状态）。

        Returns:
            np.ndarray: 当前步骤的动作，形状为 (num_groups, 2)，类型为 float32。
                        如果未加载任何动作，则返回 None 或引发错误。
        """
        if self._total_steps == 0:
            print("错误：没有可用的动作。请检查 CSV 文件。")
            # 或者返回一个默认动作，例如全零
            # return np.zeros((self.num_groups, 2), dtype=np.float32)
            raise RuntimeError("Agent 中没有加载任何动作。") # 抛出异常可能更安全

        # 获取当前步骤的动作（扁平数组）
        action_flat = self._actions[self._current_step]

        # 将动作重塑为 (num_groups, 2)
        action_reshaped = action_flat.reshape((self.num_groups, 2))

        # 更新下一步的索引，如果到达末尾则循环回 0
        self._current_step = (self._current_step + 1) % self._total_steps

        return action_reshaped

    def reset(self):
        """
        重置 Agent 状态，将动作序列指针移回开头。
        """
        self._current_step = 0
        # print("CsvReplayAgent 已重置，下一次 'act' 将从第一个动作开始。")

# --- 如何使用示例 (需要配合 evaluate.py 或类似脚本) ---
if __name__ == "__main__":
    # 1. 创建一个示例 CSV 文件 (如果需要)
    csv_file = 'sample_actions.csv'
    num_sample_steps = 10
    num_groups_example = 4
    action_dim_example = num_groups_example * 2 # 8 列

    # 生成一些随机动作数据 [-1, 1]
    sample_data = np.random.uniform(low=-1.0, high=1.0, size=(num_sample_steps, action_dim_example)).astype(np.float32)
    try:
        pd.DataFrame(sample_data).to_csv(csv_file, header=False, index=False)
        print(f"创建了示例 CSV 文件: {csv_file}")

        # 2. 定义环境配置 (仅为 Agent 初始化所需)
        env_config_example = {
            "num_groups": num_groups_example
            # 其他环境参数...
        }

        # 3. 创建 Agent 实例
        try:
            agent = CsvReplayAgent(csv_filepath=csv_file,
                                   action_space_dim=action_dim_example,
                                   num_groups=env_config_example["num_groups"])

            # 4. 模拟调用 act 和 reset
            print("\n模拟调用 agent.act():")
            for i in range(num_sample_steps + 2): # 多调用几次以测试循环
                # 模拟一个状态 (Agent 不使用它)
                dummy_state = np.zeros(10) # 状态维度无关紧要
                action = agent.act(dummy_state)
                print(f"Step {i+1}: Action (shape: {action.shape}, dtype: {action.dtype}):\n{action}")

            print("\n调用 agent.reset()")
            agent.reset()
            print("再次调用 agent.act() 应从头开始:")
            action = agent.act(dummy_state)
            print(f"Step 1 (after reset): Action (shape: {action.shape}, dtype: {action.dtype}):\n{action}")

        except (FileNotFoundError, ValueError, Exception) as e:
            print(f"\n创建或使用 Agent 时出错: {e}")

    finally:
        # 清理示例文件
        if os.path.exists(csv_file):
            os.remove(csv_file)
            # print(f"删除了示例 CSV 文件: {csv_file}")


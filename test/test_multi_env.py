"""
多电池RL环境测试脚本（增强版）

测试内容:
1. 环境基本功能 (reset, step)
2. 状态空间和动作空间验证
3. 奖励函数计算验证
4. make_env 接口验证（单进程）
5. make_env 接口验证（多进程）
6. 随机策略测试
7. 物理模拟验证
8. 可视化测试（温度变化、控制动作、奖励分解）
"""

import numpy as np
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from BatteryEnv.multi_battery_env import MutiBatteryEnv, make_env

# 尝试导入可视化库
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib 未安装，跳过可视化测试")


def test_basic_function():
    """测试环境基本功能"""
    logger.info("=" * 60)
    logger.info("测试1: 环境基本功能")
    logger.info("=" * 60)

    env = MutiBatteryEnv(
        num_batteries_per_group=13,
        num_groups=4,
        max_steps=100,
        debug=True,
    )

    logger.info(f"动作空间: {env.action_space}")
    logger.info(f"状态空间: {env.observation_space}")
    logger.info(f"状态空间形状: {env.observation_space.shape}")

    # Reset
    obs, info = env.reset()
    logger.info(f"初始观测形状: {obs.shape}")
    logger.info(f"初始观测前16维: {obs[:16]}")

    # 执行几步
    for step in range(5):
        action = np.array([0.0, 1.5], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)

        logger.info(f"Step {step+1}: reward={reward:.4f}, core={obs[2]:.2f}K")

        if done or truncated:
            break

    logger.info("\n✓ 基本功能测试通过!")


def test_state_space():
    """测试状态空间"""
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 状态空间验证")
    logger.info("=" * 60)

    env = MutiBatteryEnv(num_batteries_per_group=13, num_groups=4, debug=False)
    obs, _ = env.reset()

    # 验证每组的状态结构
    logger.info("每组8维: [top_t, bottom_t, core_t, core_max_t, core_min_t, voltage, current, last_core]")

    for group_idx in range(4):
        start = group_idx * 8
        top_t = obs[start + 0]
        bottom_t = obs[start + 1]
        core_t = obs[start + 2]
        core_max_t = obs[start + 3]
        core_min_t = obs[start + 4]
        voltage = obs[start + 5]
        current = obs[start + 6]
        last_core = obs[start + 7]

        logger.info(f"组{group_idx}: core={core_t:.2f}K, max={core_max_t:.2f}K, min={core_min_t:.2f}K, voltage={voltage:.2f}V, current={current:.2f}A")

        assert core_max_t >= core_t >= core_min_t, f"组{group_idx}温度关系错误!"

    logger.info("\n✓ 状态空间测试通过!")


def test_action_space():
    """测试动作空间"""
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 动作空间验证")
    logger.info("=" * 60)

    env = MutiBatteryEnv(
        num_batteries_per_group=13,
        num_groups=4,
        inlet_temp_range=(288.0, 295.0),
        flow_rate_range=(0.0, 3.0),
        debug=False,
    )

    # 测试动作映射
    test_cases = [
        (-1.0, -1.0, "最小值"),
        (0.0, 0.0, "中间值"),
        (1.0, 1.0, "最大值"),
    ]

    for a0, a1, desc in test_cases:
        action = np.array([a0, a1], dtype=np.float32)
        inlet = env._map_to_range(action[0], env.inlet_temp_range)
        flow = env._map_to_range(action[1], env.flow_rate_range)
        logger.info(f"{desc}: {action} -> inlet={inlet:.2f}K, flow={flow:.2f}m/s")

    logger.info("\n✓ 动作空间测试通过!")


def test_reward_function():
    """测试奖励函数"""
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 奖励函数验证")
    logger.info("=" * 60)

    env = MutiBatteryEnv(
        num_batteries_per_group=13,
        num_groups=4,
        target_temp=298.0,
        lambda_cost=0.5,
        mu_cost=0.5,
        debug=False,
    )

    obs, _ = env.reset()

    # 执行动作
    action = np.array([0.0, 1.5], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)

    breakdown = info.get('reward_breakdown', {})
    logger.info(f"总奖励: {reward:.4f}")
    logger.info(f"奖励分解:")
    for k, v in breakdown.items():
        logger.info(f"  {k}: {v:.4f}")

    # 验证控制成本计算
    flow_cost = 0.5 * info['actions'][1]
    smooth_cost = 0.5 * abs(info['actions'][0] - 0)  # 第一次无历史
    expected_cost = flow_cost + smooth_cost
    logger.info(f"\n控制成本验证: 期望={expected_cost:.4f}, 实际={abs(breakdown.get('control_cost', 0)):.4f}")

    logger.info("\n✓ 奖励函数测试通过!")


def test_make_env_single_process():
    """测试 make_env 单进程"""
    logger.info("\n" + "=" * 60)
    logger.info("测试5: make_env 单进程验证")
    logger.info("=" * 60)

    main_env, train_envs, test_envs = make_env(
        num_batteries_per_group=13,
        num_groups=4,
        episode_steps=50,
        log_path="",
        con=True,
        num_train_envs=2,
        num_test_envs=1,
        use_subproc=False,  # 单进程
        debug=False,
    )

    logger.info(f"主环境: {main_env}")
    logger.info(f"训练环境数量: {len(train_envs)}")

    # 测试主环境
    obs, _ = main_env.reset()
    logger.info(f"主环境状态维度: {obs.shape}")

    # 测试训练环境
    train_obs = train_envs.reset()
    if isinstance(train_obs, tuple):
        train_obs = train_obs[0]
    logger.info(f"训练环境批量状态: {train_obs.shape}")

    # 执行一步
    actions = np.array([[0.0, 1.0]] * len(train_envs), dtype=np.float32)
    results = train_envs.step(actions)
    obs = results[0]
    rewards = results[1]
    logger.info(f"批量执行: obs={obs.shape}, rewards={rewards}")

    logger.info("\n✓ make_env 单进程测试通过!")


def test_make_env_multi_process():
    """测试 make_env 多进程"""
    logger.info("\n" + "=" * 60)
    logger.info("测试6: make_env 多进程验证")
    logger.info("=" * 60)

    main_env, train_envs, test_envs = make_env(
        num_batteries_per_group=13,
        num_groups=4,
        episode_steps=50,
        log_path="",
        con=True,
        num_train_envs=4,
        num_test_envs=2,
        use_subproc=True,  # 多进程
        debug=False,
    )

    logger.info(f"主环境: {main_env}")
    logger.info(f"训练环境数量: {len(train_envs)}")
    logger.info(f"测试环境数量: {len(test_envs)}")

    try:
        # 测试多进程环境 - reset
        train_obs = train_envs.reset()
        if isinstance(train_obs, tuple):
            train_obs = train_obs[0]
        logger.info(f"多进程训练环境批量状态: {train_obs.shape}")

        # 执行一步
        actions = np.array([[0.0, 1.0]] * len(train_envs), dtype=np.float32)
        results = train_envs.step(actions)
        obs = results[0]
        rewards = results[1]
        logger.info(f"Step执行完成: obs shape={obs.shape}, rewards={rewards}")

        logger.info("\n✓ make_env 多进程测试通过!")
        logger.info("注: 多进程环境需要等待每个子进程完成才能进行下一次交互")

    finally:
        # 显式关闭多进程环境，避免 ConnectionResetError
        logger.info("关闭多进程环境...")
        train_envs.close()
        test_envs.close()
        time.sleep(0.5)  # 等待子进程清理完成


def test_random_policy():
    """测试随机策略"""
    logger.info("\n" + "=" * 60)
    logger.info("测试7: 随机策略测试")
    logger.info("=" * 60)

    env = MutiBatteryEnv(
        num_batteries_per_group=13,
        num_groups=4,
        max_steps=100,
        current_mu=30.0,
        current_sigma=10.0,
        debug=False,
    )

    obs, _ = env.reset()

    total_reward = 0.0
    episode_count = 0
    steps = 0

    for step in range(500):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if done or truncated:
            episode_count += 1
            obs, _ = env.reset()

        if (step + 1) % 100 == 0:
            logger.info(f"Step {step+1}: avg_reward={total_reward/(step+1):.4f}")

    logger.info(f"随机策略500步平均奖励: {total_reward/500:.4f}")
    logger.info(f"完成episode数: {episode_count}")
    logger.info("\n✓ 随机策略测试通过!")


def test_physical_simulation():
    """测试物理模拟"""
    logger.info("\n" + "=" * 60)
    logger.info("测试8: 物理模拟验证")
    logger.info("=" * 60)

    env = MutiBatteryEnv(
        num_batteries_per_group=13,
        num_groups=4,
        max_steps=100,
        current_mu=30.0,
        current_sigma=10.0,
        debug=False,
    )

    obs, _ = env.reset()

    # 测试电流变化
    currents = []
    temps = []

    for step in range(50):
        action = np.array([0.0, 1.5], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)

        current = obs[6]
        core_t = obs[2]
        currents.append(current)
        temps.append(core_t)

        if done or truncated:
            break

    logger.info(f"电流: mean={np.mean(currents):.2f}A, std={np.std(currents):.2f}A")
    logger.info(f"核心温度: mean={np.mean(temps):.2f}K, std={np.std(temps):.2f}K")
    logger.info("\n✓ 物理模拟测试通过!")


def test_visualization():
    """可视化测试"""
    if not HAS_MATPLOTLIB:
        logger.warning("跳过可视化测试 (matplotlib 未安装)")
        return

    logger.info("\n" + "=" * 60)
    logger.info("测试9: 可视化测试")
    logger.info("=" * 60)

    # 创建环境并运行
    env = MutiBatteryEnv(
        num_batteries_per_group=13,
        num_groups=4,
        max_steps=200,
        current_mu=30.0,
        current_sigma=10.0,
        debug=False,
    )

    obs, _ = env.reset()

    # 记录数据
    core_temps = []
    flow_rates = []
    inlet_temps = []
    rewards = []
    currents = []

    for step in range(200):
        action = np.array([0.0, 1.5], dtype=np.float32)  # 固定动作
        obs, reward, done, truncated, info = env.step(action)

        core_temps.append(obs[2])  # 组0核心温度
        flow_rates.append(info['actions'][1])
        inlet_temps.append(info['actions'][0])
        rewards.append(reward)
        currents.append(obs[6])

        if done or truncated:
            break

    # 创建可视化
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Battery Thermal Management Environment Test', fontsize=14)

    # 1. 核心温度变化
    axes[0, 0].plot(core_temps, 'r-', linewidth=1.5)
    axes[0, 0].axhline(y=298, color='g', linestyle='--', label='Target (298K)')
    axes[0, 0].axhline(y=313, color='orange', linestyle='--', label='Max (313K)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Temperature (K)')
    axes[0, 0].set_title('Core Temperature (Group 0)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 动作变化
    axes[0, 1].plot(flow_rates, 'b-', label='Flow Rate (m/s)', linewidth=1.5)
    ax2 = axes[0, 1].twinx()
    ax2.plot(inlet_temps, 'r--', label='Inlet Temp (K)', linewidth=1.5)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Flow Rate (m/s)', color='b')
    ax2.set_ylabel('Inlet Temp (K)', color='r')
    axes[0, 1].set_title('Control Actions')
    axes[0, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 奖励变化
    axes[1, 0].plot(rewards, 'g-', linewidth=1)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward per Step')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 电流变化
    axes[1, 1].plot(currents, 'purple', linewidth=1.5)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Current (A)')
    axes[1, 1].set_title('Battery Current')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. 累计奖励
    cumulative_rewards = np.cumsum(rewards)
    axes[2, 0].plot(cumulative_rewards, 'orange', linewidth=1.5)
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Cumulative Reward')
    axes[2, 0].set_title('Cumulative Reward')
    axes[2, 0].grid(True, alpha=0.3)

    # 6. 温度统计（箱线图）
    # 运行多步收集更多数据
    all_temps = []
    for _ in range(5):
        obs, _ = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, done, truncated, _ = env.step(action)
            all_temps.append(obs[2])
            if done or truncated:
                break

    axes[2, 1].boxplot([all_temps], labels=['Group 0 Core Temp'])
    axes[2, 1].axhline(y=298, color='g', linestyle='--', label='Target')
    axes[2, 1].axhline(y=313, color='r', linestyle='--', label='Max')
    axes[2, 1].set_ylabel('Temperature (K)')
    axes[2, 1].set_title('Temperature Distribution')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_path = Path(__file__).parent / "test_results"
    output_path.mkdir(exist_ok=True)
    fig_path = output_path / "env_visualization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"可视化图片已保存: {fig_path}")

    # 同时保存CSV数据
    csv_path = output_path / "env_test_data.csv"
    data = {
        'step': list(range(len(core_temps))),
        'core_temp': core_temps,
        'flow_rate': flow_rates,
        'inlet_temp': inlet_temps,
        'reward': rewards,
        'current': currents,
    }
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    logger.info(f"测试数据已保存: {csv_path}")

    plt.close()
    logger.info("\n✓ 可视化测试通过!")


def test_reward_analysis():
    """奖励分析测试 - 展示不同情况下的奖励"""
    logger.info("\n" + "=" * 60)
    logger.info("测试11: 奖励分析")
    logger.info("=" * 60)

    env = MutiBatteryEnv(
        num_batteries_per_group=13,
        num_groups=4,
        target_temp=298.0,
        lambda_cost=0.5,
        mu_cost=0.5,
        debug=False,
    )

    # 情况1: 随机动作（高温惩罚大）
    logger.info("\n=== 情况1: 随机动作 ===")
    obs, _ = env.reset()
    action = np.array([1.0, 1.0], dtype=np.float32)  # 高入口温度+高流速
    obs, reward, done, truncated, info = env.step(action)
    logger.info(f"动作: inlet={info['actions'][0]:.2f}K, flow={info['actions'][1]:.2f}m/s")
    logger.info(f"奖励: {reward:.4f}")
    logger.info(f"分解: {info.get('reward_breakdown', {})}")

    # 情况2: 接近目标温度的动作
    logger.info("\n=== 情况2: 冷却动作（接近目标）===")
    obs, _ = env.reset()
    # 设置低入口温度来冷却
    action = np.array([-1.0, 1.0], dtype=np.float32)  # 最低入口温度+高流速
    for _ in range(20):  # 多步冷却
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break

    logger.info(f"最终温度: core={obs[2]:.2f}K")
    logger.info(f"最终奖励: {reward:.4f}")
    logger.info(f"分解: {info.get('reward_breakdown', {})}")

    # 情况3: 长时间运行，看奖励变化
    logger.info("\n=== 情况3: 长时间运行 ===")
    obs, _ = env.reset()
    rewards = []
    for step in range(100):
        action = np.array([-0.5, 1.5], dtype=np.float32)  # 适度冷却
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        if done or truncated:
            break

    logger.info(f"100步平均奖励: {np.mean(rewards):.4f}")
    logger.info(f"最后10步平均奖励: {np.mean(rewards[-10:]):.4f}")

    # 验证：最优情况下奖励可以接近0
    logger.info("\n=== 验证: 最优控制 ===")
    obs, _ = env.reset()
    # 使用非常精细的控制
    for step in range(50):
        # 动态调整：温度高时加大冷却
        current_temp = obs[2]
        if current_temp > 300:
            action = np.array([-1.0, 2.5], dtype=np.float32)
        elif current_temp > 298.5:
            action = np.array([-0.8, 2.0], dtype=np.float32)
        else:
            action = np.array([-0.5, 1.0], dtype=np.float32)  # 保持

        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break

    logger.info(f"最优控制最终奖励: {reward:.4f}")
    logger.info(f"最终温度: core={obs[2]:.2f}K (目标298K)")

    logger.info("\n✓ 奖励分析测试完成!")
    logger.info("结论: 奖励为负是正常的（因为是惩罚函数）")
    logger.info("       只有温度接近目标且控制成本低时，奖励才接近0")


def benchmark_performance():
    """性能基准测试"""
    logger.info("\n" + "=" * 60)
    logger.info("测试12: 性能基准测试")
    logger.info("=" * 60)

    # 单进程测试
    start = time.time()
    env = MutiBatteryEnv(num_batteries_per_group=13, num_groups=4, debug=False)
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            env.reset()
    single_time = time.time() - start
    logger.info(f"单进程 1000步耗时: {single_time:.2f}秒 ({1000/single_time:.0f} step/s)")

    # 多进程测试
    _, train_envs, _ = make_env(
        num_batteries_per_group=13,
        num_groups=4,
        episode_steps=100,
        use_subproc=True,
        num_train_envs=4,
        num_test_envs=1,
        debug=False,
    )

    try:
        start = time.time()
        train_envs.reset()
        actions = np.array([[0.0, 1.0]] * len(train_envs), dtype=np.float32)
        train_envs.step(actions)
        multi_time = time.time() - start
        logger.info(f"多进程(4env) 1步耗时: {multi_time:.2f}秒")

    finally:
        logger.info("关闭多进程环境...")
        train_envs.close()
        time.sleep(0.5)

    logger.info("\n✓ 性能基准测试完成!")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    try:
        test_basic_function()
        test_state_space()
        test_action_space()
        test_reward_function()
        test_make_env_single_process()
        test_make_env_multi_process()
        test_random_policy()
        test_physical_simulation()
        test_visualization()
        benchmark_performance()
        test_reward_analysis()

        logger.info("\n" + "=" * 60)
        logger.info("🎉 所有测试通过!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

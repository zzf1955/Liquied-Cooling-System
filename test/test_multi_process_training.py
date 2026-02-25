"""
多进程训练效率对比测试

对比 DummyVectorEnv 和 SubprocVectorEnv 的训练效率
"""

import numpy as np
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from BatteryEnv.multi_battery_env import make_env


def run_training(use_subproc, num_envs=4, num_steps=200):
    """训练测试"""
    mode = "SubprocVectorEnv" if use_subproc else "DummyVectorEnv"
    logger.info(f"\n{'='*60}")
    logger.info(f"{mode} 测试 (环境数={num_envs})")
    logger.info(f"{'='*60}")

    _, train_envs, _ = make_env(
        num_batteries_per_group=13,
        num_groups=4,
        episode_steps=400,
        use_subproc=use_subproc,
        num_train_envs=num_envs,
        num_test_envs=1,
        debug=False,
    )

    try:
        start_time = time.time()

        # 初始化
        obs = train_envs.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        total_reward = 0
        temp_history = []
        completed_steps = 0
        steps_per_iter = num_envs

        for step in range(num_steps):
            # 生成动作
            actions = []
            for i in range(len(obs)):
                core_t = obs[i, 2]
                if core_t > 305:
                    action = np.array([-1.0, 3.0])
                elif core_t > 300:
                    action = np.array([-0.8, 2.5])
                elif core_t > 298:
                    action = np.array([-0.5, 2.0])
                elif core_t > 295:
                    action = np.array([-0.3, 1.5])
                else:
                    action = np.array([0.0, 1.0])
                actions.append(action)

            actions = np.array(actions, dtype=np.float32)

            # 执行
            try:
                results = train_envs.step(actions)
                obs = results[0]
                rewards = results[1]
            except Exception as e:
                logger.warning(f"步骤 {step} 出错: {e}")
                # 重置
                obs = train_envs.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                continue

            if obs is not None and len(obs) > 0:
                total_reward += np.sum(rewards)
                temp_history.extend(obs[:, 2].tolist())
                completed_steps += len(obs)

        elapsed_time = time.time() - start_time

        avg_reward = total_reward / completed_steps if completed_steps > 0 else 0
        avg_temp = np.mean(temp_history) if temp_history else 0
        speed = completed_steps / elapsed_time

        logger.info(f"完成: 耗时={elapsed_time:.2f}秒, 步数={completed_steps}, 速度={speed:.1f} step/s")
        logger.info(f"平均奖励: {avg_reward:.2f}, 平均温度: {avg_temp:.2f}K")

        return {
            "time": elapsed_time,
            "steps": completed_steps,
            "speed": speed,
            "avg_reward": avg_reward,
            "avg_temp": avg_temp,
        }

    finally:
        train_envs.close()
        time.sleep(0.3)


def test_stability():
    """稳定性测试"""
    logger.info(f"\n{'='*60}")
    logger.info("稳定性测试 - 多次短训练")
    logger.info(f"{'='*60}")

    errors = []

    for i in range(3):
        try:
            logger.info(f"\n--- 第 {i+1} 次训练 ---")
            result = run_training(use_subproc=True, num_envs=4, num_steps=50)
            logger.info(f"成功完成")
        except Exception as e:
            errors.append(str(e))
            logger.error(f"失败: {e}")

    if errors:
        logger.warning(f"出现 {len(errors)} 次错误:")
        for e in errors:
            logger.warning(f"  - {e}")
    else:
        logger.info("✓ 所有训练都成功完成")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>", level="INFO")

    # 测试1: DummyVectorEnv (单进程模拟并行)
    dummy_result = run_training(use_subproc=False, num_envs=4, num_steps=200)

    # 测试2: SubprocVectorEnv (多进程真并行)
    try:
        subproc_result = run_training(use_subproc=True, num_envs=4, num_steps=200)
    except Exception as e:
        logger.error(f"SubprocVectorEnv 测试失败: {e}")
        subproc_result = None

    # 对比
    logger.info(f"\n{'='*60}")
    logger.info("效率对比")
    logger.info(f"{'='*60}")
    logger.info(f"DummyVectorEnv: {dummy_result['speed']:.1f} step/s")
    if subproc_result:
        logger.info(f"SubprocVectorEnv: {subproc_result['speed']:.1f} step/s")
        if subproc_result['speed'] > dummy_result['speed']:
            logger.info(f"多进程加速比: {subproc_result['speed']/dummy_result['speed']:.2f}x")

    # 稳定性测试
    test_stability()

    logger.info("\n✓ 测试完成!")
    logger.info("\n结论:")
    logger.info("1. SubprocVectorEnv 需要额外的进程管理，可能不稳定")
    logger.info("2. 调试时建议使用 DummyVectorEnv")
    logger.info("3. 正式训练时可以用 SubprocVectorEnv，但要注意进程通信问题")

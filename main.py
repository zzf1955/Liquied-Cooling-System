import os
from stable_baselines3 import SAC
from environment.muti_battery_env import MutiBatteryEnv,test_muti_battery_env
from tools.logger import LogToCSVCallback

if __name__ == "__main__":
    # 创建环境
    #while True:
    #    test_muti_battery_env(random_current=True,current_change_prob=0.5)
    env = MutiBatteryEnv(episode_steps=500, num_batteries=1,current_change_prob=0.01,env_temp=300,
                         max_battery_tmp=310,min_battery_tmp=290,tmp_threshould=4,random_current=True)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 256, 256],  # 策略网络
            qf=[512, 512, 512, 512]   # 价值网络
        )
    )

    buffer_size = 100000*500  # 根据需要调整

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    model = SAC('MlpPolicy', env,verbose=1, policy_kwargs=policy_kwargs, buffer_size=buffer_size, tensorboard_log=log_dir)

    callback = LogToCSVCallback(log_dir=log_dir)

    model.learn(total_timesteps=1000000, callback=callback)

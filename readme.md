# 好好好

- tianshou设置动态种子比较麻烦，目前训练/测试环境是用一个种子
- 目前使用csv中读出来的action序列，还没写完
- 要更新的话麻烦从新版本上叉一个出来谢谢
- 训练参数以dsac为准，环境相关参数都写在env的默认参数里了，以前无用的代码我都删了
- 关键改动
  - 仿真步长改为5s
  - state内容没动

battery
├─ BatteryEnv **环境**
│    ├─ __init__.py
│    ├─ muti_battery_env.py
│    ├─ muti_battery_env_con.py
│    ├─ muti_battery_module.py
│    ├─ single_battery_env.py
│    └─ single_battery_module.py
├─ BatteryEnv_
│    ├─ muti_battery_env.py
│    ├─ muti_battery_module.py
│    ├─ single_battery_env.py
│    └─ single_battery_module.py
├─ diffusion **diffusio模块，继承自AGOD**
│    ├─ __init__.py
│    ├─ diffusion.py
│    ├─ helpers.py
│    ├─ model.py
│    └─ utils.py
├─ dsac.py **dsac训练**
├─ evaluate.py **测试自定义policy**
├─ fixed_action_agent.py **从csv读确定的action训练**
├─ log **训练log**
│    └─ 4_2
│           ├─ dsac
│           └─ sac
├─ policy **训练完的policy**
│    └─ dsac.pth
├─ readme.md
├─ sac.py **sac训练**
├─ test_dsac.py **测试dsac代码**
└─ test_log **测试用log**
       └─ dsac
              ├─ log_10.csv
              ├─ log_20.csv
              ├─ log_30.csv
              └─ log_40.csv

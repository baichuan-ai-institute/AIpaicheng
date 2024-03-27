#!/bin/sh

# 编译
cd /app/modules/LKH-3.0.7
make -s

# 切换工作路径
cd /app

# 后台运行收集kafka消息的脚本
nohup python ./bin/save_kafka.py > ./log/save_kafka.log 2>&1 &

# 启动一次daily_routine脚本, 获取当天的预测结果(可选)
# python ./bin/daily_routine.py

# 启动 Cron 任务
cron

# 启动接口
python app.py

# 阻止容器退出
# tail -f /dev/null

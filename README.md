
# AI排程项目

运用时间序列预测模型和运筹优化模型对移动储能业务进行AI自动排程.



## 场景
目前移动储能业务使用移动储能罐向用户供热.

此模式用车头将充满热(饱和水蒸气)的储能罐从热源侧送往用户侧, 当罐放热完毕, 再用车头将空罐从用户侧运回热源侧. 空罐在热源侧重新充热, 充热完毕再由车头送往用户侧, 循环往复. 车头和移动罐可以分离, 当移动罐充热或放热期间, 车头不必原地等待, 可以去执行其他取送任务.
## 目标
(1) 通过对用户的历史用热数据进行分析, 建立`时间序列预测模型`, 预测用户第二天的用热量和用热高峰期.

(2)根据预测结果, 建立`运筹优化最优解模型`, 调度车头和移动罐, 在保障用户正常用热的前提下, 生成最优的计划排程.

(3)根据车头当天实际完成的配送任务, 与计划排程相比较, 核销已完成的任务, 对余下的任务、车头和移动罐再次进行调度, 生成最优的实际排程.
## 作者

- 余承乐


## 部署

第一步: 测试环境(V100)下, cd到项目根目录

第二步: 运行以下命令生成DOCKER镜像. 格式: docker build -t [镜像名:tag] .

```bash
  docker build -t energy-storage/operation-research-pro:2.0 .
```
第三步: 将镜像发到生产环境(算法端)

第四步: 生产环境下(算法端), 运行以下命令, 使用镜像创建DOCKER容器(需要指定端口, CPU, 内存)

```bash
  docker run -d -p 9989:9989 --name es-or-container --cpus=8 --memory=8g energy-storage/operation-research-pro:2.0
```

## 项目结构
见 tree.txt

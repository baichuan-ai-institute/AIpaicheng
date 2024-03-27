import os
import sys

# 添加modules的绝对路径添加到 sys.path
package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from modules.model_utils import *
from modules.or_utils import *


# 日志设置
logger = logging.getLogger('daily_routine')
logger.addHandler(dingtalk_handler)


def supplies_and_demands_routine():
    wrap_log('Supplies & Demands Routine', color='green')

    # 提取原始数据
    collect_raw_data(SUPPLIES_AND_DEMANDS_RAW_DATA)

    # 预处理
    preprocess(SUPPLIES_AND_DEMANDS_RAW_DATA, SUPPLIES_AND_DEMANDS_PREPROCESS)

    # 训练模型
    generate_arima_model(SUPPLIES_AND_DEMANDS_PREPROCESS, SUPPLIES_AND_DEMANDS_MODEL, save=True)

    # 预测
    arima_pred(SUPPLIES_AND_DEMANDS_MODEL, SUPPLIES_AND_DEMANDS_PRED, SUPPLIES_AND_DEMANDS_HORIZON)


def inst_flows_routine():
    wrap_log('Inst Flows Routine', color='green')

    # 提取原始数据
    collect_raw_data(INST_FLOWS_RAW_DATA, inst_flow=True)

    # 预处理
    preprocess(INST_FLOWS_RAW_DATA, INST_FLOWS_PREPROCESS, inst_flow=True)

    # 训练模型
    generate_arima_model(INST_FLOWS_PREPROCESS, INST_FLOWS_MODEL, save=True, inst_flow=True)

    # 预测
    pred_inst_flow(INST_FLOWS_PREPROCESS, INST_FLOWS_PRED, INST_FLOWS_HORIZON)


def main():
    try:
        supplies_and_demands_routine()
        inst_flows_routine()
        wrap_log('Done', color='green')
    except Exception:
        logger.critical("FAILED", exc_info=True)


if __name__ == '__main__':
    main()

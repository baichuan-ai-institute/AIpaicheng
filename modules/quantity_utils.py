"""量模块, 用于计算站点的充放热量."""
import numpy as np

from modules.data_utils import *

# 日志设置
file_handler = logging.FileHandler(MODULES_LOG_FOLDER_PATH + '/quantity_utils.log')
file_handler.setFormatter(formatter)
logger = logging.getLogger('quantity_utils')
logger.setLevel("INFO")
logger.addHandler(file_handler)
logger.addHandler(dingtalk_handler)


def dep_dem_daily_quantity(sn: str, start_time: str, end_time: str) -> float | None:
    """Return either the daily charge quantity of depot or the daily release quantity of demand.

    Args:
        sn: An identifier of a vehicle or depot or demand.
        start_time: The start time as "%Y-%m-%d %H:%M:%S".
        end_time: The end time as "%Y-%m-%d %H:%M:%S".

    Returns:
        The daily charge quantity or daily release quantity in tons. Return None if instant flow data is unavailable.

    Examples:
        dep_dem_daily_quantity('200123081349', '2023-08-05 00:00:00', '2023-08-05 23:59:59')
        Returns: 14.627

    Raises:
        AssertionError: Raise when start_time is later than end_time.
    """
    try:
        # 开始时间必须早于结束时间
        assert pd.Timestamp(start_time) < pd.Timestamp(end_time), 'The start_time should be earlier than the end_time.'

        inst_flow_df = get_inst_flow_df(sn, start_time, end_time)
        if inst_flow_df.empty:
            return None

        inst_flow_ser = inst_flow_df['value'].astype(float)

        # 去除异常值
        inst_flow_ser = inst_flow_ser.where(inst_flow_ser <= 50).dropna()
        if inst_flow_ser.shape[0] < 2:
            return None

        # 持续时间[数组]
        duration_df = (inst_flow_ser.index - inst_flow_ser.index[0]).seconds.to_frame(index=False).diff().dropna()
        duration_array = duration_df.iloc[:, 0].to_numpy() / 3600

        # 平均瞬时流量[数组]
        avg_flow_ser = ((inst_flow_ser + inst_flow_ser.shift(-1)) / 2).dropna()
        avg_flow_array = avg_flow_ser.to_numpy()

        # 充热站充热量或放热站放热量 = sum(单位时间 * 平均瞬时流量)
        result = np.round(np.sum(avg_flow_array * duration_array), 3)
        return result
    except Exception as e:
        logger.critical(f'Failed to run dep_dem_daily_quantity().\nTraceback: {e}')


def dep_dem_hourly_flow(sn: str, start_time: str, end_time: str) -> pd.DataFrame:
    """Return either the hourly charge flow of depot or the hourly release fow of demand.

    Args:
        sn: An identifier of a vehicle or depot or demand.
        start_time: The start time as "%Y-%m-%d %H:%M:%S".
        end_time: The end time as "%Y-%m-%d %H:%M:%S".

    Returns:
        The hourly charge flow or hourly release flow in tons per hour.

    Raises:
        AssertionError: Raise when start_time is later than end_time.
    """
    try:
        # 开始时间必须早于结束时间
        assert pd.Timestamp(start_time) < pd.Timestamp(end_time), 'The start_time should be earlier than the end_time.'

        df = get_inst_flow_df(sn, start_time, end_time)
        if df.empty:
            return df

        # 转换值类型
        df = df.astype(float).round(6)

        # 去除异常值
        df[(df < 0) | (df > 20)] = None

        # 降采样
        df = df.resample('H').mean().round(6)

        if df.shape[0] == 24:
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.critical(f'Failed to run dep_dem_hourly_flow().\nTraceback: {e}')

"""预测模块

--- 日供热量预测
--- 日用热量预测
--- 日高峰期预测
"""

from modules.model_utils import *


def daily_supply(depot_id: str, date: str) -> dict:
    """Return daily supply of a specific SN and date.

    Args:
        depot_id: The sn of depot.
        date: Date string as "%Y-%m-%d".

    Returns:
        Daily supply.

    Examples:
        daily_supply("200123081370","2024-03-15")

        Returns: {
            'depot_id': '200123011286',
            'date': '2024-03-15',
            'pred_daily_supply': 5.38,
            'pred_hourly_flow': 0.22,
            }
    """
    # 预测每日供热量
    pred_daily_supply = get_supplies_and_demands_pred(depot_id, date)
    # 预测每小时流量
    pred_hourly_flow = abs(round(pred_daily_supply / 24, 2))
    result = {
        "depot_id": depot_id,
        "date": date,
        "pred_daily_supply": pred_daily_supply,
        "pred_hourly_flow": pred_hourly_flow
    }
    return result


def daily_demand(demand_id: str, date: str) -> dict:
    """Return daily demand of a specific SN and date.

    Args:
        demand_id: The sn of demand.
        date: Date string as "%Y-%m-%d".

    Returns:
        Daily demand.

    Examples:
        daily_demand("200123051356","2024-03-15")

        Returns: {
            'demand_id': '200123051356',
            'date': '2024-03-15',
            'pred_daily_demand': 11.52,
            'pred_hourly_flow': 0.48,
            'pred_peak_time': '16:00:00-17:00:00',
            'pred_peak_hourly_flow': 1.654,
            }
    """
    # 预测每日用热量
    pred_daily_demand = get_supplies_and_demands_pred(demand_id, date)
    # 预测每小时流量
    pred_hourly_flow = abs(round(pred_daily_demand / 24, 2))
    # 预测用热流量
    pred_inst_flow_df = get_inst_flows_pred(demand_id, date)
    # 用热高峰, 高峰期流量
    pred_peak_time, pred_peak_hourly_flow = extract_peak(pred_inst_flow_df)
    result = {
        "demand_id": demand_id,
        "date": date,
        "pred_daily_demand": pred_daily_demand,
        "pred_hourly_flow": pred_hourly_flow,
        "pred_peak_time": pred_peak_time,
        "pred_peak_hourly_flow": pred_peak_hourly_flow,
    }
    return result


def extract_peak(df) -> tuple[str|None, float]:
    """Extrac peak time and peak hourly flow from a time series DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        Peak time and peak hourly flow. Returns (None, 0) if df is empty.
    """
    df = df[df > 0]
    df = df.dropna()
    if df.empty:
        return None, 0
    peak_ts = df.idxmax().iloc[0]
    start_str = peak_ts.strftime('%H:%M:%S')
    if peak_ts.hour == 23:
        end_str = '24:00:00'
    else:
        end_str = (peak_ts + pd.Timedelta(hours=1)).strftime('%H:%M:%S')
    peak_time = start_str + '-' + end_str
    peak_hourly_flow = df['value'].max()
    return peak_time, peak_hourly_flow


def get_supplies_and_demands_pred(sn: str, date: str) -> float | int:
    """Return predictive supply or demand of a sn and date.

    Args:
        sn: An identifier of a vehicle or depot or demand.
        date: Date string like "%Y-%m-%d".

    Returns:
        The predictive result. Returns 0 if KeyError or FileNotFoundError happens.
    """
    csv_file = SUPPLIES_AND_DEMANDS_PRED + '/' + sn + '.csv'
    try:
        df = pd.read_csv(csv_file, index_col='date', parse_dates=['date'])
        target_idx = df.index.get_indexer(pd.DatetimeIndex([date]), method='nearest')[0]
        result = round(abs(df.iat[target_idx, 0]),2)
    except (KeyError, FileNotFoundError):
        result = 0
    return result


def get_inst_flows_pred(sn: str, date: str) -> pd.DataFrame:
    """Return predictive instant flow of a sn and date.

    Args:
        sn: An identifier of a vehicle or depot or demand.
        date: Date string like "%Y-%m-%d".

    Returns:
        The predictive result.
    """
    csv_file = INST_FLOWS_PRED + '/' + sn + '.csv'
    try:
        df = pd.read_csv(csv_file, index_col='date', parse_dates=['date'])
        df_dropped = df.replace(0, np.nan).dropna()
        all_dates = df_dropped.index.floor('D').drop_duplicates()
        target_date = pd.to_datetime(date)
        nearest_idx = all_dates.get_indexer([target_date], method='nearest')
        if nearest_idx == -1:
            return pd.DataFrame()
        nearest_date = all_dates[nearest_idx[0]].date()
        df = df[df.index.date == nearest_date]
    except (KeyError, FileNotFoundError):
        df = pd.DataFrame()
    return df

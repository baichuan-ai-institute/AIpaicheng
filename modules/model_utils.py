"""模型模块, 时间序列预测模型."""
import glob
import os
import pickle
from contextlib import redirect_stdout, redirect_stderr
from datetime import timedelta
from pmdarima.arima import auto_arima

from statsmodels.tsa.arima.model import ARIMA

from modules.quantity_utils import *

# 日供热量和日用热量(原始数据, 预处理结果, 滑窗结果, 模型文件, 预测结果, 窗口大小, 预测长度)
SUPPLIES_AND_DEMANDS_RAW_DATA = './data/supplies_and_demands/raw_data'
SUPPLIES_AND_DEMANDS_PREPROCESS = './data/supplies_and_demands/preprocess'
SUPPLIES_AND_DEMANDS_WINDOW = './data/supplies_and_demands/window'
SUPPLIES_AND_DEMANDS_MODEL = './data/supplies_and_demands/model'
SUPPLIES_AND_DEMANDS_PRED = './data/supplies_and_demands/pred'
SUPPLIES_AND_DEMANDS_WS = 9
SUPPLIES_AND_DEMANDS_HORIZON = 1

# 瞬时流量(同上)
INST_FLOWS_RAW_DATA = './data/inst_flows/raw_data'
INST_FLOWS_PREPROCESS = './data/inst_flows/preprocess'
INST_FLOWS_WINDOW = './data/inst_flows/window'
INST_FLOWS_MODEL = './data/inst_flows/model'
INST_FLOWS_PRED = './data/inst_flows/pred'
INST_FLOWS_WS = 240
INST_FLOWS_HORIZON = 24


def collect_raw_data(target_folder_path: str, inst_flow: bool = False):
    """Collect raw data used for prediction and save it to CSV files in the target_folder_path.

    Args:
        target_folder_path: A folder path to which the result is saved.
        inst_flow: Whether to collect instant flow data. Default is False.

    Returns:
        None
    """
    # 第一步, 收集原始数据
    wrap_log('Step 1/5 : Collect Raw Data')

    # 获取所有充放热站的SN
    sns = get_all_dep_dem_sns()

    # 所有日期
    all_dates = pd.date_range('2024-01-01', datetime.now()).strftime('%Y-%m-%d').tolist()
    if get_current_time() < 1380:
        all_dates = all_dates[:-1]

    for sn in sns:
        filepath = target_folder_path + '/' + sn + '.csv'
        try:
            existed_df = pd.read_csv(filepath, header=None, index_col=0, parse_dates=[0])
            existed_dates = existed_df.index.strftime('%Y-%m-%d').unique().tolist()
            # 只补充缺失的天数
            target_dates = [x for x in all_dates if x not in existed_dates]
        except FileNotFoundError:
            existed_df = pd.DataFrame()
            target_dates = all_dates

        # 获取目标日期的数据
        target_df = pd.DataFrame()
        if target_dates:
            if inst_flow:
                df_list = []
                for date in target_dates:
                    sub_df = dep_dem_hourly_flow(sn, date + ' 00:00:00', date + ' 23:59:59')
                    if sub_df.empty:
                        sub_df = pd.DataFrame(data=[None], index=[pd.Timestamp(date + ' 00:00:00')], columns=['value'])
                    df_list.append(sub_df)
                if df_list:
                    target_df = pd.concat(df_list, ignore_index=False)
                    target_df.columns = [1]
            else:
                data = [dep_dem_daily_quantity(sn, date + ' 00:00:00', date + ' 23:59:59') for date in target_dates]
                target_df = pd.DataFrame(data=data, index=pd.DatetimeIndex(target_dates), columns=[1])

        # 拼接新旧数据
        df = pd.concat([existed_df, target_df], ignore_index=False)

        # 去重
        df = df[~df.index.duplicated(keep='last')]

        # 按照日期重排
        df = df.sort_index()
        df.to_csv(filepath, header=False)


def preprocess(source_folder_path: str, target_folder_path: str, inst_flow: bool = False):
    """Preprocess raw data from source_folder_path, save the result to target_folder_path.

    Args:
        source_folder_path: A folder path to which raw data is saved.
        target_folder_path: A folder path to which the preprocessed result is saved.
        inst_flow: Whether to collect instant flow data. Default is False.

    Returns:
        None
    """
    # 第二步: 数据预处理
    wrap_log('Step 2/5 : Preprocess')
    csv_files = glob.glob(os.path.join(source_folder_path, '*.csv'))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None, index_col='date', names=['date', 'value'], parse_dates=['date'])

        # 去异常值
        if inst_flow:
            df[(df < 0) | (df > 20)] = None
        else:
            # 小于等于0的值置NULL
            df[df.round(0) <= 0] = None
            # 超过均值3倍的值置NULL
            df[df >= 3 * df.mean()] = None

        # 处理缺失值
        if inst_flow:
            df = df.dropna()
        else:
            # 用缺失值所在位置前后各3个元素(共6个)的平均值去填充
            df_roll = df.rolling(window=7, center=True, min_periods=1, ).mean().round(3)
            df = df.fillna(df_roll)

        # 只取最近30天的数据
        if inst_flow:
            df = df[int(-30 * 24):]
        else:
            df = df[-30:]

        # 保存
        filepath = csv_file.replace(source_folder_path, target_folder_path)
        df.to_csv(filepath)


# def slide_window(source_folder_path: str, target_folder_path: str, ws: int, horizon: int):
#     """Slide Window."""
#     # 第三步: 滑动窗口生成数据集
#     wrap_log('Step 3/5 : Slide Window')
#     csv_files = glob.glob(os.path.join(source_folder_path, '*.csv'))
#     for csv_file in csv_files:
#         df = pd.read_csv(csv_file, index_col='date', parse_dates=['date'])
#
#         # 时间序列
#         ts = df.values
#
#         # 若时间序列长度 < 窗口长度+预测长度
#         if len(ts) < ws + horizon + 1:
#             continue
#
#         # 滑动窗口生成X,y
#         # X --- 样本 {ndarray:(样本数, 变量数, 序列长度(样本长度))}
#         # Y --- 标签(预测值) {ndarray:(样本数,)}
#         X, y = SlidingWindow(ws, horizon=horizon)(ts)
#
#         # 保存
#         folder_path = target_folder_path + f'/ws_{ws}'
#         os.makedirs(folder_path, exist_ok=True)
#         sn = csv_file.removeprefix(source_folder_path + '/').removesuffix('.csv')
#         X_filepath = os.path.join(folder_path, 'X_' + sn + '.npy')
#         y_filepath = os.path.join(folder_path, 'y_' + sn + '.npy')
#         np.save(X_filepath, X)
#         np.save(y_filepath, y)


# def generate_lstm_model(source_folder_path: str, target_folder_path: str, ws: int, horizon: int,
#                         show_graph: bool = False, bs: int = 1, verbose = False, save: bool = False):
#     """Generate LSTM Model."""
#     # 第四步: 训练并生成模型
#     wrap_log('Step 4/5 : Generate Model')
#
#     # 滑窗数据集目录
#     folder_path = source_folder_path + f'/ws_{ws}'
#     model_folder_path = target_folder_path + '/ws_' + str(ws)
#
#     X_fps = glob.glob(os.path.join(folder_path, 'X_*.npy'))
#     y_fps = [x.replace('/X_', '/y_') for x in X_fps]
#     for X_fp, y_fp in list(zip(X_fps, y_fps)):
#         sn = X_fp.removeprefix(folder_path + '/X_').removesuffix('.npy')
#
#         # 加载数据集
#         X = np.load(X_fp)
#         y = np.load(y_fp)
#
#         # 划分训练集和验证集 splits: tuple[list, list]
#         splits = TimeSplitter(0.5, show_plot=False, fcst_horizon=horizon)(y)
#
#         # 模型参数
#         params = {
#             "X": X,
#             "y": y,
#             "splits": splits,
#             "path": model_folder_path,
#             "tfms": [None, TSForecasting()],
#             "batch_tfms": TSStandardize(),
#             "bs": bs,
#             "arch": "LSTMPlus",
#             "metrics": mape,
#             "seed": 42,
#         }
#
#         if show_graph:
#             params["cbs"] = ShowGraph()
#
#         # 创建时间序列预测模型
#         model = TSForecaster(**params)
#
#         if verbose:
#             # 训练模型
#             model.fit_one_cycle(5, 1e-2)
#         else:
#             # 训练模型(不打印日志)
#             with open(os.devnull, 'w') as f, redirect_stdout(f):
#                 model.fit_one_cycle(5, 1e-2)
#
#         # 保存模型
#         if save:
#             model.export(f"{sn}_lstm.pkl")


def generate_arima_model(source_folder_path: str, target_folder_path: str, save: bool = False, inst_flow=False):
    """Generate ARIMA(Autoregressive Integrated Moving Average) Model.

    Args:
        source_folder_path: Source folder path.
        target_folder_path: Target folder path.
        save: Whether to save model.
        inst_flow: Whether to generate instant flow model.

    Returns:
        None
    """
    # 第四步: 训练并生成模型
    wrap_log('Step 4/5 : Generate Model')

    csv_files = glob.glob(os.path.join(source_folder_path, '*.csv'))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col='date', parse_dates=['date'])
        df = df.dropna()
        if df.empty:
            continue

        # 训练集
        dataset = df['value'].to_numpy()
        train = dataset
        if not inst_flow:
            train = np.log10(train)

        # ARIMA模型
        # model = ARIMA(train)

        # AUTO-ARIMA模型

        # 训练
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            # fit_model = model.fit()
            model = auto_arima(
                train, start_p=1, start_q=1, max_d=5, error_action='ignore',
            )
            fit_model = model.fit(train)

        # 保存模型
        if save:
            sn = csv_file.removeprefix(source_folder_path).removesuffix('.csv')
            file_path = f"{target_folder_path}/{sn}_arima.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(fit_model, f)


# def lstm_pred(source_folder_path: str, target_folder_path: str, model_folder_path: str, ws: int, horizon: int,
#               inst_flow: bool = False) -> None:
#     """Use LSTM(Long Short-Term Memory) to predict."""
#     # 第五步: 预测
#     wrap_log('Step 5/5 : Pred')
#     # 模型文件夹
#     model_folder_path = model_folder_path + '/ws_' + str(ws)
#
#     # 所有模型文件路径
#     model_file_paths = glob.glob(model_folder_path + '/*_lstm.pkl')
#
#     # 预测结果文件夹
#     target_folder_path = target_folder_path + '/ws_' + str(ws)
#     os.makedirs(target_folder_path, exist_ok=True)
#
#     for model_file_path in model_file_paths:
#         # 加载模型
#         model = load_learner(model_file_path)
#
#         # 提取X
#         sn = model_file_path.removeprefix(model_folder_path + '/').removesuffix('_lstm.pkl')
#         X_file_path = source_folder_path + '/' + sn + '.csv'
#         X_df = pd.read_csv(X_file_path, index_col='date', parse_dates=['date'])
#         X = X_df.values[-ws:]
#         if X.shape[0] < ws:
#             X = X.reshape(-1)
#             pred = X[int(-1 * horizon):]
#             if len(pred) < horizon:
#                 pred = [0] * horizon
#         else:
#             X = X.reshape((1, 1, ws))
#             # 预测y, 不打印日志
#             with open(os.devnull, 'w') as f, redirect_stdout(f):
#                 _, _, preds = model.get_X_preds(X)
#                 pred = preds[0]
#
#         # 保存预测结果
#         data = {'value': pred}
#         if inst_flow:
#             if inst_flow == 'tomorrow':  # 预测第二天
#                 data['date'] = pd.date_range(datetime.now().date() + timedelta(days=1), periods=len(pred), freq='H')
#             elif inst_flow == 'today':  # 预测当天
#                 data['date'] = pd.date_range(datetime.now().date(), periods=len(pred), freq='H')
#             pass
#         else:
#             if get_current_time() > 1380:  # 预测第二天
#                 data['date'] = pd.to_datetime([(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')])
#             else:  # 预测当天
#                 data['date'] = pd.to_datetime([datetime.now().strftime('%Y-%m-%d')])
#
#         target_df = pd.DataFrame.from_records(data)
#         target_df = target_df.set_index('date')
#
#         y_filepath = target_folder_path + '/' + sn + '.csv'
#         try:
#             existed_df = pd.read_csv(y_filepath, index_col='date', parse_dates=['date'])
#         except FileNotFoundError:
#             existed_df = pd.DataFrame()
#
#         # 合并昨日和今日的数据
#         df = pd.concat([existed_df, target_df], ignore_index=False)
#
#         # 去除重复日期, 只保留最新计算结果
#         df = df[~df.index.duplicated(keep='last')]
#
#         # 按照日期重排
#         df = df.sort_index()
#
#         # 保存CSV文件
#         df.to_csv(y_filepath)


def arima_pred(source_folder_path: str, target_folder_path: str, horizon: int):
    """Predict by ARIMA.

    Args:
        source_folder_path: Source folder path.
        target_folder_path: A folder path to which the prediction result is saved.
        horizon: The forecasting steps.

    Returns:
        None
    """
    # 第五步: 预测
    wrap_log('Step 5/5 : Pred')

    cof = {
        # "200123011286": 2.01,  # 济宁翰蓝充热站
        # "200123011284": 4.20,  # 鲁宝放热站
        # "200123041331": 1.02,  # 奥宇包装放热站
        # "200123011273": 0.89,  # 岷山充热站
    }

    model_files = glob.glob(os.path.join(source_folder_path, '*.pkl'))
    for model_file in model_files:
        sn = model_file.removeprefix(source_folder_path + '/').removesuffix('_arima.pkl')
        # 加载模型
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

            # 使用加载的模型进行预测
            # pred = model.forecast(steps=horizon)
            pred = model.predict(n_periods=horizon)[0]

            data = {
                'value': np.round(abs(10**pred) * cof.get(sn, 1), 3)
            }

            if get_current_time() > 1380:  # 预测第二天
                data['date'] = pd.to_datetime([(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')])
            else:  # 预测当天
                data['date'] = pd.to_datetime([datetime.now().strftime('%Y-%m-%d')])

            y_filepath = target_folder_path + '/' + sn + '.csv'
            try:
                existed_df = pd.read_csv(y_filepath, index_col='date', parse_dates=['date'])
            except FileNotFoundError:
                existed_df = pd.DataFrame()

            # 如果预测值<=0, 则用前一周的平均值代替
            if pred <= 0 and not existed_df.empty:
                data['value'] = existed_df[-7:].mean().values

            target_df = pd.DataFrame(data).set_index('date')
            # 合并昨日和今日的数据
            df = pd.concat([existed_df, target_df], ignore_index=False)

            # 去除重复日期, 只保留最新计算结果
            df = df[~df.index.duplicated(keep='last')]

            # 按照日期重排
            df = df.sort_index()

            # 保存CSV文件
            df.to_csv(y_filepath)


def pred_inst_flow(source_folder_path: str, target_folder_path: str, horizon: int):
    """Predict instant flow.

    Args:
        source_folder_path: The folder containing source data.
        target_folder_path: The folder saving result.
        horizon:

    Returns:
        None
    """
    # 第五步: 预测
    wrap_log('Step 5/5 : Pred')

    csv_files = glob.glob(os.path.join(source_folder_path, '*.csv'))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col='date', parse_dates=['date'])
        if df.shape[0] < horizon:
            continue

        data = {
            'value': df['value'].iloc[int(-1 * horizon):].to_numpy()
        }
        if get_current_time() > 1380:  # 预测第二天
            data['date'] = pd.date_range((datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                                         periods=horizon, freq='H')
        else:  # 预测当天
            data['date'] = pd.date_range(datetime.now().strftime('%Y-%m-%d'),
                                         periods=horizon, freq='H')

        target_df = pd.DataFrame(data).set_index('date')
        filepath = csv_file.replace(source_folder_path, target_folder_path)
        try:
            existed_df = pd.read_csv(filepath, index_col='date', parse_dates=['date'])
        except FileNotFoundError:
            existed_df = pd.DataFrame()

        # 合并昨日和今日的数据
        df = pd.concat([existed_df, target_df], ignore_index=False)

        # 去除重复日期, 只保留最新计算结果
        df = df[~df.index.duplicated(keep='last')]

        # 按照日期重排
        df = df.sort_index()

        # 保存CSV文件
        df.to_csv(filepath)

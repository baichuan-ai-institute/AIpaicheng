# 用热量预测，不同模型（ARIMA、LSTM、RNN），效果对比
# Step 1: 获取原始数据（中牟项目的5个用户，世锦，景禧，瑞益1站，动康，茂源的日用热量）
# Step 2: 数据预处理
# Step 3: 训练模型并预测
# Step 4: 对比结果

import glob
import os
from contextlib import redirect_stdout, redirect_stderr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from tsai.basics import SlidingWindow, TimeSplitter, TSStandardize, TSForecasting, TSForecaster, ShowGraph, mae
from tsai.all import *

raw_data_folder = 'iter/240222/raw_data'
preprocess_folder = 'iter/240222/preprocess'
window_folder = 'iter/240222/window'

arima_pred_folder = 'iter/240222/pred/arima'
arima_vs_folder = 'iter/240222/vs/arima'
arima_vs_plot_folder = 'iter/240222/arima/vs_plot'

lstm_pred_folder = 'iter/240222/pred/lstm'
lstm_vs_folder = 'iter/240222/vs/lstm'
lstm_vs_plot_folder = 'iter/240222/vs_plot/lstm'

rnn_pred_folder = 'iter/240222/pred/rnn'
rnn_vs_folder = 'iter/240222/vs/rnn'
rnn_vs_plot_folder = 'iter/240222/vs_plot/rnn'

names = {
    '200123011219': '景禧',
    '200123041336': '瑞益1站',
    '200123051356': '世锦',
    '200123111205': '茂源',
    '200123111208': '动康',
}

params = {
    '200123011219': {
        # "order": (2, 0, 4),
        # "cof": 1.12,
        # "seasonal_order": (2, 0, 4, 5),
        "m": 7,
    },

    '200123041336': {
        # "order": (2, 0, 1),
        # "seasonal_order": (0,0,0,7),
        # "cof": 1.1,
        "m": 6,
    },

    '200123051356': {
        # "m": 1,
        # "cof": 1.5,
    },

    '200123111205': {
        "m": 2,
        "cof": 1.1,
    },
    '200123111208': {
        # "m": 8,
    },
}


def preprocess(source_folder_path: str, target_folder_path: str, inst_flow: bool = False):
    """Preprocess Raw Data."""
    # 第二步: 数据预处理
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

        if df.dropna().empty:
            continue

        # 处理缺失值
        if inst_flow:
            df = df.dropna()
        else:
            while df.isna().any().any():
                # 用缺失值所在位置前后各5个元素(共10个)的平均值去填充
                df_roll = df.rolling(window=11, center=True, min_periods=1).mean().round(3)
                df = df.fillna(df_roll)

        # # 只取最近30天的数据
        # if inst_flow:
        #     df = df[int(-30 * 24):]
        # else:
        #     df = df[-30:]

        # 保存
        filepath = csv_file.replace(source_folder_path, target_folder_path)
        df.to_csv(filepath)


def arima_pred(source_folder_path: str, target_folder_path: str, horizon):
    """Predict by ARIMA(Autoregressive Integrated Moving Average)."""
    csv_files = glob.glob(os.path.join(source_folder_path, '*.csv'))
    for csv_file in csv_files:
        sn = csv_file[-16:-4]
        df = pd.read_csv(csv_file, index_col='date', parse_dates=['date'])
        df = df.dropna()
        if df.empty:
            continue

        # 拆分数据集
        dataset = df['value'].to_numpy()
        train, test = dataset[:int(-1 * horizon)], dataset[int(-1 * horizon):]
        train_log10 = np.log10(train)
        test_log10 = np.log10(test)

        order = params.get(sn, {}).get("order", (0, 0, 0))
        seasonal_order = params.get(sn, {}).get("seasonal_order", (0, 0, 0, 0))
        cof = params.get(sn, {}).get("cof", 1)
        m = params.get(sn, {}).get("m", 1)
        preds = []
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            for i in range(horizon):
                # # ARIMA模型
                # model = ARIMA(
                #     train_log10,
                #     order=order,
                #     seasonal_order=seasonal_order,
                # )
                # fit_model = model.fit()
                # pred = fit_model.forecast(steps=1)[0]

                # AUTO-ARIMA模型
                model = auto_arima(
                    train_log10, start_p=0, start_q=0, max_d=5, error_action='ignore', m=m,
                )
                fit_model = model.fit(train_log10)
                pred = fit_model.predict(n_periods=1)[0]

                train_log10 = np.append(train_log10, test_log10[i])
                preds.append(np.round(10 ** pred * cof, 2))

        data = {
            "date": df.index[int(-1 * horizon):],
            'value': preds,
        }

        file_path = csv_file.replace(source_folder_path, target_folder_path)
        try:
            existed_df = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
        except FileNotFoundError:
            existed_df = pd.DataFrame()

        target_df = pd.DataFrame(data).set_index('date')

        # 去除重复日期, 只保留最新计算结果
        df = pd.concat([existed_df, target_df], ignore_index=False)
        df = df[~df.index.duplicated(keep='last')]

        # 按照日期重排
        df = df.sort_index()

        # 保存CSV文件
        df.to_csv(file_path)


def cal_rmse(x, y) -> float:
    squared_errors = (x - y) ** 2
    mse = squared_errors.mean()
    rmse = np.round(np.sqrt(mse), 3)
    return rmse


def get_vs(gt_folder, pred_folder, vs_folder, vs_plot_folder, model_name, ws=None):
    gt_csv_files = glob.glob(os.path.join(gt_folder, '*.csv'))
    for gt_csv in gt_csv_files:
        sn = gt_csv[-16:-4]
        gt_df = pd.read_csv(gt_csv, index_col='date', parse_dates=['date'])
        # gt_df = gt_df.iloc[-37:,:]

        if ws:
            pred_csv = gt_csv.replace(gt_folder, pred_folder + f'/ws_{ws}')
        else:
            pred_csv = gt_csv.replace(gt_folder, pred_folder)

        pred_df = pd.read_csv(pred_csv, index_col='date', parse_dates=['date'])

        vs_df = gt_df.join(pred_df, how='left', lsuffix='_gt', rsuffix='_arima')
        vs_df.columns = ['真实值', model_name]

        # 计算RMSE
        rmse = cal_rmse(vs_df['真实值'], vs_df[model_name])
        print(sn, rmse)

        # 画图
        fig, ax = plt.subplots()
        colors = ['b', 'orange', 'green']
        # vs_df.plot(
        #     ax=ax, marker='o', ylabel='用热量(t)', ylim=[0, 90],
        #     grid=True, legend=True, color=colors, title=names.get(sn, ''),
        # )

        # vs_df['运营统计值'].plot(ax=ax, label='运营统计值', marker='o')
        vs_df['真实值'].plot(ax=ax, label='真实值', marker='o', color='b')
        vs_df[model_name].plot(ax=ax, label=model_name, marker='o', color='orange')

        plt.ylabel('用热量(t)')
        # plt.ylim([0,90])
        plt.legend()
        plt.grid(axis='y')
        plt.title(names.get(sn, '') + '用热量预测')
        ax.autoscale(axis='x', tight=False)

        # plt.show()

        # 保存结果
        vs_df.to_csv(gt_csv.replace(gt_folder, vs_folder))
        fig.savefig(gt_csv.replace(gt_folder, vs_plot_folder).replace('.csv', '.png'))


def truncate_raw_data(source_folder_path, target_folder_path):
    csv_files = glob.glob(os.path.join(source_folder_path, '*.csv'))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None, index_col='date', names=['date', 'value'], parse_dates=['date'])
        truncated_df = df[df.index < '2024-02-01']
        truncated_df.to_csv(csv_file.replace(source_folder_path, target_folder_path), header=False)


def slide_window(source_folder_path: str, target_folder_path: str, ws: int, horizon: int):
    """Slide Window."""
    csv_files = glob.glob(os.path.join(source_folder_path, '*.csv'))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col='date', parse_dates=['date'])

        # 时间序列
        ts = df.values

        # ts = np.concatenate((ts,ts,ts))

        # 若时间序列长度 < 窗口长度+预测长度 + 1
        if len(ts) < ws + horizon + 1:
            continue

        # 滑动窗口生成X,y
        # X --- 样本 {ndarray:(样本数, 变量数, 序列长度(样本长度))}
        # Y --- 标签(预测值) {ndarray:(样本数,)}
        X, y = SlidingWindow(ws, horizon=horizon)(ts)

        # 保存
        folder_path = target_folder_path + f'/ws_{ws}'
        os.makedirs(folder_path, exist_ok=True)
        sn = csv_file.removeprefix(source_folder_path + '/').removesuffix('.csv')
        X_filepath = os.path.join(folder_path, 'X_' + sn + '.npy')
        y_filepath = os.path.join(folder_path, 'y_' + sn + '.npy')
        np.save(X_filepath, X)
        np.save(y_filepath, y)


def lstm_pred(source_folder_path: str, target_folder_path: str, preprocess_folder_path, ws: int, horizon: int,
              show_graph: bool = False, bs: int = 1, verbose = False, epoch=50):
    """Predict by LSTM."""
    # 滑窗数据集目录
    folder_path = source_folder_path + f'/ws_{ws}'

    X_fps = glob.glob(os.path.join(folder_path, 'X_*.npy'))
    y_fps = [x.replace('/X_', '/y_') for x in X_fps]
    target_folder_path = target_folder_path + '/ws_' + str(ws)
    os.makedirs(target_folder_path, exist_ok=True)
    for X_fp, y_fp in list(zip(X_fps, y_fps)):
        sn = X_fp.removeprefix(folder_path + '/X_').removesuffix('.npy')

        # 加载数据集
        X = np.load(X_fp)
        y = np.load(y_fp)

        # 划分训练集和验证集 splits: tuple[list, list]
        splits = TimeSplitter(10, show_plot=False)(y)

        # TS_tfms = [
        #     TSIdentity,
        #     TSMagAddNoise,
        #     (TSMagScale, .02, .2),
        #     (partial(TSMagWarp, ex=0), .02, .2),
        #     (partial(TSTimeWarp, ex=[0, 1, 2]), .02, .2),
        # ]
        # batch_tfms = [TSStandardize(), RandAugment(TS_tfms, N=3, M=5)]

        # 模型参数
        params = {
            "X": X,
            "y": y,
            "splits": splits,
            "tfms": [None, TSForecasting()],
            "batch_tfms": TSStandardize(),
            # "batch_tfms": batch_tfms,
            "bs": bs,
            "arch": "LSTM",
            "metrics": mae,
            "seed": 42,
        }

        if show_graph:
            params["cbs"] = ShowGraph()

        # 创建时间序列预测模型
        model = TSForecaster(**params)

        if verbose:
            # 训练模型
            model.fit_one_cycle(epoch, 1e-2)
        else:
            # 训练模型(不打印日志)
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                model.fit_one_cycle(epoch, 1e-2)

        raw_preds, target, preds = model.get_X_preds(X[splits[1]], y[splits[1]])
        preds = np.round(preds, 2)
        print(preds)

        df = pd.read_csv(f"{preprocess_folder_path}/{sn}.csv", index_col='date', parse_dates=['date'])

        target_df = pd.DataFrame(preds, index=df.index[int(-1 * horizon):], columns=['value'])
        target_df.index.name = 'date'

        file_path = f"{target_folder_path}/{sn}.csv"
        try:
            existed_df = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
        except FileNotFoundError:
            existed_df = pd.DataFrame()

        # 去除重复日期, 只保留最新计算结果
        df = pd.concat([existed_df, target_df], ignore_index=False)
        df = df[~df.index.duplicated(keep='last')]

        # 按照日期重排
        df = df.sort_index()

        # 保存CSV文件
        df.to_csv(file_path)


def rnn_pred(source_folder_path: str, target_folder_path: str, preprocess_folder_path, ws: int, horizon: int,
              show_graph: bool = False, bs: int = 1, verbose = False, epoch=50):
    """Predict by RNN."""
    # 滑窗数据集目录
    folder_path = source_folder_path + f'/ws_{ws}'

    X_fps = glob.glob(os.path.join(folder_path, 'X_*.npy'))
    y_fps = [x.replace('/X_', '/y_') for x in X_fps]
    target_folder_path = target_folder_path + '/ws_' + str(ws)
    os.makedirs(target_folder_path, exist_ok=True)
    for X_fp, y_fp in list(zip(X_fps, y_fps)):
        sn = X_fp.removeprefix(folder_path + '/X_').removesuffix('.npy')

        # 加载数据集
        X = np.load(X_fp)
        y = np.load(y_fp)

        # 划分训练集和验证集 splits: tuple[list, list]
        splits = TimeSplitter(10, show_plot=False)(y)

        # 模型参数
        params = {
            "X": X,
            "y": y,
            "splits": splits,
            "tfms": [None, TSForecasting()],
            "batch_tfms": TSStandardize(),
            "bs": bs,
            "arch": "RNN",
            "metrics": mae,
            "seed": 42,
        }

        if show_graph:
            params["cbs"] = ShowGraph()

        # 创建时间序列预测模型
        model = TSForecaster(**params)

        if verbose:
            # 训练模型
            model.fit_one_cycle(epoch, 1e-2)
        else:
            # 训练模型(不打印日志)
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                model.fit_one_cycle(epoch, 1e-2)

        raw_preds, target, preds = model.get_X_preds(X[splits[1]], y[splits[1]])
        print(preds)
        preds = np.round(preds, 2)

        df = pd.read_csv(f"{preprocess_folder_path}/{sn}.csv", index_col='date', parse_dates=['date'])

        target_df = pd.DataFrame(preds, index=df.index[int(-1 * horizon):], columns=['value'])
        target_df.index.name = 'date'

        file_path = f"{target_folder_path}/{sn}.csv"
        try:
            existed_df = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
        except FileNotFoundError:
            existed_df = pd.DataFrame()

        # 去除重复日期, 只保留最新计算结果
        df = pd.concat([existed_df, target_df], ignore_index=False)
        df = df[~df.index.duplicated(keep='last')]

        # 按照日期重排
        df = df.sort_index()

        # 保存CSV文件
        df.to_csv(file_path)


def get_vs3(gt_folder, pred_folders, vs_folder, vs_plot_folder):
    gt_csv_files = glob.glob(os.path.join(gt_folder, '*.csv'))
    for gt_csv in gt_csv_files:
        sn = gt_csv[-16:-4]
        gt_df = pd.read_csv(gt_csv, index_col='date', parse_dates=['date'])
        model_names = ['ARIMA', "LSTM", "RNN"]
        vs_df = gt_df
        for pred_folder,model_name in zip(pred_folders, model_names):

            pred_csv = gt_csv.replace(gt_folder, pred_folder)

            pred_df = pd.read_csv(pred_csv, index_col='date', parse_dates=['date'])

            pred_df.columns = [model_name]
            vs_df = vs_df.join(pred_df, how='left', lsuffix='_gt', rsuffix='_'+model_name)

        vs_df.columns = ['真实值', 'ARIMA', "LSTM", "RNN"]

        # 画图
        fig, ax = plt.subplots()
        colors = ['b', 'orange', 'green']

        # vs_df['运营统计值'].plot(ax=ax, label='运营统计值', marker='o')
        # vs_df['真实值'].plot(ax=ax, label='真实值', marker='o', color='b')
        # vs_df[model_name].plot(ax=ax, label=model_name, marker='o', color='orange')
        vs_df.plot(ax=ax, marker='o')

        plt.ylabel('用热量(t)')
        # plt.ylim([0,90])
        plt.legend()
        plt.grid(axis='y')
        plt.title(names.get(sn, '') + '用热量预测结果对比')
        ax.autoscale(axis='x', tight=False)

        # plt.show()

        # 保存结果
        vs_df.to_csv(gt_csv.replace(gt_folder, vs_folder))
        fig.savefig(gt_csv.replace(gt_folder, vs_plot_folder).replace('.csv', '.png'))




def main():
    # preprocess(raw_data_folder, preprocess_folder)
    # arima_pred(preprocess_folder, arima_pred_folder, 10)
    # get_vs(preprocess_folder, arima_pred_folder, vs_folder, vs_plot_folder)
    # slide_window(preprocess_folder, window_folder, 10, 1)

    lstm_pred(window_folder, lstm_pred_folder, preprocess_folder, ws=10, horizon=10, show_graph=False, bs=8, epoch=200)
    get_vs(preprocess_folder, lstm_pred_folder, lstm_vs_folder, lstm_vs_plot_folder, 'LSTM', ws=10)
    #
    # rnn_pred(window_folder, rnn_pred_folder, preprocess_folder, ws=10, horizon=10, show_graph=False, bs=8, epoch=200)
    # get_vs(preprocess_folder, rnn_pred_folder, rnn_vs_folder, rnn_vs_plot_folder, 'LSTM', ws=10)

    # get_vs3(preprocess_folder, [arima_pred_folder, lstm_pred_folder + '/ws_10', rnn_pred_folder +'/ws_10'], 'iter/240222/vs3', 'iter/240222/vs_plot3')

if __name__ == "__main__":
    main()

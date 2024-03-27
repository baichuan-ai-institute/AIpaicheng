import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from contextlib import redirect_stdout, redirect_stderr
from pmdarima.arima import auto_arima

raw_data_folder = 'iter/240130/raw_data'
preprocess_folder = 'iter/240130/preprocess'
old_pred_folder = 'data/supplies_and_demands/pred'
new_pred_folder = 'iter/240130/pred'
gt_folder = 'data/supplies_and_demands/gt'
vs_folder = 'iter/240130/vs'
vs_plot_folder = 'iter/240130/vs_plot'

names = {
    '200123051356': '世锦',
    '200123011219': '景禧',
    '200123041336': '瑞益1站',
    '200123111208': '动康',
    '200123111205': '茂源',
}

orders = {
    '200123051356': (0,0,0),
    '200123011219': (0,1,0),#(2,1,5),
    '200123041336': (0,1,0),
    '200123111208': (0,2,1),
    '200123111205': (2,0,1),
}

seasonal_orders = {
    #'200123011219': (0,0,0,7),
    '200123051356': (0,0,0,7),
    '200123041336': (0,0,0,7),
}

cof = {
    '200123051356': 1.55,
    '200123041336': 1.05,
    '200123111205': 1.3,
}


def cal_rmse(x, y) -> float:
    squared_errors = (x - y) ** 2
    mse = squared_errors.mean()
    rmse = np.round(np.sqrt(mse), 3)
    return rmse


def get_vs():
    gt_csv_files = glob.glob(os.path.join(gt_folder, '*.csv'))
    for gt_csv in gt_csv_files:
        sn = gt_csv[-16:-4]
        gt_df = pd.read_csv(gt_csv, index_col='date', parse_dates=['date'])
        old_pred_df = pd.read_csv(gt_csv.replace(gt_folder, old_pred_folder), index_col='date', parse_dates=['date'])
        new_pred_df = pd.read_csv(gt_csv.replace(gt_folder, new_pred_folder), index_col='date', parse_dates=['date'])
        raw_df = pd.read_csv(gt_csv.replace(gt_folder, raw_data_folder), header=None, index_col='date', names=['date', 'value'], parse_dates=['date'])

        vs_df = gt_df.join(raw_df, how='inner', lsuffix='_gt', rsuffix='_raw')
        vs_df = vs_df.join(old_pred_df, how='inner', lsuffix='_gt', rsuffix='_old_pred')
        vs_df = vs_df.join(new_pred_df, how='inner', lsuffix='_gt_', rsuffix='_new_pred')
        vs_df.columns = ['运营统计值', '流量计统计值','预测值(旧)','预测值(新)']

        # 计算RMSE
        old_rmse = cal_rmse(vs_df['预测值(旧)'], vs_df['运营统计值'])
        new_rmse = cal_rmse(vs_df['预测值(新)'], vs_df['运营统计值'])
        print(sn, old_rmse,new_rmse)

        # 画图
        fig, ax = plt.subplots()
        colors= ['b', 'purple', 'orange', 'green']
        # vs_df.plot(
        #     ax=ax, marker='o', ylabel='用热量(t)', ylim=[0, 90],
        #     grid=True, legend=True, color=colors, title=names.get(sn, ''),
        # )

        # vs_df['运营统计值'].plot(ax=ax, label='运营统计值', marker='o')
        vs_df['流量计统计值'].plot(ax=ax, label='真实值', marker='o', color='purple')
        # vs_df['预测值(旧)'].plot(ax=ax, label='预测值(旧)', marker='o', color='orange')
        vs_df['预测值(新)'].plot(ax=ax, label='预测值', marker='o', color='green')
        #
        plt.ylabel('用热量(t)')
        plt.ylim([0,90])
        plt.legend()
        plt.grid()
        plt.title(names.get(sn, ''))
        ax.autoscale(axis='x', tight=False)

        # plt.show()

        # 保存结果
        vs_df.to_csv(gt_csv.replace(gt_folder, vs_folder))
        fig.savefig(gt_csv.replace(gt_folder, vs_plot_folder).replace('.csv', '.png'))


def iter_pred(source_folder_path: str, target_folder_path: str, horizon):
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

        preds = []
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            for i in range(horizon):

                # ARIMA模型
                model = ARIMA(
                    train_log10,
                    order=orders.get(sn, (0, 0, 0)),
                    seasonal_order=seasonal_orders.get(sn, (0, 0, 0, 0)),
                )
                fit_model = model.fit()
                pred = fit_model.forecast(steps=1)[0]

                # AUTO-ARIMA模型
                # model = auto_arima(
                #     train_log10, start_p=1, start_q=1, max_d=5, error_action='ignore',
                # )
                # fit_model = model.fit(train_log10)
                # pred = fit_model.predict(n_periods=1)[0]

                train_log10 = np.append(train_log10, test_log10[i])
                preds.append(np.round(10 ** pred * cof.get(sn,1),2))
        print(sn, fit_model)
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

        # 处理缺失值
        if inst_flow:
            df = df.dropna()
        else:
            # 用缺失值所在位置前后各5个元素(共10个)的平均值去填充
            df_roll = df.rolling(window=11, center=True, min_periods=1, ).mean().round(3)
            df = df.fillna(df_roll)

        # 只取最近30天的数据
        if inst_flow:
            df = df[int(-30 * 24):]
        else:
            df = df[-30:]

        # 保存
        filepath = csv_file.replace(source_folder_path, target_folder_path)
        df.to_csv(filepath)


def main():
    print()
    # preprocess(raw_data_folder, preprocess_folder)
    iter_pred(preprocess_folder, new_pred_folder, 17)
    get_vs()


if __name__ == '__main__':
    main()

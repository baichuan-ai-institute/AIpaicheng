import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import HourLocator, DateFormatter

# df_180 = pd.read_csv('demo/data/180_20231105_instant_flow.csv', parse_dates=[0], index_col=[0], header=None)
df_179 = pd.read_csv('demo/data/179_20231105_instant_flow.csv', parse_dates=[0], index_col=[0], header=None)


def get_cumsum_df(df):
    inst_flow_ser = df[1]

    # 持续时间[数组]
    duration_df = (inst_flow_ser.index - inst_flow_ser.index[0]).seconds.to_frame(index=False).diff().dropna()
    duration_array = duration_df[0].to_numpy() / 3600

    # 平均瞬时流量[数组]
    avg_flow_ser = ((inst_flow_ser + inst_flow_ser.shift(-1)) / 2).dropna()
    avg_flow_array = avg_flow_ser.to_numpy()

    # 充热站充热量或放热站放热量 = sum(单位时间 * 平均瞬时流量)
    data = avg_flow_array * duration_array

    quantity_df = pd.DataFrame(data, index=df.index[:-1])
    cumsum_df = quantity_df.cumsum()

    return cumsum_df


# df_180_cumsum = get_cumsum_df(df_180)
df_179_cumsum = get_cumsum_df(df_179)

# 创建一张画布和两个子图，2行1列的布局
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(6.61, 2 * 4))
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6.61*3, 4 * 3))

# 在第一个子图(ax1)中画图
df_179[1].plot(ax=ax1, label='瞬时流量', xlabel='时间', color='green')

# 画一条垂直直线，x=2.5
y1_min, y1_max = ax1.get_ylim()


scheme_dep_to_dem_start_ts = pd.DatetimeIndex(['2023-11-05 13:08:00', '2023-11-05 22:07:00'])
scheme_dep_to_dem_end_ts = pd.DatetimeIndex(['2023-11-05 13:39:00','2023-11-05 22:38:00'])
scheme_dem_to_dep_start_ts = pd.DatetimeIndex(['2023-11-05 22:38:00','2023-11-05 23:58:00'])
scheme_dem_to_dep_end_ts = pd.DatetimeIndex(['2023-11-05 23:09:00','2023-11-06 00:29:00'])

# for start_ts,end_ts in zip(scheme_dep_to_dem_start_ts,scheme_dep_to_dem_end_ts):
for start_ts,end_ts in zip(scheme_dem_to_dep_start_ts,scheme_dem_to_dep_end_ts):
    ax1.vlines(x=start_ts, ymin=y1_min, ymax=y1_max, color='#9ACD4F', linestyle='--',linewidth=1.5)
    ax1.vlines(x=end_ts, ymin=y1_min, ymax=y1_max, color='red', linestyle='--',linewidth=1.5)
    print((end_ts-start_ts).seconds/60)

ax1.set_title('荣润建材')
ax1.set_ylabel('t/h', rotation=0, labelpad=10)
ax1.legend(labels=['瞬时流量', '计划排程(热源->用户)发车时间', '计划排程(热源->用户)到达时间'],loc='upper left')


real_dep_to_dem_start_ts = pd.DatetimeIndex(['2023-11-05 09:38:00','2023-11-05 16:15:00'])
real_dep_to_dem_end_ts = pd.DatetimeIndex(['2023-11-05 09:50:00','2023-11-05 16:28:00'])
real_dem_to_dep_start_ts = pd.DatetimeIndex(['2023-11-05 09:55:00','2023-11-05 16:35:00'])
real_dem_to_dep_end_ts = pd.DatetimeIndex(['2023-11-05 10:05:00','2023-11-05 16:50:00'])

# 在第二个子图(ax2)中画图
df_179[1].plot(ax=ax2, label='瞬时流量', xlabel='时间' , color='green')
# for start_ts,end_ts in zip(real_dep_to_dem_start_ts,real_dep_to_dem_end_ts):
for start_ts,end_ts in zip(real_dem_to_dep_start_ts,real_dem_to_dep_end_ts):
    ax2.vlines(x=start_ts, ymin=y1_min, ymax=y1_max, color='#9ACD4F', linestyle='-',linewidth=1.5)
    ax2.vlines(x=end_ts, ymin=y1_min, ymax=y1_max, color='red', linestyle='-',linewidth=1.5)
    print((end_ts - start_ts).seconds / 60)
ax2.set_ylabel('t/h', rotation=0, labelpad=10)
ax2.legend(labels=['瞬时流量', '实际运营(热源->用户)发车时间', '实际运营(热源->用户)到达时间'], loc='upper left')
#

df_179_cumsum[0].plot(ax=ax3, label='累计流量', xlabel='时间', color='green')
ax3.set_ylabel('t', rotation=0, labelpad=10)
ax3.legend()

# df_179[1].plot(ax=ax3, label='瞬时流量', xlabel='时间', color='green')
# ax3.set_title('荣润建材')
# ax3.set_ylabel('t/h', rotation=0, labelpad=10)
# ax3.legend()
#
# df_179_cumsum[0].plot(ax=ax4, label='累计流量', xlabel='时间', color='green')
# ax4.set_ylabel('t', rotation=0, labelpad=10)
# ax4.legend()

# 格式化X轴日期
ax1.xaxis.set_major_locator(HourLocator(interval=2))  # 设置主要刻度为每小时
ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(HourLocator(interval=2))  # 设置主要刻度为每小时
ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax3.xaxis.set_major_locator(HourLocator(interval=2))  # 设置主要刻度为每小时
ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
# ax4.xaxis.set_major_locator(HourLocator(interval=2))  # 设置主要刻度为每小时
# ax4.xaxis.set_major_formatter(DateFormatter('%H:%M'))

plt.tight_layout()
plt.show()

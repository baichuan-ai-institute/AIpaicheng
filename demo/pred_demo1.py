# 热源供热量预测DEMO
from tkinter import *
from tkinter import ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def _plot_last_30(equip_id):
    file_path = f'../data/daily_supplies_dataset/preprocess/{str(equip_id)}.csv'
    df = pd.read_csv(file_path, index_col='date', parse_dates=['date'])

    # df = df[df.index]
    df = df.iloc[-30:, ]

    x = df.index.strftime('%m-%d')
    y = df['value'].values
    print(x)
    print(y)
    # 在同一画布上绘制图形，但不清除之前的内容
    ax.plot(x, y, marker='o', label=depot_info[equip_id] + '历史供热', color=color_conf[equip_id], markersize=8)
    ax.legend()
    ax.set_xticklabels(x, rotation=45)
    ax.relim()
    ax.autoscale_view()

    # 更新画布
    canvas.draw()


def _plot_pred_1(equip_id):
    file_path = f'../data/daily_supplies_dataset/pred/ws_10/{str(equip_id)}.csv'
    df = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
    df = df.iloc[-1:, ]

    x = df.index.strftime('%m-%d')
    y = df['value'].values

    # 在同一画布上绘制图形，但不清除之前的内容
    ax.plot(x, y, marker='*', label=depot_info[equip_id], color=pred_color_conf[equip_id], markersize=10)

    ax.legend()
    ax.set_xticklabels(x, rotation=45)
    ax.relim()
    ax.autoscale_view()

    # 更新画布
    canvas.draw()


def clear_canvas():
    ax.clear()  # 清除Axes上的所有绘图
    canvas.draw()


# Create the main Tkinter window
root = Tk()
root.title("预测模块")

# 创建Matplotlib画布和初始的Axes
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()
ax.set_xlabel('日期')
ax.set_ylabel('吨')
ax.set_title('历史供热量+预测供热量')

# 热源信息
depot_info = {
    44: '安阳岷山',
    21: '柘城光大',
    35: '济宁瀚蓝',
}

color_conf = {
    44: '#EC7270',
    21: '#95DA69',
    35: '#5B79E8',
}

pred_color_conf = {
    44: 'r',
    21: 'g',
    35: 'b',
}

for depot_id, depot_name in depot_info.items():
    button1 = ttk.Button(root, text=depot_name + '历史', command=lambda equip_id=depot_id: _plot_last_30(equip_id))
    button2 = ttk.Button(root, text=depot_name + '预测', command=lambda equip_id=depot_id: _plot_pred_1(equip_id))
    button1.pack()
    button2.pack()

# 清空画布按钮
ttk.Button(root, text="清空画布", command=clear_canvas).pack()

# 退出按钮
ttk.Button(root, text="Exit", command=root.destroy).pack()
root.mainloop()

"""测试模块, 测试各个模块的功能."""
from pprint import pprint

import pandas as pd

from data_utils import *
from kafka_utils import *
from model_utils import *
from or_utils import *
from pred_utils import *
from quantity_utils import *

print()
# target = {
#     1: [49, 96, 0, 1, 47], 2: [96, 143, 1, 0, 47], 3: [145, 192, 0, 1, 47], 4: [192, 239, 1, 0, 47],
#     5: [241, 288, 0, 1, 47], 6: [288, 335, 1, 0, 47], 7: [337, 384, 0, 1, 47], 8: [384, 431, 1, 0, 47],
#     9: [433, 480, 0, 1, 47], 10: [480, 527, 1, 0, 47], 11: [529, 576, 0, 1, 47], 12: [576, 623, 1, 0, 47],
#     13: [625, 672, 0, 1, 47], 14: [672, 719, 1, 0, 47], 15: [721, 768, 0, 1, 47], 16: [768, 815, 1, 0, 47],
#     17: [817, 864, 0, 1, 47], 18: [864, 911, 1, 0, 47], 19: [913, 960, 0, 1, 47], 20: [960, 1007, 1, 0, 47],
#     21: [1009, 1056, 0, 1, 47], 22: [1056, 1103, 1, 0, 47], 23: [1105, 1152, 0, 1, 47], 24: [1152, 1199, 1, 0, 47],
#     25: [1201, 1248, 0, 1, 47], 26: [1248, 1295, 1, 0, 47], 27: [1297, 1344, 0, 1, 47], 28: [1344, 1391, 1, 0, 47],
#     29: [1393, 1440, 0, 1, 47], 30: [1440, 1487, 1, 0, 47], 31: [1465, 1512, 0, 1, 47], 32: [1512, 1559, 1, 0, 47],
#     33: [1537, 1584, 0, 1, 47], 34: [1584, 1631, 1, 0, 47], 35: [1609, 1656, 0, 1, 47], 36: [1656, 1703, 1, 0, 47],
#     37: [1681, 1728, 0, 1, 47], 38: [1728, 1775, 1, 0, 47], 39: [1753, 1800, 0, 1, 47], 40: [1800, 1847, 1, 0, 47],
#     41: [1825, 1872, 0, 1, 47], 42: [1872, 1919, 1, 0, 47], 43: [1897, 1944, 0, 1, 47], 44: [1944, 1991, 1, 0, 47],
#     45: [1969, 2016, 0, 1, 47], 46: [2016, 2063, 1, 0, 47], 47: [2041, 2088, 0, 1, 47], 48: [2088, 2135, 1, 0, 47],
#     49: [2113, 2160, 0, 1, 47], 50: [2160, 2207, 1, 0, 47], 51: [2185, 2232, 0, 1, 47], 52: [2232, 2279, 1, 0, 47],
#     53: [2257, 2304, 0, 1, 47], 54: [2304, 2351, 1, 0, 47], 55: [2329, 2376, 0, 1, 47], 56: [2376, 2423, 1, 0, 47],
#     57: [2401, 2448, 0, 1, 47], 58: [2448, 2495, 1, 0, 47], 59: [2473, 2520, 0, 1, 47], 60: [2520, 2567, 1, 0, 47],
#     61: [2545, 2592, 0, 1, 47], 62: [2592, 2639, 1, 0, 47], 63: [2617, 2664, 0, 1, 47], 64: [2664, 2711, 1, 0, 47],
#     65: [2689, 2736, 0, 1, 47], 66: [2736, 2783, 1, 0, 47], 67: [2761, 2808, 0, 1, 47], 68: [2808, 2855, 1, 0, 47]
# }
#
# sol = assign(target, 5)
# import pandas as pd
#
# from data_utils import *
# from model_utils import *
# from tqdm import tqdm
# import numpy
# import time
# for i in get_all_dep_equip_ids():
#     print(i, get_equip_name(i))


#
#
# result = collect_prj_info_and_pred()

# executed_list = collect_executed()
# print(daily_supply(35, '2023-09-08'))
# print(daily_demand(45, '2023-09-11'))
# print(daily_demand(66, '2023-09-11'))
# print(daily_demand(105, '2023-09-11'))
# print(daily_demand(108, '2023-09-11'))
# print(daily_scheme(40, '2023-09-18'))
# pprint(daily_execution(40, '2023-09-27'))


# print(wrap_log('hello world'))

# dep_ids = get_all_dep_equip_ids()
# dem_ids = get_all_dem_equip_ids()

# for i in dep_ids:
#     q = dep_dem_daily_quantity(i, '2023-09-22 00:00:00', '2023-09-22 23:59:59')
#     print(i, q)
#
# for i in dem_ids:
#     q = dep_dem_daily_quantity(i, '2023-09-22 00:00:00', '2023-09-22 23:59:59')
#     print(i, q)

# parser = configparser.ConfigParser()
# parser.read(CONFIG_FILE_PATH)
# config=parser['MongoDB']
# host, port = config['Server'], config['Port']

# print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] ', sys.version)

# raise ValueError('这是一次异常')

# data={
#   "depot_id": 40,
#   "demand_ids": [
#     45,
#     66,
#     105,
#     108
#   ],
#   "veh_ids": [
#     100,
#     110,
#     120,
#     130,
#     140,
#     150,
#     160
#   ],
#   "tank_ids": [
#             68, 67, 64, 63, 62,
#             61, 56, 55, 54, 53,
#             52, 51, 50, 104, 106,
#             107, 149, 151, 152, 153,
#   ],
#   "datestr": "2023-10-12"
# }


# result = daily_scheme(**data)

# get_gps_dict([41])

# data = {
#   "datestr": "2023-10-16",
#   "tank_ids":
#   [
#     68,
#     67,
#     64,
#     63,
#     62,
#     61,
#     56,
#     55,
#     54,
#     53,
#     52,
#     51,
#     50,
#     104,
#     106,
#     107,
#     149,
#     151,
#     152,
#     153
#   ],
#   "depot_id": 21,
#   "demand_ids":
#   [
#     45,
#     66,
#     105,
#     108
#   ],
#   "veh_ids":
#   [
#     100,
#     110,
#     120,
#     130,
#     140,
#     150,
#     160
#   ]
# }
#
# result = daily_scheme(**data)

# ids = get_all_equip_ids()
# for i in ids:
#   sn = id_map_sn(i)
#   belong_to = id_map_belong_to(i)
#   print(i, sn, belong_to)

# data = {
#     "datestr":"2023-10-16",
#     "tank_ids":[96,90,89,23,99,93,70,33,31,30,27,122,119,104,88],
#     "depot_id":35,
#     "demand_ids":[91,39],
#     "veh_ids":[1687376539856932866,1697518660637765634,1669550179402911746,1686177246441639938,1686177245741191169
#     ]
#
# }
#
# result = daily_scheme(**data)

# data = daily_demand(39, '2023-10-17')
# pprint(data)
# data2 = daily_demand(39, '2023-10-18')
# pprint(data2)
#
# data3 = daily_demand(91, '2023-10-17')
# pprint(data3)
#
# data4 = daily_demand(91, '2023-10-18')
# pprint(data4)

# data = {
#     "datestr":"2023-10-17",
#     "tank_ids":[96,90,89,23,99,93,70,33,31,30,27,122,119,104,88],
#     "depot_id":35,
#     "demand_ids":[91,39],
#     "veh_ids":[1687376539856932866,1697518660637765634,1669550179402911746,1686177246441639938,1686177245741191169],
#     "save": False,
# }
#
# result = daily_scheme(**data)
# pprint(result)

# data = {
#     "datestr":"2023-10-18",
#     "tank_ids":[96,90,89,23,99,93],
#     "depot_id":21,
#     "demand_ids":[22,23,24],
#     "veh_ids":[1687376539856932866,1697518660637765634,1669550179402911746]
# }
#
# result = daily_scheme(**data)


# df = pd.read_csv('/home/daidai/gy/dynamic/es/data/dem_inst_flows_dataset/raw_data/91.csv',
#                  header=None, index_col='date', names=['date', 'value'], parse_dates=['date'])
# start_ts = pd.Timestamp('2023-10-07')
# end_ts = pd.Timestamp('2023-10-18')
# df = df[(start_ts<=df.index) & (df.index<end_ts)]
# #df = df[~((start_ts<=df.index) & (df.index<end_ts))]
# df.plot()
# plt.show()

# csv_files = glob.glob(os.path.join(DEM_INST_FLOWS_RAW_DATA, '*.csv'))
# for csv_file in tqdm(csv_files):


# csv_file = './data/dem_inst_flows_dataset/raw_data/45.csv'
#     df = pd.read_csv(csv_file, header=None, index_col='date', names=['date', 'value'], parse_dates=['date'])
#     df = df[~((start_ts <= df.index) & (df.index <= end_ts))]
#     df.to_csv(csv_file, header=False)

# data = {"date": "2023-10-19",
#  "demand_ids": [93],
#  "depot_id": 99,
#  "tank_ids": [46],
#  "veh_ids": [1714583474747314177, 1714584276849233922]}
#
# result = daily_execution(**data)

# data ={'date': '2023-10-19',
#  'demand_ids': [91, 39],
#  'depot_id': 35,
#  'tank_ids': [89, 70, 33, 31, 30, 119],
#  'veh_ids': [1687376539856932900, 1697518660637765600, 1669550179402911700]}
#
# result = daily_scheme(**data)
# equip_sn = id_map_sn(157)
# data = get_hbase_df_period(equip_sn, pd.Timestamp('2023-10-24 00:00:00'), pd.Timestamp('2023-10-24 23:59:59'))


# deps = get_all_dep_equip_ids()
# dems = get_all_dem_equip_ids()
# print(deps)
# print(dems)

# data = {
#     "date":"2023-10-25",
#     "tank_ids":[96,94,88,86,85,78,76,75,73,71,87,83,82,81,80,77,72],
#     "depot_id":35,
#     "demand_ids":[157,39,34,18,16,15,14,91],
#     "veh_ids":[1716627333035692034,1716341736454852609,1716341416853082114,1716627668554846209,1716626450889674754,1716341565876703234, 1716341565876703235],
# }
#
# result = daily_scheme(**data)

# all_tables = get_all_table_names_in_hbase()

# all_days = [str(x, encoding='utf-8').split('_')[1] for x in all_tables]
#
# all_days2 = sorted(list(set(x for x in all_days)))
# all_days3 = sorted(list(set(str(x, encoding='utf-8').split('_')[1] for x in all_tables)))
#
# print(all_days2 == all_days3)

# pred_date = 'today'
#
# if pred_date == 'tomorrow':
#     all_days = sorted(list(set(str(x, encoding='utf-8').split('_')[1] for x in all_tables)))
# elif pred_date == 'today':
#     all_days = sorted(list(set(str(x, encoding='utf-8').split('_')[1] for x in all_tables if
#                                not x.endswith(bytes(datetime.now().strftime('%Y-%m-%d'), encoding='utf-8')))))
# else:
#     all_days = []
#
# print(all_days[-1])

# ids = [46, 47, 43, 103, 150, 158, 180, 179, 205, 203, 195, 194, 185, 184]
# for i in ids:
#     df = get_hbase_df_period(i, pd.Timestamp('2023-10-24 00:00:00'), pd.Timestamp('2023-10-24 12:00:00'))
#     print(i, df)
#     print()

# print(daily_demand(46, '2023-10-24'))


# print()
# dems = get_all_dem_equip_ids()
# print(dems)
#
# deps = get_all_dep_equip_ids()
# print(deps)

# # 21
# params = {
#   "date": "2023-10-31",
#   "tank_ids": [
#     96,
#     90,
#     89,
#     23,
#     99,
#     93,
#     70,
#     33,
#     31,
#     30,
#     27,
#     122,
#     119,
#     104,
#     88
#   ],
#   "depot_id": 21,
#   "demand_ids": [
#     22,
#     23,
#     24,
#   ],
#   "veh_ids": [
#     1687376539856932900,
#     1697518660637765600,
#     1669550179402911700,
#     1686177246441640000,
#     1686177245741191200
#   ]
# }
#
# # 35
# params = {
#   "date": "2023-10-31",
#   "tank_ids": [
#     96,
#     90,
#     89,
#     23,
#     99,
#     93,
#     70,
#     33,
#     31,
#     30,
#     27,
#     122,
#     119,
#     104,
#     88,
#     301,
#     302,
#     304,
#     305,
#     306,
#     307,
#   ],
#   "depot_id": 35,
#   "demand_ids": [
#       14,15,16,17,18,38,39,91,157,158
#   ],
#   "veh_ids": [
#     1687376539856932900,
#     1697518660637765600,
#     1669550179402911700,
#     1686177246441640000,
#     1686177245741191200,
#     1686177245741191300,
#     1686177245741191200,
#   ]
# }
#
# # 205
# params = {
#   "date": "2023-10-31",
#   "tank_ids": [
#     96,
#     90,
#     89,
#     23,
#     99,
#     93,
#     70,
#     33,
#     31,
#     30,
#     27,
#     122,
#     119,
#     104,
#     88
#   ],
#   "depot_id": 205,
#   "demand_ids": [
#     203
#   ],
#   "veh_ids": [
#     1687376539856932900,
#     1697518660637765600,
#     1669550179402911700,
#     1686177246441640000,
#     1686177245741191200
#   ]
# }
#
# # 40
# params = {
#   "date": "2023-10-31",
#   "tank_ids": [
#     96,
#     90,
#     89,
#     23,
#     99,
#     93,
#     70,
#     33,
#     31,
#     30,
#     27,
#     122,
#     119,
#     104,
#     88,
#     11,
#     12,
#     14,
#     15,
#     16,
#     17,
#   ],
#   "depot_id": 40,
#   "demand_ids": [
#       65,66,45,105,108
#   ],
#   "veh_ids": [
#     1687376539856932900,
#     1697518660637765600,
#     1669550179402911700,
#     1686177246441640000,
#     1686177245741191200,
#     1686177245741191300,
#     1686177245741191200,
#   ]
# }
#
# # 44
# params = {
#   "date": "2023-11-01",
#   "tank_ids": [
#     96,
#     90,
#     89,
#     23,
#     99,
#     93,
#     70,
#     33,
#     31,
#     30,
#     27,
#     122,
#     119,
#     104,
#     88,
#     11,
#     12,
#     14,
#     15,
#     16,
#     17,
#   ],
#   "depot_id": 40,
#   "demand_ids": [
#       180, 179
#   ],
#   "veh_ids": [
#     1687376539856932900,
#     1697518660637765600,
#     1669550179402911700,
#     1686177246441640000,
#     1686177245741191200,
#     1686177245741191300,
#     1686177245741191200,
#   ]
# }
#
# # data = daily_execution(**params)
# data = daily_scheme(**params)
# pprint(data)


params = {
    "date": "2023-10-31",
    "tank_ids": [
        96,
        90,
        89,
        23,
        99,
        93,
        70,
        33,
        31,
        30,
        27,
        122,
        119,
        104,
        88
    ],
    "depot_id": 21,
    "demand_ids": [
        22,
        23,
        24,
    ],
    "veh_ids": [
        1687376539856932900,
        1697518660637765600,
        1669550179402911700,
        1686177246441640000,
        1686177245741191200
    ]
}

# data = daily_execution(**params)
# pprint(data)

#
# tables = get_all_table_names_in_hbase()
# pprint(tables)

# params = {
#   "date":"2023-11-03",
#   "tank_ids":[96,94,88,86,85,78,76,75,73,71,87,83,82,81,80,77,72],
#   "depot_id":35,
#   "demand_ids":[39,34,15],
#   "veh_ids":
#     [1719969010584489985,1716627668554846209,1716341736454852609,1717116321680953346,1716626450889674754,1716341416853082114,1716627333035692034,1716341565876703234]}
# result1 = daily_scheme(**params)
# result2 = daily_execution(**params)
#
# pprint(result1)
# pprint(result2)

# # 测试预测供热量
# for depot_id in get_all_dep_equip_ids():
#   params = {
#     "date": "2023-11-17",
#     "depot_id": depot_id,
#   }
#   test_supply = daily_supply(**params)
#   print(depot_id, test_supply)
#   print()

# # 测试预测用热量
# for demand_id in get_all_dem_equip_ids():
#   params = {
#     "date": "2023-11-17",
#     "demand_id": demand_id,
#   }
#   test_supply = daily_demand(**params)
#   print(demand_id, test_supply)
#   print()

# 测试计划排程

params = {
    "date": "2023-11-03",
    "tank_ids": [
        96,
        90,
        89,
        23,
        99,
        93,
        70,
        33,
        31,
        30,
        27,
        122,
        119,
        104,
        88,
        11,
        12,
        14,
        15,
        16,
        17,
    ],
    "depot_id": 217,
    "demand_ids": [
        65, 66, 45, 105, 108
    ],
    "veh_ids": [
        1687376539856932900,
        1697518660637765600,
        1669550179402911700,
        1686177246441640000,
        1686177245741191200,
        1686177245741191300,
        1686177245741191200,
    ]
}

params = {
    "date": "2023-11-17",
    "tank_ids": [
        96,
        90,
        89,
        78,
        75,
        23,
        99,
        93,
        70,
        33,
        31,
        30,
        27,
        122,
        119,
        104,
        88,
        11,
        12,
        14,
        15,
    ],
    "depot_id": 35,
    "demand_ids": [
        39, 34, 15
    ],
    "veh_ids": [
        1687376539856932900,
        1697518660637765600,
        1669550179402911700,
        1686177246441640000,
        1686177245741191200,
        1686177245741191300,
        1686177245741191200
    ]
}
#

# params = {
#   "date": "2023-11-03",
#   "tank_ids": [
#     96,
#     90,
#     89,
#     23,
#     99,
#     93,
#     70,
#     33,
#     31,
#     30,
#     27,
#     122,
#     119,
#     104,
#     88
#   ],
#   "depot_id": 21,
#   "demand_ids": [
#     22,
#     23,
#     24,
#   ],
#   "veh_ids": [
#     1687376539856932900,
#     1697518660637765600,
#     1669550179402911700,
#     1686177246441640000,
#     1686177245741191200
#   ]
# }
params = {
    "date": "2023-12-07",
    "tank_ids": [96, 94, 88, 86, 85, 78, 76, 75, 73, 71, 87, 83, 82, 81, 77, 72],
    "depot_id": 35,
    "demand_ids": [39, 157, 16, 18, 15, 14, 34, 91],
    "veh_ids": [1719969010584489985, 1716341565876703234, 1716341416853082114, 1717116321680953346, 1716627668554846209,
                1716627333035692034, 1716626450889674754, 1716341736454852609],
}
# test_scheme = daily_scheme(**params)
# test_scheme = daily_execution(**params)
# pprint(test_scheme)
#
# print(test_scheme['scheme_type'])

# depot_ids = get_all_dep_equip_ids()
# print(depot_ids)
# demand_ids = get_all_dem_equip_ids()
# print(demand_ids)


# equip_names = ['0Y83', '0V76', '6W97', '9Z53', '2v60', '7D03', '5E83']
# equip_ids = [get_equip_id(name) for name in equip_names]
# print(equip_ids)
# sn = id_map_sn(179)
# log_df = get_hbase_df_period(sn, pd.Timestamp('2023-11-05 00:00:00'), pd.Timestamp('2023-11-05 23:59:59'))

# equip_id = 303
# sn = id_map_sn(equip_id)
# h_df = get_hbase_df_period(sn, pd.Timestamp('2023-11-06 00:00:00'), pd.Timestamp('2023-11-06 00:10:00'))
# if current_time := get_current_time() >= 1380:
#   print('23点之后')
# else:
#   print('23点之前')


# params = {
#   "date": "2023-12-12",
#   "depot_id": 21,
#   "demand_ids": [22, 34, 14],
#   "veh_ids": [1734415793653911553],
#   "tank_ids": [99, 93, 119, 98],
# }
#
#
#
# data = daily_execution(**params)


params = {
    "date": "2023-12-13",
    "tank_ids": [51376, 51379, 51374, 122, 41344, 81382]
    , "depot_id": 35,
    "demand_ids": [39, 91],
    "veh_ids": [1724676967228952577, 1687376539856932866, 1697518660637765634,
                1669550179402911746, 1729428336403386369, 1729439440877752322,
                1729417612302061573, 1729444994903805953],
}

params = {
    "date": "2023-12-14",
    "tank_ids": [51376, 51379, 51374, 30, 122, 114, 113, 104, 41344, 75, 67, 112, 81382, 163],
    "depot_id": 35, "demand_ids": [91, 39],
    "veh_ids": [1687376539856932866, 1729439440877752322, 1729417612302061573, 1729444994903805953,
                1729428336403386369],
}

params = {"date": "2023-12-14",
          "depot_id": 35,
          "demand_ids": [91, 39],
          "veh_ids": [1687376539856932866, 1729439440877752322, 1729417612302061573, 1729444994903805953, 1729428336403386369],
          "tank_ids": [51376, 51379, 51374, 30, 122, 114, 113, 104, 41344, 75, 67, 112, 81382, 163]
          }


# data = daily_scheme(**params)
# data = daily_execution(**params)

# supply = get_pred_supply(21, '2023-9-12')
# print(supply)
#
# supply2 = daily_supply(25, '2023-12-13')
# pprint(supply2)

# demand = get_pred_demand(39, '2023-12-13')
# print(demand)
#
# demand2 = daily_demand(39, '2023-12-13')
# pprint(demand2)
#
# demand2 = daily_demand(39, '2023-12-14')
# pprint(demand2)
#
# demand3 = daily_demand(39, '2023-12-11')
# pprint(demand3)

# inst_flow = get_pred_inst_flow(39, '2023-12-11')
# print(inst_flow)

# params = {
#     "date": "2023-12-13",
#     "depot_id": 21,
#     "demand_ids": [22, 34, 14],
#     "veh_ids": [1734415793653911553],
#     "tank_ids": [99, 93, 119, 98],
# }
# scheme = daily_scheme(**params)
# exe_scheme = daily_execution(**params)

# dep_ids = get_all_dep_equip_ids()
# print(dep_ids, len(dep_ids))
# dem_ids = get_all_dem_equip_ids()
# print(dem_ids, len(dem_ids))
# for i in dep_ids + dem_ids:
#     sn = id_map_sn(i)
#     print(i, sn)

# for i,j in zip(dep_ids, dem_ids):
#     check_node(i, [j])

# dep_gps = get_gps_dict(dep_ids)
# pprint(dep_gps)
# dem_gps = get_gps_dict(dem_ids)
# pprint(dem_gps)

# params = {
#     'date': '2023-12-14',
#     'depot_id': 44,
#     'demand_ids': [180, 179]
#     , 'veh_ids': [1645313025854455810, 1645315134058446849],
#     'tank_ids': [183, 181, 177, 99, 74, 178, 95, 90],
# }
#
# exe_scheme = daily_execution(**params)
# params = {
#     'date': '2023-12-15',
#     'depot_id': 217,
#     'demand_ids': [105, 66, 45],
#     'veh_ids': [1716337822250668033, 1716340249150459905, 1716340464729296898, 1716340656882946050, 1716341230676316162],
#     'tank_ids': [153, 107, 104, 68, 64, 61, 56, 53, 52, 152, 151, 149, 67, 204],
# }
#
# params = {"date": "2023-12-15", "depot_id": 35, "demand_ids": [157, 39, 34, 18, 16, 15, 14, 91], "veh_ids": [1719969010584489985, 1716341565876703234, 1716341416853082114, 1717116321680953346, 1716627668554846209, 1716627333035692034, 1716626450889674754, 1716341736454852609], "tank_ids": [96, 94, 88, 86, 85, 78, 76, 75, 73, 71, 87, 83, 82, 81, 77, 72]}
#
#
# sch_scheme = daily_scheme(**params)
# exe_scheme = daily_execution(**params)
# token = get_token()
# print(token)


#metric_df = get_metric_df("200123011286", pd.Timestamp("2023-12-14 00:00:00"), pd.Timestamp("2023-12-14 23:59:59"))
# q,df = dep_dem_daily_quantity(35, "2023-12-14 00:00:00", "2023-12-14 23:59:59", output_raw=True)
# q,df = dep_dem_daily_quantity(39, "2023-12-10 00:00:00", "2023-12-10 23:59:59", output_raw=True)
# print(q)

# params_217 = {
#     'date': '2023-12-15',
#     'depot_id': 217,
#     'demand_ids': [105, 66, 45],
#     'veh_ids': [1716337822250668033, 1716340249150459905, 1716340464729296898, 1716340656882946050, 1716341230676316162],
#     'tank_ids': [153, 107, 104, 68, 64, 61, 56, 53, 52, 152, 151, 149, 67, 204],
# }
# sch_scheme_217 = daily_scheme(**params_217)
# exe_schem_217 = daily_execution(**params_217)

# params_35 = {
#     "date": "2023-12-15",
#     "depot_id": 35,
#     "demand_ids": [157, 39, 34, 18, 16, 15, 14, 91],
#     "veh_ids": [1719969010584489985, 1716341565876703234, 1716341416853082114, 1717116321680953346, 1716627668554846209, 1716627333035692034, 1716626450889674754, 1716341736454852609],
#     "tank_ids": [96, 94, 88, 86, 85, 78, 76, 75, 73, 71, 87, 83, 82, 81, 77, 72],
# }
# sch_scheme_35 = daily_scheme(**params_35)
# exe_schem_35 = daily_execution(**params_35)

params_44 = {
    'date': '2023-12-15',
    'depot_id': 44,
    'demand_ids': [180, 179],
    'veh_ids': [1645313025854455810, 1645315134058446849],
    'tank_ids': [183, 181, 177, 99, 74, 178, 95, 90],
}
# sch_scheme_44 = daily_scheme(**params_44)
# exe_schem_44 = daily_execution(**params_44)
#
# params_205 = {
#     'date': '2023-12-15',
#     'depot_id': 205,
#     'demand_ids': [203],
#     'veh_ids': [1716628383020978177],
#     'tank_ids': [197, 200, 199, 207],
# }
# sch_scheme_205 = daily_scheme(**params_205)
# exe_schem_205 = daily_execution(**params_205)

# dep_ids = get_all_dep_equip_ids()
# dem_ids = get_all_dem_equip_ids()

start_time = '2023-12-17 00:00:00'
end_time = '2023-12-17 23:59:59'

# equip_id = 105
# equip_id = 45
# equip_sn = id_map_sn(equip_id)
# start_ts = pd.Timestamp(start_time)
# end_ts = pd.Timestamp(end_time)
#
#
# q = dep_dem_daily_quantity(equip_id, start_time, end_time)
# q_df = get_metric_df(equip_sn, start_ts, end_ts)
# Q_df = get_metric_df(equip_sn, start_ts, end_ts, "ces_s_st_cd")

# raw_data = './data/supplies_and_demands/raw_data'
# raw_data = './data/supplies_and_demands/raw_data'
# raw_data3 = './'
#
# csv_files = glob.glob(os.path.join(raw_data, '*.csv'))
#
# for csv_file in csv_files:
#     df = pd.read_csv(csv_file, header=None, index_col='date', names=['date', 'value'], parse_dates=['date'])
#     if csv_file == './data/supplies_and_demands/raw_data/200123011219.csv':
#         pass
#     csv_file2 = csv_file.replace('raw_data', 'raw_data')
#     df2 = pd.read_csv(csv_file2, header=None, index_col='date', names=['date', 'value'], parse_dates=['date'])
#
#     # df['value2'] = df2['value'].to_numpy()
#
#     data3_list = []
#     # df_drop = df.dropna()
#     # df2_drop = df2.dropna()
#     #
#     # df3 = pd.concat([df_drop, df2_drop])
#
#     for i in range(df.shape[0]):
#         data = df['value'].iat[i]
#         data2 = df2['value'].iat[i]
#         if (data is not float(np.nan)) and (data2 is not float(np.nan)):
#             data3 = max(data, data2)
#         elif (data is not float(np.nan)) and (data2 is float(np.nan)):
#             data3 = data
#         elif (data is float(np.nan)) and (data2 is not float(np.nan)):
#             data3 = data2
#         else:
#             data3 = None
#         data3_list.append(data)
#
#     df3 = pd.DataFrame(data3_list, index= df.index)
#     if csv_file == './data/supplies_and_demands/raw_data/200123011219.csv':
#         df4 = df.dropna()
#         pass
#     csv_file3 = csv_file.replace('raw_data', 'raw_data3')
#     df3.to_csv(csv_file3)


# from statsmodels.tsa.arima.model import ARIMA
#
# # 读取时间序列数据
# df = pd.read_csv('./data/supplies_and_demands/preprocess/200123011219.csv', index_col=['date'], parse_dates=['date'])
#
#
# # 拆分数据集
# train_size = int(len(df) -1)
# train, test = df[:train_size].to_numpy(), df[train_size:].to_numpy()
#
# # 训练ARIMA模型
# model = ARIMA(train) #order=(p, d, q)
# fit_model = model.fit()
#
# # 模型评估
# predictions = fit_model.forecast(steps=len(test))
# print(predictions)


# sn = '200123011273'
# for sn in get_all_dep_dem_sns():
#     result = daily_demand(sn, '2024-01-15')
#     pprint(result)
#     print()



params_217 = {
    'date': '2024-01-15',
    'depot_id': 205,
    'demand_ids': [105, 45, 66, 355, 354, 357],
    'veh_ids': [1716337822250668033, 1716340249150459905, 1716340464729296898, 1716340656882946050, 1716341230676316162],
    'tank_ids': [149, 151, 152, 153, 52, 53, 56, 61, 320, 64, 321, 322, 67, 68, 204, 348, 349, 350, 104, 107],
}

params_217 = {
    'date': '2024-01-15',
    'depot_id': '200123081370',
    'demand_ids': ['200123051356', '200123011219', '200123041336', '200123111208', '200123111291', '200123111205'],
    'veh_ids': [1716337822250668033, 1716340249150459905, 1716340464729296898, 1716340656882946050, 1716341230676316162],
    'tank_ids': [149, 151, 152, 153, 52, 53, 56, 61, 320, 64, 321, 322, 67, 68, 204, 348, 349, 350, 104, 107],
}

# sch_scheme_217 = daily_scheme(**params_217)
# params_217 = {
#     'date': '2024-01-15',
#     'depot_id': '200123081370',
#     'demand_ids': ['200123051356', '200123011219', '200123041336'],
#     'veh_ids': [1716337822250668033, 1716340249150459905, 1716340464729296898, 1716340656882946050, 1716341230676316162],
#     'tank_ids': [149, 151, 152, 153, 52, 53, 56, 61, 320, 64, 321, 322, 67, 68, 204, 348, 349, 350, 104, 107],
# }

# params_217_2 = {"date": "2024-01-19", "depot_id": "200123011291", "demand_ids": ["200123011282", "200123011293", "200123011285"], "veh_ids": [1734415793653911553], "tank_ids": ["200123051371"]}
# params_217_2 = {"date":"2024-01-19","tank_ids":["200123051376","200123051353","200123051374","200123051389","200123051424","200123051412","200123051414","200123051426","200123041350","200123051379"],"depot_id":"200123011286","demand_ids":["200123041331","200123011284"],"veh_ids":[1687376539856932866,1729444994903805953,1729428336403386369,1729439440877752322,1729417612302061573]}

params_217 = {"date": "2024-01-19", "depot_id": "200123011286", "demand_ids": ["200123041331", "200123011284"], "veh_ids": [1687376539856932866, 1729444994903805953, 1729428336403386369, 1729439440877752322, 1729417612302061573], "tank_ids": ["200123051376", "200123051353", "200123051374", "200123051389", "200123051424", "200123051412", "200123051414", "200123051426", "200123041350", "200123051379"]}
params_217 = {
  "date": "2024-01-19",
  "depot_id": "200123081370",
  "demand_ids": [
    "200123051356",
    "200123011219",
    "200123041336"
  ],
  "veh_ids": [
    1687376539856932900,
    1697518660637765600,
    1669550179402911700,
    1686177246441640000,
    1686177245741191200,
    1686177245741191300,
    1686177245741191200
  ],
  "tank_ids": [
    "200123081341",
    "200123051355",
    "200123051412",
    "200123041337",
    "200123041345",
    "200123041339",
    "200123041348",
    "200123011220",
    "200123011215",
    "200123081345",
    "200123081342",
    "200123081343",
    "200123041350",
    "200123081373",
    "200123111292",
    "200123111262",
    "200123111261",
    "200123111292",
    "200123111262",
    "200123111261",
    "200123111204",
    "200123111203",
    "200123111206",
    "200123111203"
  ]
}

# sch_scheme_217 = daily_scheme(**params_217)
# exe_scheme_217 = daily_execution(**params_217)
# print_task_type(sch_scheme_217)
# print_task_type({'scheme':[]})
print('hello')
sn = "200123081341"
# print(get_device_sns("ces"))
# v = get_latest_value("esc", sn, "esc_T_R_TbPsP2")
# sn = "200123051356"
start_time = '2024-03-14 00:00:00'
end_time = '2024-03-14 01:00:00'
# df = get_period_df(sn, start_time, end_time, "ces_S_R_Ifr")
# print(df)
# gps = get_gps(sn)
# print(gps)

# fence = get_all_e_fence()
# pprint(fence)
sns = [
    "200123051356",
    "200123011219",
    "200123041336"
  ]

# gpss = get_gps_dict(sns)
# print(gpss)
date = "2024-01-19"
depot_id = "200123081370"
demand_ids = [
                 "200123051356",
                 "200123011219",
                 "200123041336"
             ]
veh_ids = [
              1687376539856932900,
              1697518660637765600,
              1669550179402911700,
              1686177246441640000,
              1686177245741191200,
              1686177245741191300,
              1686177245741191200
          ]
tank_ids = [
    "200123081341",
    "200123051355",
    "200123051412",
    "200123041337",
    "200123041345",
    "200123041339",
    "200123041348",
    "200123011220",
    "200123011215",
    "200123081345",
    "200123081342",
    "200123081343",
    "200123041350",
    "200123081373",
    "200123111292",
    "200123111262",
    "200123111261",
    "200123111292",
    "200123111262",
    "200123111261",
    "200123111204",
    "200123111203",
    "200123111206",
    "200123111203"
]
# scheme = daily_scheme(date,depot_id,demand_ids,veh_ids,tank_ids)
# pprint(scheme)

# print(daily_supply("200123011286","2024-03-15"))
print(daily_demand("200123051356","2024-03-15"))

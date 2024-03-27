# 安阳项目计划排程
# 热源：安阳岷山充装站
# 用户：利华制药，荣润建材
# 时间：2023-11-05

from pprint import pprint

from modules.or_utils import *


class DemoScheme(DailyScheme):
    def __init__(self, data):
        super().__init__(data)

    def get_distance_matrix(self, locations: list[tuple]) -> dict[int, dict[int, int]]:
        """Return a distance matrix of locations."""
        distance_matrix = {}
        for from_counter, from_node in enumerate(locations):
            distance_matrix[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distance_matrix[from_counter][to_counter] = 0
                else:
                    # 测地线距离
                    distance_matrix[from_counter][to_counter] = math.ceil(self.cof * geodesic(from_node, to_node).m)
        return distance_matrix


data = {
    "date": "2023-11-05",
    "depot_id": 44,  # 安阳岷山充装站
    "demand_ids": [180, 179],  # 利华制药，安阳荣润
    "veh_ids": [
        '豫A7290Y', #'豫A0627U', #'豫A0888E',
    ],
    "tank_ids": [
        '0Y83', '0V76', '6W97', '9Z53', '2v60', '7D03', #'5E83',
    ],
    "gps": {
        44: (36.029829834925195, 114.27325831401535),
        180: (36.0630212384106, 114.373449851858),
        179: (36.03415850239441, 114.30443121641106),

    },
    "pred": {
        180: {
            'demand': 3.52 * 5,
            'inst_flow': pd.DataFrame(
                data=[
                    0.75088778, 1.26536995, 0.45968192, 0.26674107, 0.23284327, 0.36231465, 0.23659143, 0.64450115,
                    1.5483666, 1.74008822, 2.31329719, 0.69203522, 0.31491346, 0.31635346, 0.33533961, 0.60770949,
                    0.68294935, 0.5482786, 1.18045551, 1.96518687, 2.43364654, 1.47637115, 2.10082097, 0.97810652,
                ],
                index=pd.DatetimeIndex(
                    [
                        '2023-11-05 00:00:00', '2023-11-05 01:00:00', '2023-11-05 02:00:00', '2023-11-05 03:00:00',
                        '2023-11-05 04:00:00', '2023-11-05 05:00:00', '2023-11-05 06:00:00', '2023-11-05 07:00:00',
                        '2023-11-05 08:00:00', '2023-11-05 09:00:00', '2023-11-05 10:00:00', '2023-11-05 11:00:00',
                        '2023-11-05 12:00:00', '2023-11-05 13:00:00', '2023-11-05 14:00:00', '2023-11-05 15:00:00',
                        '2023-11-05 16:00:00', '2023-11-05 17:00:00', '2023-11-05 18:00:00', '2023-11-05 19:00:00',
                        '2023-11-05 20:00:00', '2023-11-05 21:00:00', '2023-11-05 22:00:00', '2023-11-05 23:00:00',
                    ], dtype='datetime64[ns]', name='date'),
                columns=['value'],
            ),
        },
        179: {
            'demand': 3.52 * 2,
            'inst_flow': pd.DataFrame(
                data=[
                    0.0, 0.0, 0.0, 0.04016528, 0.79272795, 0.78265838, 0.77241094, 0.76388592, 0.72458262, 0.52445185,
                    0.66210273, 0.60387315, 0.60912511, 0.61428421, 0.60420653, 0.49821409, 0.37484679, 0.49390228,
                    0.63562632, 0.82939849, 0.83108108, 0.77446685, 0.53733448, 0.56992732,
                ],
                index=pd.DatetimeIndex(
                    [
                        '2023-11-05 00:00:00', '2023-11-05 01:00:00', '2023-11-05 02:00:00', '2023-11-05 03:00:00',
                        '2023-11-05 04:00:00', '2023-11-05 05:00:00', '2023-11-05 06:00:00', '2023-11-05 07:00:00',
                        '2023-11-05 08:00:00', '2023-11-05 09:00:00', '2023-11-05 10:00:00', '2023-11-05 11:00:00',
                        '2023-11-05 12:00:00', '2023-11-05 13:00:00', '2023-11-05 14:00:00', '2023-11-05 15:00:00',
                        '2023-11-05 16:00:00', '2023-11-05 17:00:00', '2023-11-05 18:00:00', '2023-11-05 19:00:00',
                        '2023-11-05 20:00:00', '2023-11-05 21:00:00', '2023-11-05 22:00:00', '2023-11-05 23:00:00',
                    ], dtype='datetime64[ns]', name='date'),
                columns=['value'],
            ),
        },
    },
}

scheme = DemoScheme(data)
scheme.get_scheme_ovrpdptw()
# scheme.save('./demo/data/daily_scheme')

# 结果
result = scheme.scheme
pprint(result)

real_veh_w_tank = [[scheme.tank_ids[i] for i in j] for j in scheme.veh_w_tank]

for i, j in enumerate(scheme.veh_sol):
    out_str = ''
    pre_node = None
    for k, node in enumerate(j[:-1]):
        if node != 0 and node != pre_node:
            out_str += f'{node}{scheme.ovrpdptw_tt[i][k]}[{real_veh_w_tank[i][k]}] --- '
            pre_node = node
    out_str += f'{j[-1]}{scheme.ovrpdptw_tt[i][-1]}[{real_veh_w_tank[i][-1]}]'
    print(out_str)

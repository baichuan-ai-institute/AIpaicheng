"""运筹学模块, 运筹优化最优解模型."""
import math
import os
from copy import deepcopy
from json import dump, load, loads, JSONDecodeError

import lkh
import numpy as np
from geopy.distance import geodesic
from gurobipy import tupledict, tuplelist
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from ortools.sat.python import cp_model

from modules.data_utils import *
from modules.kafka_utils import KAFKA_LOG_FOLDER
from modules.pred_utils import get_supplies_and_demands_pred, get_inst_flows_pred

# 日计划排程存放目录
DAILY_SCHEME = './data/daily_scheme'

# 日实际排程存放目录
DAILY_EXECUTION = './data/daily_execution'


class DailyScheme:
    """Daily Scheme"""

    def __init__(self, data):
        self.depot_id: int = data.get('depot_id')  # 热源ID
        self.demand_ids: list = data.get('demand_ids', [])  # 用户ID 列表
        self.veh_ids: list = data.get('veh_ids', [])  # 车头ID 列表
        self.tank_ids: list = data.get('tank_ids', [])  # 罐箱ID 列表
        self.pred: dict = data.get('pred', {})  # 用户的预测数据
        self.gps: dict = data.get('gps', {})  # gps数据
        self.datestr: str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        self.params: dict = {'date': self.datestr, 'depot_id': self.depot_id, 'demand_ids': self.demand_ids,
                             'veh_ids': self.veh_ids, 'tank_ids': self.tank_ids}  # 接口传参
        self.speed = 60  # 车速(km/h)
        self.cof = 1  # 距离系数
        self.service_time: int = 22  # 服务时间(入场 + 甩挂 + 出场)
        self.window: int = 30  # due_time(min) - ready_time(min)
        self.tank_avg_cap: float = 3.52  # 移动罐平均每罐放热量(t/罐)
        self.recharge_time: int = 40  # 再充热时间(min)
        self.cal_map_real: dict = {}  # 映射字典: dict[计算ID, 真实ID]
        self.lkh_solver_path: str = data.get('lkh_solver_path', './modules/LKH-3.0.7/LKH')  # lkh包路径
        self.vrpspdtw_prob: lkh.LKHProblem | None = None  # vrpspdtw本体
        self.vrpspdtw_sol: list = []  # vrpspdtw的解
        self.veh_tasks: dict = {}  # 车头的任务
        self.veh_routes: dict = {}  # 车头的路径
        self.tank_tasks: dict = {}  # 罐箱的任务
        self.veh_tt: dict = {}  # 车头的时间表(timetable)
        self.veh_sol: dict | list | None = None  # 车头的解
        self.tank_sol: dict | list | None = None  # 罐箱的解
        self.veh_w_tank: dict | list | None = None  # 车头携带罐箱
        self.scheme: dict = {}  # 排程结果
        self.veh_tasks_map_tank_tasks = None  # 车头任务和罐箱任务的映射关系
        self.tasks_type = None  # 车头的任务类型 1-送满罐, 2-取空罐, 3-空车头
        self.ovrpdptw_prob: dict = {}  # ovrpdptw本体
        self.ovrpdptw_sol: list = []  # ovrpdptw的解
        self.ovrpdptw_tt: list = []  # ovrpdptw的时间表(timetable)
        self.veh_sol_status: int = 0  # 车头的解的状态 0-无解, 1-最优解, 2-可行解
        self.tank_sol_status: int = 0  # 罐箱的解的状态 0-无解, 1-最优解, 2-可行解
        self.max_time = 1440  # 车头每天的最大运行时间

    def create_vrpspdtw(self):
        """Create VRPSPDTW(Vehicle Routing Problem with Simultaneous Pickup-Delivery and Time Windows)."""
        # DEPOT_SECTION 热源ID(计算用) == [1]
        ds = [1]

        # NODE_COORD_SECTION 节点坐标 : dict[节点ID(计算用), (X坐标, Y坐标)]
        ncs = {1: self.gps[self.depot_id]}

        # PICKUP_AND_DELIVERY_SECTION 取送: list[list[节点ID, 0(没有用)，最早到达时间, 最晚到达时间, 服务时间, 取量，送量]]
        pds = [[1, 0, 0, 14400, 0, 0, 0]]

        # 映射字典: dict[计算ID, 真实ID]
        _cal_map_real = {1: self.depot_id}

        # 用户id(计算用)从2开始
        node_id = 2

        for demand_id in self.demand_ids:
            # gps
            lat, lng = self.gps[demand_id]

            # 热源到该点行驶时间
            t1j = self.cal_travel_time(ncs[1], (lat, lng))

            # 预测用热量(t)
            demand = self.pred[demand_id]['demand']

            # 送罐数 == 取罐数
            deliveries = math.ceil(demand / self.tank_avg_cap)
            if deliveries == 0:
                continue

            # 瞬时流量
            inst_flow = self.pred[demand_id]['inst_flow']

            # 需求时间线
            demand_timeline = get_demand_timeline(inst_flow, demand)

            # 同时取送
            for num in range(1, deliveries + 1):
                # 添加pds
                dead_line_idx = demand_timeline[demand_timeline['value'] <= (self.tank_avg_cap * num)].index
                if dead_line_idx.empty:
                    dead_time = t1j + int(self.service_time / 2)
                else:
                    dead_line = dead_line_idx[-1]
                    dead_time = int(dead_line.hour * 60 + dead_line.minute)
                due_time = max(dead_time - int(self.service_time / 2), t1j)
                ready_time = max(due_time - self.window, 0)
                pds.append([node_id, 0, ready_time, due_time, self.service_time, 1, 1])

                # 添加ncs
                ncs[node_id] = lat, lng

                # 添加ID映射
                _cal_map_real[node_id] = demand_id
                node_id += 1

        # weight_dict = {(i,j):dij/v}  dij = geodesic(i的坐标,j的坐标)  v = veh_speed
        weight_dict = {(i, j): self.cal_travel_time(ncs[i], ncs[j])
                       for j in range(1, len(ncs) + 1) for i in range(1, j + 1)}

        # EDGE_WEIGHT_SECTION
        ews = get_edge_weight_section(weight_dict)

        # keyword arguments
        params = {
            'name': 'prob',
            'type': 'VRPSPDTW',
            'dimension': len(ncs),
            'capacity': 1,
            'edge_weight_type': 'EXPLICIT',
            'edge_weight_format': 'LOWER_DIAG_ROW',
            'edge_weights': ews,
            'depots': ds,
            'pickup_and_delivery': pds,
        }

        # create vrpspdtw problem
        self.vrpspdtw_prob = lkh.LKHProblem(**params)
        self.cal_map_real = _cal_map_real

    def solve_vrpspdtw(self):
        """Solve VRPSPDTW using LKH."""
        vehicles = self.vrpspdtw_prob.dimension
        routes = []

        # 先尝试用vehicles = dimension求解, 得到vehicles的上限
        try:
            output = lkh.solve(self.lkh_solver_path, problem=self.vrpspdtw_prob, vehicles=vehicles, max_trials=600,
                               runs=5)
            routes = [x for x in output if x != []]
            vehicles = len(routes)
            # print('vehicles = dimension 有可行解, upper bound = {}'.format(vehicles))
        except IndexError:
            # print('WARNING! vehicles = dimension 无可行解!')
            pass

        # 再逐渐减少vehicles, 求解, 求出最小的vehicles
        while vehicles > 1:
            try:
                vehicles -= 1
                output = lkh.solve(self.lkh_solver_path, problem=self.vrpspdtw_prob, vehicles=vehicles, max_trials=600,
                                   runs=5)
                routes = [x for x in output if x != []]
                # print('vehicles = {} 有可行解'.format(len(routes)))
            except IndexError:
                # print('vehicles = {} 无可行解!!'.format(vehicles))
                break
        self.vrpspdtw_sol = routes

    def create_vrpspdtw_cp(self):
        """Create CP(Constraint Programming) Problem."""
        # 路径字典: dict[route_id, route]
        veh_routes_cal = {int(x + 1): self.vrpspdtw_sol[x] for x in range(len(self.vrpspdtw_sol))}

        # 时间窗字典
        window_dict = {x[0]: x[2:5] for x in self.vrpspdtw_prob.pickup_and_delivery}

        # 取送罐数字典
        pd_dict = {x[0]: x[5:7] for x in self.vrpspdtw_prob.pickup_and_delivery}

        # _veh_tt：按route_id存放路径时间表timetable
        _veh_tt = {}
        for route_id, route in veh_routes_cal.items():
            # 热源-用户-热源
            sec_node = route[0]

            # 第一个点按照最晚时间到达
            arr_time = window_dict[sec_node][1]
            service_time = window_dict[sec_node][2]
            leave_time = arr_time + service_time
            start_time = arr_time - self.vrpspdtw_prob.get_weight(0, sec_node - 1)
            timetable = [(start_time, start_time), (arr_time, leave_time)]
            leave_node = sec_node

            # 热源-用户1-用户2-...用户n-热源
            if len(route) > 1:
                for arr_node in route[1:]:
                    arr_time = leave_time + self.vrpspdtw_prob.get_weight(leave_node - 1, arr_node - 1)
                    ready_time = window_dict[arr_node][0]
                    service_time = + window_dict[arr_node][2]
                    leave_time = max(arr_time, ready_time) + service_time  # 早到必须等, 两者取大
                    timetable.append((arr_time, leave_time))
                    leave_node = arr_node

            # 2nd to last node -> end node
            end_time = leave_time + self.vrpspdtw_prob.get_weight(leave_node - 1, 0)
            timetable.append((end_time, end_time))
            _veh_tt[route_id] = timetable

        # 车的任务
        _veh_tasks = {}
        for route_id, timetable in _veh_tt.items():
            start_time = timetable[0][0]
            end_time = timetable[-1][-1]
            start_node = 0
            end_node = 0
            duration = end_time - start_time
            _veh_tasks[route_id] = [start_time, end_time, start_node, end_node, duration]

        # 车的路径(真实ID)
        _veh_routes = {k: [self.cal_map_real[x] for x in v] for k, v in veh_routes_cal.items()}

        # 所有到达用户的时间 dict[demand_id, list[tuple[arr_time,route_id,pd0]]]
        all_arr_times_raw = {}
        for route_id, route in _veh_routes.items():
            all_arr_times_raw.setdefault(route[0], []).append(
                (_veh_tt[route_id][1][0], route_id, pd_dict[veh_routes_cal[route_id][0]]))

        all_arr_times_sorted = {key: sorted(val, key=lambda x: x[0]) for key, val in all_arr_times_raw.items()}
        used_route_id = {key: [] for key in all_arr_times_sorted}

        # 罐任务, 车任务映射罐任务， 车任务类型
        _tank_tasks = {}
        _veh_tasks_map_tank_tasks = {x: [] for x in _veh_tasks}
        _tasks_type = {x: [] for x in _veh_tasks}
        count = 1
        for route_id, route in veh_routes_cal.items():
            sec_node = self.cal_map_real[route[0]]
            timetable = _veh_tt[route_id]
            pd0 = pd_dict[route[0]]
            start_time = timetable[0][0]
            arr_time = timetable[1][0]
            all_arr_times = all_arr_times_sorted[sec_node]

            # 罐任务结束时间: 下次运来罐的时间, 如果是当天最后一罐则默认运来时间+1000
            end_time = arr_time + 200
            if pd0 == [1, 1] or pd0 == [0, 1]:  # 同时取送, 或仅送罐
                # 罐的去程
                for item in all_arr_times:
                    if item[0] <= arr_time:
                        if arr_time == all_arr_times[-1][0]:
                            break
                    else:
                        if item[1] not in used_route_id[sec_node] and item[2] != [1, 0]:
                            end_time = item[0]
                            used_route_id[sec_node].append(item[1])
                            break
                _tank_tasks[count] = [start_time, end_time, 0, sec_node, end_time - start_time]
                _veh_tasks_map_tank_tasks[route_id].append(count)
                _tasks_type[route_id].append(1)  # 满罐被送达, task_type = 1
                count += 1

                # 罐的回程
                if pd0 == [1, 1]:
                    start_time = timetable[-2][1]
                    end_time = timetable[-1][0] + self.recharge_time
                    _tank_tasks[count] = [start_time, end_time, sec_node, 0, end_time - start_time]
                    _veh_tasks_map_tank_tasks[route_id].append(count)
                    _tasks_type[route_id].append(2)  # 空罐被送回, task_type = 2
                    count += 1
                elif len(route) == 1 and pd0 == [0, 1]:  # 仅送罐, 回程空车头
                    _tasks_type[route_id].append(3)
                elif len(route) == 2 and pd0 == [0, 1] and pd_dict[route[1]] == [1, 0]:  # 先送满罐去A, 后去B取空罐回热源
                    third_node = self.cal_map_real[route[1]]
                    start_time = timetable[-2][1]
                    end_time = timetable[-1][0] + self.recharge_time
                    _tank_tasks[count] = [start_time, end_time, third_node, 0, end_time - start_time]
                    _veh_tasks_map_tank_tasks[route_id].append(count)
                    _tasks_type[route_id].append(2)
                    count += 1

            elif pd0 == [1, 0]:  # 仅取罐, 去程空车头
                start_time = timetable[-2][1]
                end_time = timetable[-1][0] + self.recharge_time
                _tank_tasks[count] = [start_time, end_time, sec_node, 0, end_time - start_time]
                _veh_tasks_map_tank_tasks[route_id].append(count)
                _tasks_type[route_id] = [3, 2]
                count += 1

        self.veh_tasks = _veh_tasks
        self.veh_routes = _veh_routes
        self.veh_tt = _veh_tt
        self.tank_tasks = _tank_tasks
        self.veh_tasks_map_tank_tasks = _veh_tasks_map_tank_tasks
        self.tasks_type = _tasks_type

    def solve_vrpspdtw_cp(self):
        """Sole CP(Constraint Programming) Problem using designated workers."""
        # 重筛车罐ID,防止车罐数大于任务数造成的无解
        self.veh_ids = self.veh_ids[:len(self.veh_tasks)]
        self.tank_ids = self.tank_ids[:len(self.tank_tasks)]

        _veh_sol = assign_designated_workers(self.veh_tasks, len(self.veh_ids))
        _tank_sol = assign_designated_workers(self.tank_tasks, len(self.tank_ids))
        if _veh_sol is None:
            wrap_log('No solution for self.veh_tasks.', color='red')
            raise FourHundredError("No solution for current veh_ids, try larger veh_ids.")
        if _tank_sol is None:
            wrap_log('No solution for self.tank_tasks.', color='red')
            raise FourHundredError("No solution for current tank_ids, try larger tank_ids.")

        # 每个车头任务veh_task_id应携带哪个罐
        _veh_w_tank = {}
        if _veh_sol and _tank_sol:
            self.veh_sol = rearrange_sol(_veh_sol, self.veh_tasks)
            self.tank_sol = rearrange_sol(_tank_sol, self.tank_tasks)
            for veh_task_id in self.veh_tasks:
                tank_task_ids = self.veh_tasks_map_tank_tasks[veh_task_id]
                w_tank = []
                for target_id in tank_task_ids:
                    for key, sub_sol in self.tank_sol.items():
                        if target_id in sub_sol:
                            w_tank.append(key)
                            break
                _veh_w_tank[veh_task_id] = w_tank
            self.veh_w_tank = _veh_w_tank

    def create_ovrpdptw(self):
        """Create OVRPDPTW(Open Vehicle Routing Problem with Pickup and Delivery and Time Windows)."""
        # 热源信息
        demands = [0]
        locations = [self.gps[self.depot_id]]
        time_windows = [(0, 1429)]
        pickups_deliveries = []

        # 映射字典: dict[计算索引, 真实索引], 热源的计算和真实索引都为0
        _cal_map_real = {0: 0}

        # 用户id(计算用)从1开始
        node_id = 1
        for demand_id in self.demand_ids:
            # 用户gps
            lat, lng = self.gps.get(demand_id, (None, None))
            if not lat or not lng:
                continue

            # 预测用热量(t)
            demand = self.pred[demand_id]['demand']

            # 送罐数
            deliveries = math.ceil(demand / self.tank_avg_cap)
            if deliveries == 0:
                continue

            # 预测瞬时流量(t/h)
            inst_flow = self.pred[demand_id]['inst_flow']

            # 需求时间线
            demand_timeline = get_demand_timeline(inst_flow, demand)

            # 热源到用户的时间 + 服务时间
            t0j = self.cal_travel_time(locations[0], (lat, lng)) + self.service_time
            ready_time, final_time = extra_ready_time_and_final_time(demand_timeline)

            for num in range(1, deliveries + 1):
                dead_idx = demand_timeline[demand_timeline['value'] <= (self.tank_avg_cap * num)].index
                if dead_idx.empty:
                    due_time = t0j
                else:
                    due_time = max(t0j, int(dead_idx[-1].hour * 60 + dead_idx[-1].minute))
                due_time = min(due_time, max(final_time - t0j, 0))

                # 满罐, 取货点(热源) -> 送货点(用户)
                locations += [self.gps[self.depot_id], (lat, lng)]
                demands += [1, -1]
                time_windows += [(0, 1429), (ready_time, due_time)]
                pickups_deliveries.append([node_id, node_id + 1])
                _cal_map_real[node_id] = self.depot_id
                _cal_map_real[node_id + 1] = demand_id
                node_id += 2

                # 空罐, 取货点(用户) -> 送货点(热源)
                locations += [(lat, lng), self.gps[self.depot_id]]
                demands += [1, -1]
                time_windows += [(due_time, 1429), (0, 1429)]
                pickups_deliveries.append([node_id, node_id + 1])
                _cal_map_real[node_id] = demand_id
                _cal_map_real[node_id + 1] = self.depot_id
                node_id += 2

        # 距离矩阵
        distance_matrix = self.get_distance_matrix(locations)

        # 时间矩阵
        time_matrix = self.get_time_matrix(distance_matrix)

        # create ovrpdptw problem
        self.ovrpdptw_prob = {
            'depot': 0,  # 热源索引
            'num_vehicles': len(self.veh_ids),  # 车数
            'vehicle_capacity': 1,  # 车容量
            'locations': locations,  # 点位位置(GPS)
            'demands': demands,  # 点位需求
            'time_windows': time_windows,  # 点位时间窗
            'pickups_deliveries': pickups_deliveries,  # [[取货点, 送货点]]
            'distance_matrix': distance_matrix,  # 距离矩阵
            'time_matrix': time_matrix,  # 时间矩阵
        }
        self.cal_map_real = _cal_map_real

    def solve_ovrpdptw(self):
        """Solve OVRPDPTW."""
        data = self.ovrpdptw_prob

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # 创建模型
        routing = pywrapcp.RoutingModel(manager)

        # 行驶距离回调函数
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        distance_callback_index = routing.RegisterTransitCallback(distance_callback)

        # 行驶时间回调函数
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["time_matrix"][from_node][to_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        # 需求回调函数
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demand NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        # 空跑回调函数
        def run_free_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            map_from_node = self.cal_map_real.get(from_node)
            map_to_node = self.cal_map_real.get(to_node)
            if [from_node, to_node] in data["pickups_deliveries"]:
                return 0
            elif routing.IsStart(from_index) or routing.IsEnd(to_index):
                return 0
            else:
                if map_from_node == map_to_node:
                    return 0
                else:
                    return 10

        run_free_index = routing.RegisterTransitCallback(run_free_callback)

        # 添加"距离"维度
        routing.AddDimension(
            distance_callback_index,
            0,
            10 ** 9,
            True,
            "Distance",
        )
        distance_dimension = routing.GetDimensionOrDie("Distance")

        # 添加"时间"维度
        routing.AddDimension(
            time_callback_index,
            360,  # allow waiting time
            self.max_time,  # maximum time per vehicle
            False,  # Don't force start accumulation to zero.
            'Time',
        )
        time_dimension = routing.GetDimensionOrDie('Time')

        # 添加"容量"维度
        routing.AddDimension(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_capacity"],  # 约束: 每辆车的装货量不能超过最大容量
            True,  # start accumulation to zero
            "Capacity",
        )

        # 添加"空跑"维度
        routing.AddDimension(
            run_free_index,
            0,
            10 ** 5,  # 最大访问节点数
            True,
            "Run_Free",
        )
        run_free_dimension = routing.GetDimensionOrDie('Run_Free')

        # 设置弧的成本
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

        # 约束: 每一组Pickup&Delivery, 必须由同一辆车自提和配送
        # 约束: Pickup 必须发生在 Delivery 之前
        for request in data["pickups_deliveries"]:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
            routing.solver().Add(time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index))

        # 约束: Pickup和Delivery必须在节点的时间窗内
        for node, time_window in enumerate(data["time_windows"]):
            if node == data["depot"]:
                continue
            index = manager.NodeToIndex(node)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # 约束: 每辆车必须在起点的时间窗之内, 从起点出发
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data["time_windows"][data["depot"]][0], data["time_windows"][data["depot"]][1]
            )

        # 约束: 每辆车结束行驶时间尽量早
        for i in range(data["num_vehicles"]):
            # routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

        # 约束: 每辆车的行驶距离平均
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # 约束: 每辆车的工作时长(不同于行驶时长)平均
        time_dimension.SetGlobalSpanCostCoefficient(10)

        # 约束: 每辆车尽量不空跑
        for i in range(data["num_vehicles"]):
            routing.AddVariableTargetToFinalizer(run_free_dimension.CumulVar(routing.End(i)), 0)

        # 约束: 总车辆数尽可能少
        routing.SetFixedCostOfAllVehicles(10)

        # Allow to dropping nodes.
        penalty = 100000
        for node in range(1, len(data["distance_matrix"])):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        # search_parameters.first_solution_strategy = (
        #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(45)
        # search_parameters.log_search = True  # 启用求解日志
        # print(search_parameters)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        # print_solution(manager, routing, solution, self.cal_map_real)
        if solution:
            # print_solution(manager, routing, solution, self.cal_map_real)
            routes, timetables = get_routes_and_timetables(solution, routing, manager)
            # self.veh_sol_status = 1 if dropped_nodes == [] else 2
            self.ovrpdptw_sol = [[self.cal_map_real[i] for i in j[1:-1]] for j in routes if j[1:-1]]
            self.ovrpdptw_tt = [[i for i in j[1:-1]] for j in timetables if j[1:-1]]

    def create_ovrpdptw_cp(self):
        """Create CP(Constraint Programming) Problem."""
        # 车头路径: dict[route_id, route]
        _veh_routes = {route_id + 1: route for route_id, route in enumerate(self.ovrpdptw_sol)}

        # 车头时间表: dict[route_id, tt]
        _veh_tt = {route_id + 1: tt for route_id, tt in enumerate(self.ovrpdptw_tt)}

        # 所有到达用户的时间 dict[demand_id, list[tuple[arr_time,route_id]]]
        all_arr_times_raw = {}
        for route_id, route in _veh_routes.items():
            tt = _veh_tt[route_id]
            for node, arr_window in zip(route[1::2], tt[1::2]):
                all_arr_times_raw.setdefault(node, []).append((arr_window[1], route_id))

        # 所有到达用户的时间, 按到达时间重新排序
        all_arr_times_sorted = {node: sorted(val, key=lambda x: x[0]) for node, val in all_arr_times_raw.items()}

        count = 1
        _tank_tasks = {}
        used_arr_idx = {node: [] for node in all_arr_times_sorted}
        _veh_routes_map_tank_task = []
        for route_id, route in _veh_routes.items():
            tt = _veh_tt[route_id]
            single_veh_route_map_tank_task = []
            for start_node, end_node, leave_window, arr_window in zip(route[::2], route[1::2], tt[::2], tt[1::2]):
                # 送满罐, 热源 -> 用户
                if start_node == self.depot_id and end_node in self.demand_ids:
                    all_arr_times = all_arr_times_sorted[end_node]
                    start_time = leave_window[0]
                    arr_time = arr_window[1]
                    end_time = arr_time + 1000  # 最后一罐的用完时间 = 到达时间 + 1000
                    for idx, next_arr_tuple in enumerate(all_arr_times):
                        next_arr_time = next_arr_tuple[0]
                        if next_arr_time > arr_time and idx not in used_arr_idx[end_node]:
                            end_time = next_arr_time
                            used_arr_idx[end_node].append(idx)
                            break
                    _tank_tasks[count] = [start_time, end_time, start_node, end_node, end_time - start_time]
                    single_veh_route_map_tank_task += [count, count]
                    count += 1

                # 送空罐, 用户 -> 热源
                elif start_node in self.demand_ids and end_node == self.depot_id:
                    start_time = leave_window[0]
                    arr_time = arr_window[1]
                    end_time = arr_time + self.recharge_time
                    _tank_tasks[count] = [start_time, end_time, start_node, end_node, end_time - start_time]
                    single_veh_route_map_tank_task += [count, count]
                    count += 1
            _veh_routes_map_tank_task.append(single_veh_route_map_tank_task)

        self.veh_tt = self.ovrpdptw_tt
        self.tank_tasks = _tank_tasks
        self.veh_tasks_map_tank_tasks = _veh_routes_map_tank_task

    def solve_ovrpdptw_cp(self):
        """Sole CP(Constraint Programming) Problem using designated workers."""
        _veh_sol = self.ovrpdptw_sol
        _tank_sol = assign_designated_workers(self.tank_tasks, min(len(self.tank_ids), len(self.tank_tasks)))

        if not _veh_sol:
            wrap_log('No solution for self.veh_tasks.', color='red')
        if _tank_sol is None:
            wrap_log('No solution for self.tank_tasks.', color='red')

        if _veh_sol and _tank_sol:
            _veh_w_tank = []
            for i, veh_route in enumerate(_veh_sol):
                _w_tank = []
                for j, node in enumerate(veh_route):
                    tank_task_id = self.veh_tasks_map_tank_tasks[i][j]
                    tank_id = None
                    for k, v in _tank_sol.items():
                        if tank_task_id in v:
                            tank_id = k
                            break
                    _w_tank.append(tank_id)
                _veh_w_tank.append(_w_tank)

            assigned_tank_tasks_num = sum([len(i) for i in _tank_sol.values()])
            if assigned_tank_tasks_num == len(self.tank_tasks):
                self.tank_sol_status = 1
            elif 0 < assigned_tank_tasks_num < len(self.tank_tasks):
                self.tank_sol_status = 2
            self.veh_sol = _veh_sol
            self.tank_sol = list(_tank_sol.values())
            self.veh_w_tank = _veh_w_tank

    def get_scheme_vrpspdtw(self):
        """Get scheme using VRPSPDTW model."""
        _scheme = []

        # create vrpspdtw problem
        self.create_vrpspdtw()

        # solve vrpspdtw problem
        if self.vrpspdtw_prob:
            if self.vrpspdtw_prob.dimension < 3:
                wrap_log('DIMENSION < 3, VRPSPDTW is invalid', color='red')
            else:
                self.solve_vrpspdtw()

        if self.vrpspdtw_sol:
            # create cp problem
            self.create_vrpspdtw_cp()

            # solve cp problem
            self.solve_vrpspdtw_cp()

            if self.veh_sol and self.tank_sol:
                for veh_id, sub_sol in self.veh_sol.items():
                    sub = {"veh_id": self.veh_ids[veh_id]}
                    tasks = []
                    idx = 1
                    for task_id in sub_sol:
                        timetable = self.veh_tt[task_id]
                        task1 = {
                            "task_id": idx,
                            "task_type": 1,
                            "tank_id": self.tank_ids[self.veh_w_tank[task_id][0]],
                            "start_node": self.depot_id,
                            "start_time": encode_time_str(self.datestr, timetable[0][0]),
                            "end_node": self.veh_routes[task_id][0],
                            "end_time": encode_time_str(self.datestr, timetable[1][0]),
                        }
                        task2 = {
                            "task_id": idx + 1,
                            "task_type": 2,
                            "tank_id": self.tank_ids[self.veh_w_tank[task_id][-1]],
                            "start_node": self.veh_routes[task_id][-1],
                            "start_time": encode_time_str(self.datestr, timetable[-2][1]),
                            "end_node": self.depot_id,
                            "end_time": encode_time_str(self.datestr, timetable[-1][0]),
                        }

                        tasks.append(task1)
                        tasks.append(task2)
                        idx += 2
                    sub["tasks"] = tasks
                    _scheme.append(sub)
            else:
                wrap_log('No solution for CP Model in daily scheme.', color='red')
        else:
            wrap_log('No solution for VRPSPDTW in daily scheme.', color='red')

        self.scheme = {
            "date": self.datestr,
            "depot_id": self.depot_id,
            "demand_ids": self.demand_ids,
            "scheme": _scheme,
        }

    def get_scheme_ovrpdptw(self):
        """Get scheme using OVRPDPTW model."""
        _scheme = []
        _scheme_type = 0

        # create ovrpdptw problem
        self.create_ovrpdptw()

        # solve ovrpdptw problem
        if self.ovrpdptw_prob:
            self.solve_ovrpdptw()

        if self.ovrpdptw_sol:
            # create cp problem
            self.create_ovrpdptw_cp()

            # solve cp problem
            self.solve_ovrpdptw_cp()

            if self.veh_sol and self.tank_sol:
                for veh_id, route in enumerate(self.veh_sol):
                    sub = {"veh_id": self.veh_ids[veh_id]}
                    tt = self.veh_tt[veh_id]
                    w_tank = self.veh_w_tank[veh_id]
                    tasks = []
                    task_id = 1
                    for i in range(0, len(route), 2):
                        start_node, end_node = route[i], route[i + 1]
                        start_time = encode_time_str(self.datestr, tt[i][1])
                        end_time = encode_time_str(self.datestr, tt[i + 1][1])
                        start_w_tank = w_tank[i]
                        if start_w_tank is None:
                            continue
                        tank_id = self.tank_ids[start_w_tank]

                        # task_type
                        if start_node == self.depot_id and end_node in self.demand_ids:
                            task_type = 1
                        elif start_node in self.demand_ids and end_node == self.depot_id:
                            task_type = 2
                        else:
                            task_type = 3

                        task = {
                            "task_id": task_id,
                            "task_type": task_type,
                            "tank_id": tank_id,
                            "start_node": start_node,
                            "start_time": start_time,
                            "end_node": end_node,
                            "end_time": end_time,
                            "veh_color_id": veh_id + 1,
                        }
                        tasks.append(task)
                        task_id += 1
                    sub["tasks"] = tasks
                    _scheme.append(sub)
                # if self.veh_sol_status == 1 and self.tank_sol_status == 1:
                #     _scheme_type = 1
                # elif self.veh_sol_status == 2 or self.tank_sol_status == 2:
                #     _scheme_type = 2
                #     wrap_log('Daily scheme is feasible: ' + dumps(self.params), color='blue')
                _scheme_type = 1  # 只要有解就是1
            else:
                wrap_log('No solution for CP Model of daily scheme.', color='red')
        else:
            wrap_log('No solution for OVRPDPTW of daily scheme.', color='red')

        self.scheme = {
            "date": self.datestr,
            "depot_id": self.depot_id,
            "demand_ids": self.demand_ids,
            "scheme": _scheme,
            "scheme_type": _scheme_type,
        }

    def save(self, folder_path: str):
        """Save the result scheme to json in the folder path.

        Args:
            folder_path: The folder path where the result scheme is saved.

        Returns:
            None
        """
        if self.scheme:
            folder_path = folder_path + '/' + str(self.depot_id)
            os.makedirs(folder_path, exist_ok=True)
            with open(folder_path + '/' + self.datestr + '.json', 'w+') as f:
                dump(self.scheme, f)

    def cal_travel_distance(self, x_gps: tuple[float, float], y_gps: tuple[float, float]) -> int:
        """Calculate the traveling distance from x_gps to y_gps.

        Args:
            x_gps: (latitude, longitude)
            y_gps: (latitude, longitude)

        Returns:
            The traveling distance in meters.
        """
        return math.ceil(geodesic(x_gps, y_gps).m * self.cof)

    def cal_travel_time(self, x_gps, y_gps) -> int:
        """Calculate the traveling time from x_gps to y_gps.

        Args:
            x_gps: (latitude, longitude)
            y_gps: (latitude, longitude)

        Returns:
            The traveling time in minutes.
        """
        return math.ceil(self.cal_travel_distance(x_gps, y_gps) / 1000 / self.speed * 60)

    def get_distance_matrix(self, locations: list[tuple[float, float]]) -> dict[int, dict[int, int]]:
        """Return a distance matrix of locations.

        Args:
            locations: A list of (latitude, longitude).

        Returns:
            Distance matrix.
        """
        distance_matrix = {}
        for from_counter, from_node in enumerate(locations):
            distance_matrix[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distance_matrix[from_counter][to_counter] = 0
                elif from_counter == 0 or to_counter == 0:
                    distance_matrix[from_counter][to_counter] = 0
                else:
                    # 测地线距离
                    distance_matrix[from_counter][to_counter] = self.cal_travel_distance(from_node, to_node)
        return distance_matrix

    def get_time_matrix(self, distance_matrix: dict[int, dict[int, int]]) -> dict[int, dict[int, int]]:
        """Return a time matrix corresponding to a distance matrix.

        Args:
            distance_matrix: Distance matrix.

        Returns:
            Time matrix.
        """
        time_matrix = {}
        for i, dist_dict in distance_matrix.items():
            time_dict = {}
            for j, dij in dist_dict.items():
                if dij == 0:
                    tij = 0
                else:
                    tij = math.ceil(dij / 1000 / self.speed * 60) + self.service_time
                time_dict[j] = tij
            time_matrix[i] = time_dict
        return time_matrix


class DailyExecution(DailyScheme):
    """Daily Execution"""

    def __init__(self, data: dict):
        super(DailyExecution, self).__init__(data)
        self.tank_kafka: dict = {}  # 收到的有关self.tank_ids的Kafka message
        self.kafka_message: list | None = None  # 收到的所有Kafka message
        self.executed = {}  # 已执行任务
        self.daily_scheme_vehs = 0  # 计划排程用到的车头数
        self.max_time = 10 ** 7

    def get_kafka(self):
        """Get all kafka messages of specific date."""
        file_path = KAFKA_LOG_FOLDER + '/' + self.datestr + '.log'
        _kafka_message = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        _kafka_message.append(loads(line))
                    except JSONDecodeError as e:
                        wrap_log(f"存在无法解析的KAFKA数据：{e}", color='blue')
            self.kafka_message = _kafka_message
        except FileNotFoundError:
            wrap_log(f'KAFKA日志 {file_path} 不存在', color='red')

    def get_executed(self):
        """Get executed tasks from kafka messages."""
        # 只筛选属于本项目的kafka消息
        target_kafka_message = []
        target_dep_dem_ids = [str(x) for x in [self.depot_id, *self.demand_ids]]
        if self.kafka_message is None:
            return
        for message in self.kafka_message:
            data = message['data']
            for sub in data:
                if sub['subjectId'] in target_dep_dem_ids:
                    target_kafka_message.append(sub)

        # 解析该项目的kafka消息, 得到已执行的任务
        executed_list = []
        if target_kafka_message:
            # 筛选每个罐的kafka
            _tank_kafka = {}
            tank_ids = set([x['equipId'] for x in target_kafka_message])
            for i in tank_ids:
                message = []
                for msg in target_kafka_message:
                    if msg['equipId'] == i:
                        message.append(msg)
                _tank_kafka[i] = message
            self.tank_kafka = _tank_kafka

            # 'tasksStatus': '0'---初始化, '1'---待充热, '2'---充热中, '3'---充热结束
            #                '4'---配送中, '5'---待放热, '6'---放热中, '7'---放热完成
            for tank_id, message in _tank_kafka.items():
                if len(message) <= 1:
                    continue

                tmp3 = [0]
                tmp4 = []
                for i in range(len(message)):
                    tmp = message[i]['taskStatus']
                    try:
                        tmp2 = message[i + 1]['taskStatus']
                        if (tmp in ['1', '2', '3', '4'] and tmp2 in ['1', '2', '3', '4']) or (
                                tmp in ['5', '6', '7', '0'] and tmp2 in ['5', '6', '7', '0']):
                            if len(tmp3) <= 1:
                                tmp3.append(i + 1)
                            else:
                                tmp3[1] = i + 1
                        elif (tmp in ['1', '2', '3', '4'] and tmp2 in ['5', '6', '7', '0']) or (
                                tmp in ['5', '6', '7', '0'] and tmp2 in ['1', '2', '3', '4']):
                            tmp4.append(tmp3)
                            tmp3 = [i + 1]
                    except IndexError:
                        tmp4.append(tmp3)

                if len(tmp4) > 1:
                    for i in range(len(tmp4) - 1):
                        start_message = message[tmp4[i][-1]]
                        end_message = message[tmp4[i + 1][0]]

                        start_node = int(start_message['subjectId'])
                        end_node = int(end_message['subjectId'])

                        if start_node == end_node:
                            continue
                        tank_id = int(start_message['equipId'])

                        type_1_fieldnames = {
                            'enterDepotTime',
                            'chargeStartTime',
                            'chargeEndTime',
                            'departDepotTime',
                        }
                        type_2_fieldnames = {
                            'enterDemandTime',
                            'releaseStartTime',
                            'releaseEndTime',
                            'departDemandTime',
                        }

                        start_intersect_type1 = set(start_message.keys()) & type_1_fieldnames
                        start_intersect_type2 = set(start_message.keys()) & type_2_fieldnames
                        end_intersect_type1 = set(end_message.keys()) & type_1_fieldnames
                        end_intersect_type2 = set(end_message.keys()) & type_2_fieldnames

                        if start_intersect_type1 != set() and start_intersect_type2 == set() \
                                and end_intersect_type1 == set() and end_intersect_type2 != set():
                            task_type = 1
                            start_time = start_message[start_intersect_type1.pop()]
                            end_time = end_message[end_intersect_type2.pop()]
                        elif start_intersect_type1 == set() and start_intersect_type2 != set() \
                                and end_intersect_type1 != set() and end_intersect_type2 == set():
                            task_type = 2
                            start_time = start_message[start_intersect_type2.pop()]
                            end_time = end_message[end_intersect_type1.pop()]
                        else:
                            task_type = None
                            start_time = None
                            end_time = None

                        sub = {
                            'task_type': task_type,
                            'tank_id': tank_id,
                            'start_node': start_node,
                            'start_time': start_time,
                            'end_node': end_node,
                            'end_time': end_time,
                        }
                        executed_list.append(sub)

        if executed_list:
            _executed = {}
            for task in executed_list:
                if task['task_type'] == 1:
                    _executed.setdefault(task['end_node'], {'deliveries': [], 'pickups': []})['deliveries'].append(task)
                elif task['task_type'] == 2:
                    _executed.setdefault(task['start_node'], {'deliveries': [], 'pickups': []})['pickups'].append(task)
            self.executed = _executed

    def create_vrpspdtw(self):
        """Create VRPSPDTW(Vehicle Routing Problem with Simultaneous Pickup-Delivery and Time Windows)."""
        # DEPOT_SECTION 热源ID(计算用) == [1]
        ds = [1]

        # NODE_COORD_SECTION 节点坐标 : dict[节点ID(计算用), (X坐标, Y坐标)]
        ncs = {1: self.gps[self.depot_id]}

        # PICKUP_AND_DELIVERY_SECTION 取送: list[list[节点ID, 0(没有用)，最早到达时间, 最晚到达时间, 服务时间, 取量，送量]]
        pds = [[1, 0, 0, 14400, 0, 0, 0]]

        # 映射字典: dict[计算ID, 真实ID]
        _cal_map_real = {1: self.depot_id}

        # 用户id(计算用)从2开始
        node_id = 2

        # 现在时间
        current_time = get_current_time() + 1  # 考虑算法计算用时

        for demand_id in self.demand_ids:
            # gps
            lat, lng = self.gps[demand_id]

            # 热源到该点行驶时间
            t1j = self.cal_travel_time(ncs[1], (lat, lng))

            # 预测用热量(t)
            demand = self.pred[demand_id]['demand']

            # 送罐数 == 取罐数
            deliveries = math.ceil(demand / self.tank_avg_cap)
            pickups = deliveries
            if deliveries == 0:
                continue

            # 预测瞬时流量(t/h)
            inst_flow = self.pred[demand_id]['inst_flow']

            # 需求时间线
            demand_timeline = get_demand_timeline(inst_flow, demand)

            # 已完成(送, 取)
            if self.executed.get(demand_id):
                executed_deliveries = sorted(self.executed.get(demand_id)['deliveries'],
                                             key=lambda x: x.get('end_time'))
                executed_pickups = sorted(self.executed.get(demand_id)['pickups'],
                                          key=lambda x: x.get('start_time'))
            else:
                executed_deliveries = []
                executed_pickups = []

            # 送罐任务
            delta = 0
            for num in range(1, deliveries + 1):
                dead_line_idx = demand_timeline[demand_timeline['value'] <= (self.tank_avg_cap * num)].index
                if dead_line_idx.empty:
                    dead_time = t1j + int(self.service_time / 2)
                else:
                    dead_line = dead_line_idx[-1]
                    dead_time = int(dead_line.hour * 60 + dead_line.minute)

                if num <= len(executed_deliveries):
                    continue
                elif num == len(executed_deliveries) + 1:
                    if current_time > dead_time - int(self.service_time / 2):
                        travel_time = self.cal_travel_time(ncs[1], (lat, lng))
                        delta = current_time - dead_time + int(self.service_time / 2) + travel_time
                    else:
                        delta = 0

                dead_time += delta
                due_time = max(dead_time - int(self.service_time / 2), t1j)
                ready_time = max(due_time - self.window, 0)
                pds.append([node_id, 0, ready_time, due_time, int(self.service_time / 2), 0, 1])

                # 添加ncs
                ncs[node_id] = lat, lng

                # 添加ID映射
                _cal_map_real[node_id] = demand_id
                node_id += 1

            # 取罐任务
            delta = 0
            for num in range(1, pickups + 1):
                dead_line_idx = demand_timeline[demand_timeline['value'] <= (self.tank_avg_cap * num)].index
                if dead_line_idx.empty:
                    dead_time = t1j + int(self.service_time / 2)
                else:
                    dead_line = dead_line_idx[-1]
                    dead_time = int(dead_line.hour * 60 + dead_line.minute)

                if num <= len(executed_pickups):
                    continue
                elif num == len(executed_pickups) + 1:
                    delta = max(current_time - dead_time, 0)

                dead_time += delta
                ready_time = max(dead_time, 0)
                due_time = max(ready_time + self.window, t1j)
                pds.append([node_id, 0, ready_time, due_time, int(self.service_time / 2), 1, 0])

                # 添加ncs
                ncs[node_id] = lat, lng

                # 添加ID映射
                _cal_map_real[node_id] = demand_id
                node_id += 1

        weight_dict = {(i, j): self.cal_travel_time(ncs[i], ncs[j])
                       for j in range(1, len(ncs) + 1) for i in range(1, j + 1)}

        # EDGE_WEIGHT_SECTION
        ews = get_edge_weight_section(weight_dict)

        # keyword arguments
        params = {
            'name': 'prob',
            'type': 'VRPSPDTW',
            'dimension': len(ncs),
            'capacity': 1,
            'edge_weight_type': 'EXPLICIT',
            'edge_weight_format': 'LOWER_DIAG_ROW',
            'edge_weights': ews,
            'depots': ds,
            'pickup_and_delivery': pds,
        }

        # create vrpspdtw problem
        self.vrpspdtw_prob = lkh.LKHProblem(**params)
        self.cal_map_real = _cal_map_real

    def get_ava_veh_tank(self):
        occupied_tanks = []
        if self.tank_kafka:
            for tank_id, messages in self.tank_kafka.items():
                last_msg = messages[-1]
                if last_msg['subjectId'] != str(self.depot_id):
                    occupied_tanks.append(int(tank_id))

        # 当前能用的罐和车头
        # ava_tanks = [x for x in self.tank_ids if x not in occupied_tanks]
        ava_tanks = self.tank_ids
        ava_vehs = self.veh_ids

        self.tank_ids = ava_tanks
        self.veh_ids = ava_vehs

    def parse_executed(self):
        """Parse executed tasks. Compared with daily scheme. Return executed tasks."""
        # load daily scheme
        filepath = DAILY_SCHEME + '/' + str(self.depot_id) + '/' + self.datestr + '.json'
        try:
            with open(filepath, 'r') as f:
                _scheme = load(f)
        except FileNotFoundError:
            wrap_log(f'当日计划排程不存在: {filepath}, 将采用当前传参自动生成当日计划排程: {dumps(self.params)}')
            daily_scheme(**self.params)
            wrap_log(f'当日计划排程生成完毕: {filepath}')
            with open(filepath, 'r') as f:
                _scheme = load(f)

        # 获取计划排程的车头数
        self.daily_scheme_vehs = len(_scheme['scheme'])

        # 计划的罐的任务
        sch_tank_tasks = {}
        for sub_scheme in _scheme['scheme']:
            veh_id = sub_scheme['veh_id']
            for task in sub_scheme['tasks']:
                if task['tank_id'] in self.tank_ids:
                    task['veh_id'] = veh_id
                    sch_tank_tasks.setdefault(task['tank_id'], []).append(task)

        # 实际执行的罐的任务
        exe_tank_tasks = {}
        if self.executed:
            for demand_id, sub_executed in self.executed.items():
                exe_combo = sub_executed['deliveries'] + sub_executed['pickups']
                for task in exe_combo:
                    if task['tank_id'] in self.tank_ids:
                        exe_tank_tasks.setdefault(task['tank_id'], []).append(task)

        # 解析实际执行任务, 如果这个罐被送到用户点, 则认为是该车送的, 该车完成了1个任务
        # 如果没有找到车头ID, 现阶段不传给前端, 即使该任务已完成
        parse_tasks = {}
        for tank_id in exe_tank_tasks:
            exe_tasks = sorted(exe_tank_tasks[tank_id], key=lambda x: x['start_time'])
            sch_tasks = sorted(sch_tank_tasks.get(tank_id), key=lambda x: x['start_time']).copy() if sch_tank_tasks.get(
                tank_id) else []
            if sch_tasks:
                for exe in exe_tasks:
                    veh_id = None
                    for sch in sch_tasks:
                        if exe['start_node'] == sch['start_node'] and exe['end_node'] == exe['end_node']:
                            veh_id = sch['veh_id']
                            parse_tasks.setdefault(veh_id, []).append(exe)
                            sch_tasks.remove(sch)
                            break
                    if not veh_id:
                        pass

        result = []
        if parse_tasks:
            for veh_id, tasks in parse_tasks.items():
                sub = {'veh_id': veh_id}
                idx = 1
                tmp = []
                for task in sorted(tasks, key=lambda x: x['start_time']):
                    try:
                        veh_color_id = self.veh_ids.index(veh_id) + 1
                    except ValueError:
                        continue
                    task['task_id'] = idx
                    task['veh_color_id'] = veh_color_id
                    tmp.append(task)
                    idx += 1
                sub['tasks'] = tmp
                result.append(sub)
        return result

    def get_scheme_vrpspdtw(self):
        """Get scheme using VRPSPDTW model."""
        self.get_kafka()
        self.get_executed()
        self.scheme = {
            "date": self.datestr,
            "depot_id": self.depot_id,
            "demand_ids": self.demand_ids,
            "executed": self.parse_executed(),
            "new_scheme": [],
        }

        # create vrpspdtw problem
        self.create_vrpspdtw()

        if self.vrpspdtw_prob:
            if self.vrpspdtw_prob.dimension < 3:
                wrap_log('DIMENSION < 3, VRPSPDTW无效', color='red')
            else:
                self.solve_vrpspdtw()

        if self.vrpspdtw_sol:
            # create cp problem
            self.create_vrpspdtw_cp()

            # solve cp problem
            self.solve_vrpspdtw_cp()

        if self.veh_sol and self.tank_sol:
            _new_scheme = []
            for veh_id, sub_sol in self.veh_sol.items():
                sub = {"veh_id": self.veh_ids[veh_id]}
                tasks = []
                idx = 1
                for task_id in sub_sol:
                    timetable = self.veh_tt[task_id]
                    task1_type, task2_type = self.tasks_type[task_id]
                    route = self.veh_routes[task_id]

                    # 仅送罐,回程空车头
                    if len(route) == 1 and task1_type == 1 and task2_type == 3:
                        task1_tank_id = self.tank_ids[self.veh_w_tank[task_id][0]]
                        task2_tank_id = None
                    # 仅取罐, 去程空车头
                    elif len(route) == 1 and task1_type == 3 and task2_type == 2:
                        task1_tank_id = None
                        task2_tank_id = self.tank_ids[self.veh_w_tank[task_id][0]]
                    # 取送罐
                    else:
                        task1_tank_id = self.tank_ids[self.veh_w_tank[task_id][0]]
                        task2_tank_id = self.tank_ids[self.veh_w_tank[task_id][-1]]

                    # 去程任务
                    task1 = {
                        "task_id": idx,
                        "task_type": task1_type,
                        "tank_id": task1_tank_id,
                        "start_node": self.depot_id,
                        "start_time": encode_time_str(self.datestr, timetable[0][0]),
                        "end_node": route[0],
                        "end_time": encode_time_str(self.datestr, timetable[1][0]),
                    }
                    tasks.append(task1)
                    idx += 1

                    # 若存在先取用户A送罐, 后去用户B取罐, 再返回热源的情况
                    if len(route) == 2 and route[0] != route[1]:
                        ex_task = {
                            "task_id": idx,
                            "task_type": 3,
                            "tank_id": None,
                            "start_node": route[0],
                            "start_time": encode_time_str(self.datestr, timetable[1][1]),
                            "end_node": route[1],
                            "end_time": encode_time_str(self.datestr, timetable[2][0]),
                        }
                        tasks.append(ex_task)
                        idx += 1

                    # 回程任务
                    task2 = {
                        "task_id": idx,
                        "task_type": task2_type,
                        "tank_id": task2_tank_id,
                        "start_node": route[-1],
                        "start_time": encode_time_str(self.datestr, timetable[-2][1]),
                        "end_node": self.depot_id,
                        "end_time": encode_time_str(self.datestr, timetable[-1][0]),
                    }
                    tasks.append(task2)
                    idx += 1
                sub["tasks"] = tasks
                _new_scheme.append(sub)
            self.scheme["new_scheme"] = _new_scheme

    def create_ovrpdptw(self):
        """Create OVRPDPTW(Open Vehicle Routing Problem with Pickup and Delivery and Time Windows)."""
        # 热源信息
        demands = [0]
        locations = [self.gps[self.depot_id]]
        time_windows = [(0, 10 ** 7)]
        pickups_deliveries = []

        # 映射字典: dict[计算索引, 真实索引], 热源的计算和真实索引都为0
        _cal_map_real = {0: 0}

        # 用户id(计算用)从1开始
        node_id = 1

        # 当前时间
        current_time = get_current_time() + 1  # 加1考虑算法计算时间
        for demand_id in self.demand_ids:
            # 用户gps
            lat, lng = self.gps.get(demand_id, (None, None))
            if not lat or not lng:
                continue

            # 预测用热量(t)
            demand = self.pred[demand_id]['demand']

            # 送罐数
            deliveries = math.ceil(demand / self.tank_avg_cap)
            pickups = deliveries
            if deliveries == 0:
                continue

            # 预测瞬时流量(t/h)
            inst_flow = self.pred[demand_id]['inst_flow']

            # 需求时间线
            demand_timeline = get_demand_timeline(inst_flow, demand)

            # 热源到用户的时间 + 服务时间
            t0j = self.cal_travel_time(locations[0], (lat, lng)) + self.service_time
            # ready_time, final_time = extra_ready_time_and_final_time(demand_timeline)

            # 已完成(送, 取)
            if self.executed.get(demand_id):
                executed_deliveries = sorted(self.executed.get(demand_id)['deliveries'],
                                             key=lambda x: x.get('end_time'))
                executed_pickups = sorted(self.executed.get(demand_id)['pickups'],
                                          key=lambda x: x.get('start_time'))
            else:
                executed_deliveries = []
                executed_pickups = []

            delta = 0
            for num in range(1, deliveries + 1):
                dead_idx = demand_timeline[demand_timeline['value'] <= (self.tank_avg_cap * num)].index
                if dead_idx.empty:
                    due_time = t0j
                else:
                    due_time = max(t0j, int(dead_idx[-1].hour * 60 + dead_idx[-1].minute))

                if num <= len(executed_deliveries):
                    continue
                elif num == len(executed_deliveries) + 1:
                    if current_time > due_time:
                        delta = delta + current_time - due_time

                due_time = min(due_time + delta, max(1429 + delta - t0j, 0))
                # ready_time = max(current_time + t0j, ready_time)
                ready_time = current_time + t0j
                due_time = max(due_time, ready_time)

                # 满罐, 取货点(热源) -> 送货点(用户)
                locations += [self.gps[self.depot_id], (lat, lng)]
                demands += [1, -1]
                time_windows += [(0, 10 ** 7), (ready_time, due_time)]
                pickups_deliveries.append([node_id, node_id + 1])
                _cal_map_real[node_id] = self.depot_id
                _cal_map_real[node_id + 1] = demand_id
                node_id += 2

            delta = 0
            for num in range(1, pickups + 1):
                dead_idx = demand_timeline[demand_timeline['value'] <= (self.tank_avg_cap * num)].index
                if dead_idx.empty:
                    due_time = t0j
                else:
                    due_time = max(t0j, int(dead_idx[-1].hour * 60 + dead_idx[-1].minute))

                if num <= len(executed_pickups):
                    continue
                elif num == len(executed_pickups) + 1:
                    if current_time > due_time:
                        delta = delta + current_time - due_time

                due_time = min(due_time + delta, max(1429 + delta - t0j, 0))
                ready_time = current_time + t0j
                due_time = max(due_time, ready_time)

                # 空罐, 取货点(用户) -> 送货点(热源)
                locations += [(lat, lng), self.gps[self.depot_id]]
                demands += [1, -1]
                time_windows += [(due_time, 10 ** 7), (0, 1429 + delta)]
                pickups_deliveries.append([node_id, node_id + 1])
                _cal_map_real[node_id] = demand_id
                _cal_map_real[node_id + 1] = self.depot_id
                node_id += 2

        # 距离矩阵
        distance_matrix = self.get_distance_matrix(locations)

        # 时间矩阵
        time_matrix = self.get_time_matrix(distance_matrix)

        num_vehicles = self.daily_scheme_vehs if self.daily_scheme_vehs else len(self.veh_ids)

        # create ovrpdptw problem
        self.ovrpdptw_prob = {
            'depot': 0,  # 热源索引
            'num_vehicles': num_vehicles,  # len(self.veh_ids),  # 车数
            'vehicle_capacity': 1,  # 车容量
            'locations': locations,  # 点位位置(GPS)
            'demands': demands,  # 点位需求
            'time_windows': time_windows,  # 点位时间窗
            'pickups_deliveries': pickups_deliveries,  # [[取货点, 送货点]]
            'distance_matrix': distance_matrix,  # 距离矩阵
            'time_matrix': time_matrix,  # 时间矩阵
        }
        self.cal_map_real = _cal_map_real

    def get_scheme_ovrpdptw(self):
        """Get scheme using OVRPDPTW model."""
        self.get_kafka()
        self.get_executed()
        self.scheme = {
            "date": self.datestr,
            "depot_id": self.depot_id,
            "demand_ids": self.demand_ids,
            "executed": self.parse_executed(),
            "new_scheme": [],
            "scheme_type": 0,
        }
        _scheme_type = 0

        # create ovrpdptw problem
        self.create_ovrpdptw()

        # solve ovrpdptw problem
        if self.ovrpdptw_prob:
            self.solve_ovrpdptw()

        if self.ovrpdptw_sol:
            # create cp problem
            self.create_ovrpdptw_cp()

            # solve cp problem
            self.solve_ovrpdptw_cp()

            if self.veh_sol and self.tank_sol:
                _new_scheme = []
                for veh_id, route in enumerate(self.veh_sol):
                    sub = {"veh_id": self.veh_ids[veh_id]}
                    tt = self.veh_tt[veh_id]
                    w_tank = self.veh_w_tank[veh_id]
                    tasks = []
                    for i in range(0, len(route), 2):
                        start_node, end_node = route[i], route[i + 1]
                        start_time = encode_time_str(self.datestr, tt[i][1])
                        end_time = encode_time_str(self.datestr, tt[i + 1][1])
                        start_w_tank = w_tank[i]
                        if start_w_tank is None:
                            continue
                        tank_id = self.tank_ids[start_w_tank]

                        # task_type
                        if start_node == self.depot_id and end_node in self.demand_ids:
                            task_type = 1
                        elif start_node in self.demand_ids and end_node == self.depot_id:
                            task_type = 2
                        else:
                            task_type = 3

                        task = {
                            "task_id": i // 2 + 1,
                            "task_type": task_type,
                            "tank_id": tank_id,
                            "start_node": start_node,
                            "start_time": start_time,
                            "end_node": end_node,
                            "end_time": end_time,
                            "veh_color_id": veh_id + 1,
                        }
                        tasks.append(task)
                    sub["tasks"] = tasks
                    _new_scheme.append(sub)
                self.scheme["new_scheme"] = _new_scheme
                # if self.veh_sol_status == 1 and self.tank_sol_status == 1:
                #     _scheme_type = 1
                # elif self.veh_sol_status == 2 or self.tank_sol_status == 2:
                #     _scheme_type = 2
                #     wrap_log('Daily execution is feasible: ' + dumps(self.params), color='blue')
                self.scheme["scheme_type"] = 1
            else:
                wrap_log('No solution for CP Model of daily execution.', color='red')
        else:
            wrap_log('No solution for OVRPDPTW of daily execution.', color='red')


def daily_scheme(date: str | None = None, depot_id: str | None = None, demand_ids: list | None = None,
                 veh_ids: list | None = None, tank_ids: list | None = None, save: bool = True) -> dict | None:
    """Return daily scheme.

    Args:
        date: Date string as "%Y-%m-%d".
        depot_id: Depot id.
        demand_ids: A list of demand id.
        veh_ids: A list of vehicle id.
        tank_ids: A list of tank id.
        save: Whether to save the scheme to json.

    Returns:
        The daily scheme.

    Examples:
        date = "2024-01-19",
        depot_id = "200123081370",
        demand_ids = [
            "200123051356",
            "200123011219",
            "200123041336"
            ],
        veh_ids = [
            1687376539856932900,
            1697518660637765600,
            1669550179402911700,
            1686177246441640000,
            1686177245741191200,
            1686177245741191300,
            1686177245741191200
        ],
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
        scheme = daily_scheme(date, depot_id, demand_ids, veh_ids, tank_ids)
        print(scheme)
    """
    # 检查热源与用户是否有效
    check_node(depot_id, demand_ids)

    data = {
        "date": date,
        "depot_id": depot_id,
        "demand_ids": demand_ids,
        "veh_ids": veh_ids,
        "tank_ids": tank_ids,
        "gps": get_gps_dict([depot_id, *demand_ids]),
        "pred": {
            demand_id: {
                "demand": get_supplies_and_demands_pred(demand_id, date),
                "inst_flow": get_inst_flows_pred(demand_id, date),
            } for demand_id in demand_ids
        },
    }

    _scheme = DailyScheme(data)

    # get daily scheme
    _scheme.get_scheme_ovrpdptw()

    # save
    if save:
        _scheme.save(DAILY_SCHEME)

    # 暂接口不返回task_type不为1或2或tank_id=None的任务
    result = filter_scheme(_scheme.scheme)
    return result


def daily_execution(date: str | None = None, depot_id: int | None = None, demand_ids: list | None = None,
                    veh_ids: list | None = None, tank_ids: list | None = None, save: bool = True) -> dict | None:
    """Return daily execution.

    Args:
        date: Date string as "%Y-%m-%d".
        depot_id: Depot id.
        demand_ids: A list of demand id.
        veh_ids: A list of vehicle id.
        tank_ids: A list of tank id.
        save: Whether to save the scheme to json.

    Returns:
        The daily execution.

    Examples:
        date = "2024-01-19",
        depot_id = "200123081370",
        demand_ids = [
            "200123051356",
            "200123011219",
            "200123041336"
            ],
        veh_ids = [
            1687376539856932900,
            1697518660637765600,
            1669550179402911700,
            1686177246441640000,
            1686177245741191200,
            1686177245741191300,
            1686177245741191200
        ],
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
        scheme = daily_execution(date, depot_id, demand_ids, veh_ids, tank_ids)
        print(scheme)
    """
    # 检查热源与用户是否有效
    check_node(depot_id, demand_ids)

    data = {
        "date": date,
        "depot_id": depot_id,
        "demand_ids": demand_ids,
        "veh_ids": veh_ids,
        "tank_ids": tank_ids,
        "pred": {
            demand_id: {
                "demand": get_supplies_and_demands_pred(demand_id, date),
                "inst_flow": get_inst_flows_pred(demand_id, date),
            } for demand_id in demand_ids
        },
        "gps": get_gps_dict([depot_id, *demand_ids]),
    }

    _scheme = DailyExecution(data)

    # get daily scheme
    _scheme.get_scheme_ovrpdptw()

    # save
    if save:
        _scheme.save(DAILY_EXECUTION)

    # 暂不返回task_type不为1或2或tank_id=None的任务
    result = filter_scheme(_scheme.scheme, field='new_scheme')
    return result


def assign_designated_workers(tasks_dict: dict[int, list], num_workers: int) -> dict | None:
    """Return solution of assignment cp model with designated number of workers. Workers need to execute a set of tasks.
    And each worker should be assigned to least one task.

    Args:
        tasks_dict: A dictionary of tasks. The key is a task_id starting from 1. The value is a task as
                    [start_time[int], end_time[int], start_node,end_node, duration[int]].
        num_workers: The number of workers.

    Returns:
        The solution of model. The key is a worker_id starting from 0. The value is task_id.

    Examples:
        tasks_dict = {
            1: [49, 96, 0, 1, 47], 2: [96, 143, 1, 0, 47], 3: [145, 192, 0, 1, 47], 4: [192, 239, 1, 0, 47],
            5: [241, 288, 0, 1, 47], 6: [288, 335, 1, 0, 47], 7: [337, 384, 0, 1, 47], 8: [384, 431, 1, 0, 47],
            9: [433, 480, 0, 1, 47], 10: [480, 527, 1, 0, 47], 11: [529, 576, 0, 1, 47], 12: [576, 623, 1, 0, 47],
            13: [625, 672, 0, 1, 47], 14: [672, 719, 1, 0, 47], 15: [721, 768, 0, 1, 47], 16: [768, 815, 1, 0, 47],
            17: [817, 864, 0, 1, 47], 18: [864, 911, 1, 0, 47], 19: [913, 960, 0, 1, 47], 20: [960, 1007, 1, 0, 47],
            21: [1009, 1056, 0, 1, 47], 22: [1056, 1103, 1, 0, 47], 23: [1105, 1152, 0, 1, 47],
            24: [1152, 1199, 1, 0, 47], 25: [1201, 1248, 0, 1, 47], 26: [1248, 1295, 1, 0, 47],
            27: [1297, 1344, 0, 1, 47], 28: [1344, 1391, 1, 0, 47], 29: [1393, 1440, 0, 1, 47],
            30: [1440, 1487, 1, 0, 47],
        }
        num_workers = 5
        assign_designated_workers(tasks_dict, num_workers)

        Returns: {
            0: [1, 2, 7, 14, 21, 30],
            1: [11, 12, 15, 16, 27, 28],
            2: [3, 6, 9, 10, 13, 22],
            3: [4, 19, 20, 23, 24, 29],
            4: [5, 8, 17, 18, 25, 26]
        }
    """
    # 任务数
    num_tasks = len(tasks_dict)

    # 成本字典
    costs = {}
    for k in range(num_workers):
        costs[0, num_tasks + 1, k] = 0
        for i, task_i in tasks_dict.items():
            costs[0, i, k] = 0
            costs[i, num_tasks + 1, k] = task_i[-1]
            for j, task_j in tasks_dict.items():
                if j != i and task_i[1] <= task_j[0] <= task_i[1] + 1440 and task_i[3] == task_j[2]:
                    costs[i, j, k] = task_i[4]

    # 创建CP-SAT模型
    m = cp_model.CpModel()

    # 决策变量 x, y
    x = tupledict({(i, j, k): m.NewBoolVar(f'x[{i}, {j}, {k}]') for i, j, k in costs})
    y = tupledict({k: m.NewBoolVar(f'y[{k}]') for k in range(num_workers)})

    # 整数变量 u, v
    up_bound = int(max(x[1] for x in tasks_dict.values()) - min(x[0] for x in tasks_dict.values()))
    u = m.NewIntVar(0, up_bound, 'u')
    v = m.NewIntVar(0, up_bound, 'v')

    # 约束1: 每个任务最多被执行一次
    for task_id in tasks_dict:
        # m.AddExactlyOne(x.select(task_id, '*', '*'))  # 每个任务都被执行, 且只执行一次
        m.AddAtMostOne(x.select(task_id, '*', '*'))  # 每个任务最多被执行一次

    # 约束2: 生产块是否连接, 通过构造costs实现
    # 约束3: 班组是否参加生产, 保证在即y[k] = 0时, x[i] = 0
    for i, j, k in x:
        if i != 0 or j != num_tasks + 1:
            m.Add(x[i, j, k] <= y[k])

    # 约束4: 流平衡约束
    for worker in range(num_workers):
        m.AddExactlyOne(x.select(0, '*', worker))
        m.AddExactlyOne(x.select('*', num_tasks + 1, worker))
        for task_id in tasks_dict:
            m.Add(sum(x.select(task_id, '*', worker)) == sum(x.select('*', task_id, worker)))

    # 约束5: 最长工作时间约束
    for worker in range(num_workers):
        m.Add(sum(costs[i, j, k] * x[i, j, k] for i, j, k in x if k == worker) <= u)

    # 约束6: 最短工作时间约束
    M = 10 ** 8
    for worker in range(num_workers):
        m.Add(v <= (1 - y[worker]) * M + sum(costs[i, j, k] * x[i, j, k] for i, j, k in x if k == worker))

    # 约束7: 所有班组都要参与工作
    for worker in range(num_workers):
        m.AddAtLeastOne(x[i, j, k] for i, j, k in x if j != 0 and j != num_tasks + 1 and k == worker)  # 所有班组都参加工作

    # 目标函数
    alpha = 2
    beta = 1
    m.Maximize(alpha * sum(x.values()) + beta * (u - v))  # 最大化任务完成数

    # 设置求解器参数
    solver = cp_model.CpSolver()

    # 启用日志输出
    # solver.parameters.log_search_progress = True

    # 设置求解时间限制
    solver.parameters.max_time_in_seconds = 40

    # 求解
    status = solver.Solve(m)

    # 提取结果
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # print(f"ObjectiveValue = {solver.ObjectiveValue()}")
        # print(solver.StatusName(status), 'u:', solver.Value(u), 'v:', solver.Value(v))

        bestX_tl = tuplelist([
            (i, j, k) for i, j, k in x if solver.BooleanValue(x[i, j, k]) and (i != 0 or j != num_tasks + 1)
        ])
        bestY = {k: solver.Value(y[k]) for k in y if solver.BooleanValue(y[k])}
        tmp_sol2 = {}
        for k in bestY:
            val_tl = bestX_tl.select('*', '*', k)
            start = 0
            tmp_sol = []
            while start != num_tasks + 1:
                tmp = val_tl.select(start, '*')
                tmp_sol += tmp
                start = tmp[0][1]
            tmp_sol2[k] = tmp_sol
        # 解
        result = {key: [x[0] for x in val[1:]] for key, val in tmp_sol2.items()}
        return result
    else:
        # 无解
        return None


def get_edge_weight_section(weight_dict: dict) -> list:
    """Get EDGE_WEIGHT_SECTION from weight_dict.

    Args:
        weight_dict: Weight dictionary.

    Returns:
        Edge weight section.
    """
    length = len(weight_dict)
    weight_key = list(weight_dict)
    col = math.floor(math.sqrt(length))
    row, rem = divmod(length, col)
    if rem > 0:
        ews_key_list = [weight_key[col * i:col * i + col] for i in range(row)] + [weight_key[-rem:]]
    else:
        ews_key_list = [weight_key[col * i:col * i + col] for i in range(row)]
    result = [[weight_dict[i] for i in x] for x in ews_key_list]
    return result


def encode_time_str(date_str: str, minute: int) -> str:
    """Encode date_str with minute. Return datetime string as '%Y-%m-%d %H:%M:%S'.

    Args:
        date_str: Date string as "%Y-%m-%d".
        minute: An integer means minutes.

    Returns:
        Datetime string as '%Y-%m-%d %H:%M:%S'.
    """
    return (pd.Timedelta(minute, unit='minute') + pd.Timestamp(date_str)).strftime('%Y-%m-%d %H:%M:%S')


def rearrange_sol(old: dict, tasks: dict) -> dict:
    """Rearrange an old solution and return a new solution in order of task time.

    Args:
        old: The old solution.
        tasks: The tasks.

    Returns:
        The new solution.
    """
    # 按照任务时间先后重排sol
    new_key = sorted(old.keys())
    new_val = sorted(old.values(), key=lambda x: tasks[x[0]][:2])
    new = {x: y for x, y in zip(new_key, new_val)}
    return new


def filter_scheme(old: dict, field: str = 'scheme', remove_tomorrow: bool = True) -> dict:
    """Remove the tasks in scheme whose task type isn't in {1, 2} or tank id is None.

    Args:
        old: The old scheme.
        field: The field name in the scheme.
        remove_tomorrow: Whether to remove tomorrow tasks.

    Returns:
        The filtered scheme.
    """
    new = deepcopy(old)
    datestr = old['date']
    new_scheme = new[field]
    for sub_scheme in new_scheme:
        old_tasks = sub_scheme['tasks']
        new_tasks = {'tasks': []}
        i = 1
        for task in old_tasks:
            if task['task_type'] in {1, 2} and task['tank_id']:
                if remove_tomorrow:
                    if task['end_time'].split(' ')[0] == datestr:
                        task['task_id'] = i
                        new_tasks['tasks'].append(task)
                        i += 1
                else:
                    task['task_id'] = i
                    new_tasks['tasks'].append(task)
                    i += 1
        sub_scheme.update(new_tasks)
    new[field] = [x for x in new[field] if x['tasks']]
    return new


def get_demand_timeline(inst_flow: pd.DataFrame, demand: float) -> pd.DataFrame:
    """Get the timeline based on both instant flow and demand. Return a timeline DataFrame.

    Args:
        inst_flow: Instant flow DataFrame.
        demand: The value of demand.

    Returns:
        The timeline DataFrame.
    """
    dropped_df = inst_flow.replace(0, np.nan).dropna()
    if inst_flow.empty or dropped_df.empty:
        timeline_df = pd.DataFrame(
            data=[demand * x / 1440 for x in range(1440)],
            index=pd.date_range(datetime.now().strftime("%Y-%m-%d"), periods=1440, freq='1min'),
            columns=['value'],
        )
    else:
        last_ser = inst_flow.iloc[-1]
        last_df = last_ser.rename(last_ser.name + pd.Timedelta(hours=1)).to_frame().T
        concat_df = pd.concat([inst_flow, last_df])
        interpolated_df = concat_df.resample('1T').interpolate(method='linear')
        drop_last_df = interpolated_df.drop(interpolated_df.index[-1])
        timeline_df = drop_last_df.cumsum() / drop_last_df.sum() * demand
    return timeline_df


def get_routes_and_timetables(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array whose i,j entry is the jth location visited by
    vehicle i along its route.
    """
    routes = []
    timetables = []
    time_dimension = routing.GetDimensionOrDie("Time")
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        time_var = time_dimension.CumulVar(index)
        timetable = [(solution.Min(time_var), solution.Max(time_var))]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            time_var = time_dimension.CumulVar(index)
            timetable.append((solution.Min(time_var), solution.Max(time_var)))
        routes.append(route)
        timetables.append(timetable)
    return routes, timetables


def print_solution(manager, routing, solution, cal_map_real: dict):
    """Prints solution on the console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    # Display dropped nodes.
    dropped_nodes = "Dropped nodes:"
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue

        if solution.Value(routing.NextVar(node)) == node:
            dropped_nodes += f" {cal_map_real[manager.IndexToNode(node)]}"
    print(dropped_nodes)

    time_dimension = routing.GetDimensionOrDie("Time")
    run_free_dimension = routing.GetDimensionOrDie('Run_Free')
    total_time = 0
    max_route_distance = 0
    for vehicle_id in range(routing.vehicles()):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += (
                f"{cal_map_real[manager.IndexToNode(index)]}"
                # f"{manager.IndexToNode(index)}"
                f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                f" Distance({route_distance})"
                " -> "
            )
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        time_var = time_dimension.CumulVar(index)
        plan_output += (
            f"{cal_map_real[manager.IndexToNode(index)]}"
            # f"{manager.IndexToNode(index)}"
            f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
            f" Distance({route_distance})\n"
        )
        plan_output += f"Time of the route: {solution.Min(time_var)}min\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Run Free: {solution.Value(run_free_dimension.CumulVar(index))}\n"
        print(plan_output)
        total_time += solution.Min(time_var)
        max_route_distance = max(route_distance, max_route_distance)
    print(f"Total time of all routes: {total_time}min")
    print(f"Maximum of the route distances: {max_route_distance}m")


def extra_ready_time_and_final_time(df: pd.DataFrame) -> tuple[int, int]:
    """Extract ready time and final time from a timeline DataFrame.

    Args:
        df: A timeline DataFrame.

    Returns:
        (ready_time, final_time)

    Raises:
        AssertionError: Raise when ready_time >= final_time.
    """
    df = df.replace(0, np.nan).dropna()
    ready_time = max(int(df.index[0].hour * 60) - 360, 0)
    final_time = min(int((df.index[-1].hour + 1) * 60) + 360, 1429)
    assert ready_time < final_time
    return ready_time, final_time


def print_task_type(scheme: dict):
    """Parse the daily scheme and print it.

    Args:
        scheme: Daily scheme.

    Returns:
        None
    """
    _scheme = scheme.get('scheme')
    if _scheme:
        print('VEH_ID\tTASKS\tTASKS_TYPES')
        for veh_id, veh_scheme in enumerate(_scheme):
            veh_tasks = veh_scheme['tasks']
            task_types = [x['task_type'] for x in veh_tasks]
            print(f"{veh_id}\t\t{len(veh_tasks)}\t\t{task_types}")

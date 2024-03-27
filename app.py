from flask import Flask, request, jsonify, Response

from modules.data_utils import *
from modules.or_utils import daily_scheme, daily_execution, print_task_type
from modules.pred_utils import daily_supply, daily_demand

# 日志设置
file_handler = logging.FileHandler('./log/app.log')
file_handler.setFormatter(formatter)

logger = logging.getLogger('app')
logger.setLevel("INFO")
logger.addHandler(file_handler)

# 创建APP对象
app = Flask(__name__)


# 日供热预测接口
@app.route('/es/pred/daily-supply', methods=['POST'])
def get_pred_daily_supply() -> tuple[Response, int]:
    # 检查内容类型
    if not request.is_json:
        return jsonify({"code": 400, "message": 'Content-Type must be application/json'}), 400

    # 检查必传参数和值类型
    required_params = {
        "depot_id": str,
        "date": str
    }

    # 检查参数类型
    err_params = check_required_params(required_params)
    if err_params:
        return err_params

    # 获取参数
    params = {
        "depot_id": str(request.json.get("depot_id")),
        "date": str(request.json.get("date"))
    }
    wrap_log('Reqeust: ' + dumps(params))

    result = jsonify({"code": 200, "message": "DATA IS NULL", "data": None}), 200
    # 获取data
    try:
        data = daily_supply(**params)
        result = jsonify({"code": 200, "message": "success", "data": data}), 200
    except FourHundredError as e:
        result = jsonify({"code": 400, "message": str(e)}), 400
    except Exception as e:
        result = jsonify({"code": 500, "message": str(e)}), 500
    finally:
        logger.info(f"Reqeust: {dumps(request.json)}\nResponse: {result[0].get_data(as_text=True)}")
        return result


# 日用热预测接口
@app.route('/es/pred/daily-demand', methods=['POST'])
def get_pred_daily_demand() -> tuple[Response, int]:
    # 检查内容类型
    if not request.is_json:
        return jsonify({"code": 400, "message": 'Content-Type must be application/json'}), 400

    # 检查必传参数和值类型
    required_params = {
        "demand_id": str,
        "date": str
    }
    err_params = check_required_params(required_params)
    if err_params:
        return err_params

    # 获取参数
    params = {
        "demand_id": str(request.json.get("demand_id")),
        "date": str(request.json.get("date")),
    }
    wrap_log(dumps(params))

    result = jsonify({"code": 200, "message": "DATA IS NULL", "data": None}), 200
    # 获取data
    try:
        data = daily_demand(**params)
        result = jsonify({"code": 200, "message": "success", "data": data}), 200
    except FourHundredError as e:
        result = jsonify({"code": 400, "message": str(e)}), 400
    except Exception as e:
        result = jsonify({"code": 500, "message": str(e)}), 500
    finally:
        logger.info(f"Reqeust: {dumps(request.json)}\nResponse: {result[0].get_data(as_text=True)}")
        return result


# 日计划排程接口
@app.route('/es/or/daily-scheme', methods=['POST'])
def get_daily_scheme() -> tuple[Response, int]:
    # 检查内容类型
    if not request.is_json:
        return jsonify({"code": 400, "message": 'Content-Type must be application/json'}), 400

    # 检查必传参数和值类型
    required_params = {
        "date": str,
        "depot_id": str,
        "demand_ids": list,
        "veh_ids": list,
        "tank_ids": list,
    }
    err_params = check_required_params(required_params)
    if err_params:
        return err_params

    # 获取参数
    params = {
        "date": request.json.get("date"),
        "depot_id": request.json.get("depot_id"),
        "demand_ids": request.json.get("demand_ids"),
        "veh_ids": request.json.get("veh_ids"),
        "tank_ids": request.json.get("tank_ids"),
    }
    wrap_log(dumps(params))

    result = jsonify({"code": 200, "message": "DATA IS NULL", "data": None}), 200
    # 获取data
    try:
        data = daily_scheme(**params)
        print_task_type(data)
        result = jsonify({"code": 200, "message": "success", "data": data}), 200
    except FourHundredError as e:
        result = jsonify({"code": 400, "message": str(e)}), 400
    except FiveHundredError as e:
        result = jsonify({"code": 500, "message": str(e)}), 500
    except Exception as e:
        result = jsonify({"code": 500, "message": str(e)}), 500
    finally:
        logger.info(f"Reqeust: {dumps(request.json)}\nResponse: {result[0].get_data(as_text=True)}")
        return result


# 日实际排程接口
@app.route('/es/or/daily-execution', methods=['POST'])
def get_daily_execution() -> tuple[Response, int]:
    # 检查内容类型
    if not request.is_json:
        return jsonify({"code": 400, "message": 'Content-Type must be application/json'}), 400

    # 检查必传参数和值类型
    required_params = {
        "date": str,
        "depot_id": str,
        "demand_ids": list,
        "veh_ids": list,
        "tank_ids": list,
    }
    err_params = check_required_params(required_params)
    if err_params:
        return err_params

    # 获取参数
    params = {
        "date": request.json.get("date"),
        "depot_id": request.json.get("depot_id"),
        "demand_ids": request.json.get("demand_ids"),
        "veh_ids": request.json.get("veh_ids"),
        "tank_ids": request.json.get("tank_ids"),
    }
    wrap_log(dumps(params))
    result = jsonify({"code": 200, "message": "DATA IS NULL", "data": None}), 200

    # 获取data
    try:
        data = daily_execution(**params)
        print_task_type(data)
        result = jsonify({"code": 200, "message": "success", "data": data}), 200
    except FourHundredError as e:
        result = jsonify({"code": 400, "message": str(e)}), 400
    except FiveHundredError as e:
        result = jsonify({"code": 500, "message": str(e)}), 500
    except Exception as e:
        result = jsonify({"code": 500, "message": str(e)}), 500
    finally:
        logger.info(f"Reqeust: {dumps(request.json)}\nResponse: {result[0].get_data(as_text=True)}")
        return result


# 健康检查接口
@app.route('/health', methods=['GET'])
def health_check():
    # 在此处可以添加更复杂的健康检查逻辑
    # 如果一切正常，返回一个表示健康的JSON响应
    response = {'status': 'ok'}
    return jsonify(response), 200


def check_required_params(required_params: dict) -> tuple[Response, int]:
    """Check whether the params of the HTTP request is valid."""
    for param, param_type in required_params.items():
        if param not in request.json:
            return jsonify({"code": 400, "message": str(param) + "is required."}), 400
        if not isinstance(request.json[param], param_type):
            return jsonify({"code": 400, "message": "The type of " + str(param) + "is incorrect."}), 400


if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=9989)

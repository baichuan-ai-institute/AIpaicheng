"""数据模块, 用于从各种数据源获取数据."""
import logging
from configparser import ConfigParser
from datetime import datetime
from json import dumps
from logging.handlers import TimedRotatingFileHandler

import pandas as pd
import requests
from dingtalkchatbot.chatbot import DingtalkChatbot
from termcolor import colored

# 数据源配置文件
CONFIG_FILE_PATH = 'conf/config.ini'

# 模块日志目录
MODULES_LOG_FOLDER_PATH = 'log/modules_log'

# kafka消息存放目录
KAFKA_LOG_FOLDER = 'log/kafka'


class DingTalkHandler(logging.Handler):
    """钉钉机器人Handler"""
    def emit(self, record):
        log_entry = self.format(record)
        self.send_to_dingtalk(log_entry)

    @staticmethod
    def send_to_dingtalk(message):
        """Send message to DingTalk."""
        webhook = 'https://oapi.dingtalk.com/robot/send?' \
                  'access_token=78e7d1258e752b780e0f1849b642f89f9576d737e9386f680f254334ff292a45'
        chatbot = DingtalkChatbot(webhook)
        title = '来自 AI排程项目[测试环境-V100] 的消息'
        # title = '来自 AI排程项目[生产环境-算法端] 的消息'
        if 'INFO' in message:
            headline = '<font color="#13B455">级别:安全</font>'
        elif 'WARNING' in message:
            headline = '<font color="#F9DD26">级别:警告</font>'
        elif 'ERROR' in message:
            headline = '<font color="#F9BE00">级别:错误</font>'
        elif 'CRITICAL' in message:
            headline = '<font color="#F80000">级别:致命</font>'
        else:
            headline = ''

        message_datetime = datetime.strptime(message.split('\t')[0], '%Y-%m-%d %H:%M:%S,%f')
        if message_datetime.isoweekday() in range(1, 6) and message_datetime.hour in range(9, 18) \
                and message_datetime.minute in range(10):
            # 发送钉钉消息
            chatbot.send_markdown(
                title=title,
                text=f'### **{headline}**\n'
                     f'**{title}**\n'
                     f'**{message}**\n\n',
                at_mobiles=['18811347977'],
            )


# 模块日志设置
formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')

file_handler = TimedRotatingFileHandler(MODULES_LOG_FOLDER_PATH + '/data_utils_log', when='midnight', backupCount=14)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel("WARNING")

dingtalk_handler = DingTalkHandler()
dingtalk_handler.setLevel("ERROR")
dingtalk_handler.setFormatter(formatter)

logger = logging.getLogger('data_utils_logger')
logger.setLevel("INFO")
logger.addHandler(file_handler)
logger.addHandler(dingtalk_handler)
logger.addHandler(console_handler)


class FourHundredError(Exception):
    """400 Error"""
    def __init__(self, err_str: str):
        super().__init__()
        self.err_str = err_str

    def __str__(self):
        wrap_log(self.err_str, color='red')
        return self.err_str


class FiveHundredError(Exception):
    """500 Error"""
    def __init__(self, err_str: str):
        super().__init__()
        self.err_str = err_str

    def __str__(self):
        wrap_log(self.err_str, color='red')
        return self.err_str


def parse_config_ini() -> ConfigParser:
    """Return ConfigParser."""
    parser = ConfigParser()
    parser.read(CONFIG_FILE_PATH)
    return parser


def get_device_sns(device_type: str) -> list[str] | None:
    """Return all SN of specific device type.

    Args:
        device_type: The device type. It can be one of {"ces", "esc"}.

    Returns:
        A list of all SN. Return None if any error happens.

    Examples:
        get_device_sns("ces")

        Returns: ['200123011286', '200123041346', '200123011284', '200123081346', '200123011288']
    """
    try:
        # 接口URL
        config = parse_config_ini()['BigDataAPI']
        url = f"https://{config['Server']}:{config['Port']}/admin/algorithm/get-equip-sn-by-device-type" \
              f"?deviceType={device_type}"

        # 请求头
        headers = {'token': config['Token']}

        # 返回响应
        response = requests.request("GET", url, headers=headers)

        if response.status_code == 200:
            # 解析响应内容并提取 token
            data = response.json().get("data")
            if data:
                # 记录接口成功的日志
                logger.info(f"OK\tURL: {url}\tHEADERS: {dumps(headers)}")
                return data
            else:
                # 记录接口成功但是数据为空的日志
                logger.warning(f"NULL\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
        else:
            # 记录接口报错的日志
            logger.error(f"FAILED\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
    except Exception:
        # 记录函数报错的日志
        logger.critical(f"FAILED\tARGS: {dumps({'device_type': device_type})}", exc_info=True)


def get_latest_value(device_type: str, sn: str, metric: str) -> float | None:
    """Return the latest value of a specific device_type, sn and metric.

    Args:
        device_type: The device type. It can be one of {"ces", "esc"}.
        sn: An identifier of a vehicle or depot or demand.
        metric: An identifier of a sensor.

    Returns:
        The latest value. Returns None if any error happens.

    Examples:
        get_latest_value("esc", sn, "esc_T_R_TbPsP2")

        Returns: 1.34675
    """
    try:
        # 接口URL
        config = parse_config_ini()['BigDataAPI']
        url = f"https://{config['Server']}:{config['Port']}/admin/algorithm/signal?" \
              f"deviceType={device_type}&metricList={metric}&sn={sn}"

        # 请求头
        headers = {'token': config['Token']}

        # 返回响应
        response = requests.request("GET", url, headers=headers)

        if response.status_code == 200:
            # 解析响应内容并提取 token
            data = response.json().get("data")
            if data:
                metric_data = data.get('metricData')
                if metric_data:
                    value = metric_data[0].get('value')
                    result = float(value) if value is not None else value
                    # 记录接口成功的日志
                    logger.info(f"OK\tSN: {sn}\tURL: {url}\tHEADERS: {dumps(headers)}")
                    return result
                else:
                    # 记录接口成功但是数据为空的日志
                    logger.warning(f"NULL\tSN: {sn}\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
            else:
                # 记录接口成功但是数据为空的日志
                logger.warning(f"NULL\tSN: {sn}\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
        else:
            # 记录接口报错的日志
            logger.error(f"FAILED\tSN: {sn}\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
    except Exception:
        # 记录函数报错的日志
        logger.critical(
            f"FAILED\tARGS: {dumps({'device_type': device_type, 'sn': sn, 'metric': metric})}",
            exc_info=True,
        )


def get_period_df(sn: str, start_time: str, end_time: str, metric) -> pd.DataFrame | None:
    """Return a DataFrame of a specific sn and metric between start time and end time.

    Args:
        sn: An identifier of a vehicle or depot or demand.
        start_time: The start time as "%Y-%m-%d %H:%M:%S".
        end_time: The end time as "%Y-%m-%d %H:%M:%S".
        metric: An identifier of a sensor.

    Returns:
        A DataFrame. Returns None if any error happens.

    Examples:
        get_period_df("200123051356", "2024-03-14 00:00:00", "2024-03-14 01:00:00", "ces_S_R_Ifr")

        Returns:
            ts                    value
            2024-03-14 00:00:02   0.0
            2024-03-14 00:01:02   0.0
            2024-03-14 00:02:02   0.1
            2024-03-14 00:03:02   0.1
            2024-03-14 00:04:02   0.2
            2024-03-14 00:05:02   0.2
            ...                   ...
            2024-03-14 00:57:02   0.3
            2024-03-14 00:58:02   0.3
            2024-03-14 00:59:02   0.3
    """
    try:
        # 接口URL
        config = parse_config_ini()['BigDataAPI']
        url = f"https://{config['Server']}:{config['Port']}/admin/algorithm/signal-history?" \
              f"sn={sn}&startTime={start_time}&endTime={end_time}&metric={metric}"

        # 请求头
        headers = {'token': config['Token']}

        # 返回响应
        response = requests.request("GET", url, headers=headers)

        message = f'SN: {sn}\tURL: {url}\tHEADERS: {dumps(headers)}'

        if response.status_code == 200:
            # 解析响应内容并提取 token
            data = response.json().get("data")
            if data:
                df = pd.DataFrame.from_records(data, index='ts')
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
                df = df.sort_index()

                # 记录接口成功的日志
                logger.info(f"OK\t{message}")
                return df
            else:
                # 记录接口成功但是数据为空的日志
                logger.warning(f"NULL\t{message}\tRESPONSE: {response.text}")
        else:
            # 记录接口报错的日志
            logger.error(f"FAILED\t{message}\tRESPONSE: {response.text}")
    except Exception:
        # 记录函数报错的日志
        logger.critical(
            f"FAILED\tARGS: {dumps({'sn': sn, 'start_time': start_time, 'end_time': end_time, 'metric': metric})}",
            exc_info=True,
        )


def get_gps(sn: str) -> tuple[float | None, float | None]:
    """Return the current latitude and longitude of a specific sn.

    Args:
        sn: An identifier of a vehicle or depot or demand.

    Returns:
        The latitude and longitude of the equipment SN. Returns (None, None) if GPS data is missing.

    Examples:
        get_gps("200123081341")

        Returns: (34.652229, 114.006099)
    """
    try:
        # 接口URL
        config = parse_config_ini()['BigDataAPI']
        url = f"https://{config['Server']}:{config['Port']}/admin/algorithm/get-gps-sn-code?deviceSnCode={sn}"

        # 请求头
        headers = {'token': config['Token']}

        # 返回响应
        response = requests.request("GET", url, headers=headers)

        if response.status_code == 200:
            # 解析响应内容并提取 token
            body = response.json()
            data = body.get("data")
            if data:
                # 记录接口成功的日志
                lat, lng = data.get('lat'), data.get('lng')
                if lat and lng:
                    logger.info(f"OK\tSN: {sn}\tURL: {url}\tHEADERS: {dumps(headers)}")
                    return float(lat), float(lng)
                else:
                    # 成功但是数据为空
                    logger.warning(f"NULL\tSN: {sn}\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
                    return None, None
            else:
                # 成功但是数据为空
                logger.warning(f"NULL\tSN: {sn}\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
                return None, None
        else:
            # 记录接口报错的日志
            logger.error(f"FAILED\tSN: {sn}\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
            return None, None
    except Exception:
        # 记录函数报错的日志
        logger.critical(f"FAILED\tARGS: {dumps({'sn': sn})}", exc_info=True)
        return None, None


def get_all_veh_sns() -> list[str] | None:
    """Return all SN of vehicles."""
    return get_device_sns('esc')


def get_all_dep_dem_sns() -> list[str] | None:
    """Return all SN of depots and demands."""
    return get_device_sns('ces')


def get_inst_flow_df(sn, start_time: str, end_time: str) -> pd.DataFrame:
    """Return an instant flow DataFrame between start time and end time.

    Args:
        sn: An identifier of a vehicle or depot or demand.
        start_time: The start time as "%Y-%m-%d %H:%M:%S".
        end_time: The end time as "%Y-%m-%d %H:%M:%S".

    Returns:
        A DataFrame. Returns an empty DataFrame if no data is found.
    """
    df = get_period_df(sn, start_time, end_time, "ces_S_R_Ifr")
    if df is None:
        return pd.DataFrame()
    else:
        return df


def get_all_e_fence() -> tuple[dict | None, dict | None, dict | None]:
    """Return all electric fence, all subject type and all subject id.

    Returns:
        All electric fence is a dict where the key is sn, and the value is a list of (longitude, latitude).
        All subject type is a dict where the key is sn, and the value is the subject type.
        All subject id is a dict where the key is sn, and the value is the subject id.
        Returns (None, None, None) if any error happens or data is missing.

    Examples:
        get_all_e_fence()

        Returns:
            ({'200123011219': [(114.0264251, 34.6490876),
                               (114.0260057, 34.6473347),
                               (114.0275299, 34.6472747),
                               (114.0276992, 34.6489443),
                               (114.0273767, 34.6489976),
                               (114.0273767, 34.6489976),
                               (114.0264251, 34.6490876)]},
             {'200123011219': 2},
             {'200123011219': '1655404599825186818'})
    """
    try:
        # 接口URL
        config = parse_config_ini()['BigDataAPI']
        url = f"https://{config['Server']}:{config['Port']}/admin/algorithm/electric-fence"

        # 请求参数
        headers = {'token': config['Token']}

        # 返回响应
        response = requests.request("GET", url, headers=headers)

        if response.status_code == 200:
            # 解析响应内容并提取 token
            body = response.json()
            data = body.get("data")
            if data:
                all_e_fence = {}
                all_subject_type = {}
                all_subject_id = {}
                for sub in data:
                    sn = sub.get('sn')
                    e_fence = sub.get('e_fence')
                    subject_type = sub.get('subject_type')
                    subject_id = sub.get('subject_id')
                    if sn and e_fence and subject_type:
                        all_e_fence[sn] = [(float(lng), float(lat)) for x in e_fence if
                                           (lat := x.get('lat')) and (lng := x.get('lng'))]
                        all_subject_type[sn] = subject_type
                        all_subject_id[sn] = subject_id

                # 记录接口成功的日志
                logger.info(f"OK\tURL: {url}\tHEADERS: {dumps(headers)}")
                return all_e_fence, all_subject_type, all_subject_id
            else:
                # 成功但是数据为空
                logger.warning(f"NULL\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
                return None, None, None
        else:
            # 记录接口报错的日志
            logger.error(f"FAILED\tURL: {url}\tHEADERS: {dumps(headers)}\tRESPONSE: {response.text}")
            return None, None, None
    except Exception:
        # 记录函数报错的日志
        logger.critical(f"FAILED", exc_info=True)
        return None, None, None


def wrap_log(log: str, color: str | None = None):
    """Print log with color and datetime."""
    print(colored('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] ' + log, color))


def check_node(sn: str | None = None, sns: list[str] | None = None):
    """Check whether the depot and all demands are known.

    Args:
        sn: An identifier of a vehicle or depot or demand.
        sns: A list of sn.

    Returns:
        None

    Raises:
        FourHundredError: Raise if any unknown sn is found.
    """
    # 所有热源和用户的SN
    all_dep_dem_sns = get_all_dep_dem_sns()
    if all_dep_dem_sns:
        if sn:
            if sn not in all_dep_dem_sns:
                raise FourHundredError(f'Unknown sn = {sn}.')
        if sns:
            for _sn in sns:
                if _sn not in all_dep_dem_sns:
                    raise FourHundredError(f'Unknown sn = {sn}.')


def get_gps_dict(sns: list) -> dict[str, tuple[float, float]]:
    """Return a dict of gps of sns.

    Args:
        sns: A list of sn.

    Returns:
        A dict of gps where the key is sn and the value is (longitude, latitude).

    Examples:
        get_gps_dict(["200123051356", "200123011219", "200123041336"]

        Returns: {
            '200123051356': (34.6584475, 114.0225029),
            '200123011219': (34.6490876, 114.0264251),
            '200123041336': (34.662396, 114.0215706),
        }
    """
    all_e_fence, _, _ = get_all_e_fence()
    all_gps = {_sn: (e_fence[0][1], e_fence[0][0]) for _sn, e_fence in all_e_fence.items() if
               isinstance(all_e_fence, dict)}
    result = {}
    for sn in sns:
        if gps := all_gps.get(sn):
            result[sn] = gps
        else:
            logger.warning(f"Failed to get GPS of {sn}.")
    return result


def get_current_time() -> int:
    """Return the current time['%H:%M'] in minutes.

    Examples:
        get_current_time()

        Returns: 1053
    """
    now = datetime.now()
    return int(now.hour * 60 + now.minute)

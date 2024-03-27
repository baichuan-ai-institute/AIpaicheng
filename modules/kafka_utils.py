"""kafka模块, 用于接收并保存kafka消息."""
import os
from json import loads

from kafka import KafkaConsumer

from modules.data_utils import *

# 日志设置
file_handler = logging.FileHandler(MODULES_LOG_FOLDER_PATH + '/kafka_utils_log')
file_handler.setFormatter(formatter)
logger = logging.getLogger('kafka_utils_logger')
logger.setLevel("INFO")
logger.addHandler(file_handler)
logger.addHandler(dingtalk_handler)


def save_kafka():
    """Save kafka message to daily log file."""
    try:
        # 读取 Kafka 服务器的地址,端口,主题
        config = parse_config_ini()['Kafka']
        bootstrap_servers = config['Server'] + ':' + config['Port']
        topic = config['Topic']

        # 创建 Kafka 消费者
        consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers)

        # 创建日志文件夹
        os.makedirs(KAFKA_LOG_FOLDER, exist_ok=True)

        # 根据kafka消息的'timestamp'字段按天保存消息, 日志文件名格式: %Y-%m-%d.log
        for message in consumer:
            kafka_message = loads(message.value.decode('utf-8'))
            timestamp = kafka_message.get('timestamp')

            if timestamp:
                # 解析时间戳并获取日期部分
                message_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').date()

                # 构造日志文件路径
                log_file_name = message_date.strftime('%Y-%m-%d.log')
                log_file_path = os.path.join(KAFKA_LOG_FOLDER, log_file_name)

                # 将消息写入日期对应的日志文件
                with open(log_file_path, 'a') as log_file:
                    log_file.write(dumps(kafka_message) + "\n")
                    log_file.flush()

        # 关闭 Kafka 消费者
        consumer.close()
    except Exception:
        logger.critical(f"FAILED", exc_info=True)


if __name__ == "__main__":
    save_kafka()

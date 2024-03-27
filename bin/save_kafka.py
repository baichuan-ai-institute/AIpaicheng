# 一直运行, 接收kafka消息
import sys
import os

# 添加modules的绝对路径添加到 sys.path
package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from modules.kafka_utils import *


def main():
    save_kafka()


if __name__ == '__main__':
    main()

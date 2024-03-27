# 使用Python 3.10.6的官方镜像作为基础镜像
FROM python:3.10.6

# 设置环境变量，将/root/.local/bin目录添加到PATH
ENV PATH=$PATH:/root/.local/bin

# 设置工作目录
WORKDIR /app

# 复制当前目录所有内容到容器中的/app目录
COPY . .

# 复制定时任务到/etc/cron.d
COPY ./bin/cron_job /etc/cron.d/

# 安装Python依赖项
RUN pip install --user -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 使用清华大学的Debian镜像源, 安装cron, tzdata, vim, 设置时区, 设置定时任务
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ buster main" > /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/debian/ buster main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y cron tzdata vim lsof && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    chmod 0644 /etc/cron.d/cron_job && \
    crontab /etc/cron.d/cron_job


# 声明容器内部应用程序使用的端口
EXPOSE 9989

# 设置执行权限
RUN chmod +x ./bin/start.sh

# 启动
CMD ["./bin/start.sh"]

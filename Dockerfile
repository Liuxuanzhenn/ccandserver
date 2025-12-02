# 支持多架构
# 默认使用构建平台，可通过 --platform 参数覆盖
# 例如：docker build --platform linux/arm64 -t model-convert-deploy .
FROM python:3.10-slim-bullseye

LABEL maintainer="carolineleeton@outlook.com"

# 更换APT源为清华源并安装系统依赖
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制整个项目目录到容器
COPY . /app

# 设置 pip 源为清华源
ENV PIP_INDEX_URL https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip

# 安装依赖
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    -r requirements.txt

# 可选：如果需要 opencv-python-headless（如果代码中使用了 OpenCV）
# RUN pip uninstall opencv-python -y 2>/dev/null || true
# RUN pip install --no-cache-dir opencv-python-headless

# 运行 Flask 应用
CMD ["python", "-m", "app.server"]

# 暴露 Flask 应用运行的端口
EXPOSE 5000


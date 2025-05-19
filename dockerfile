FROM python:3.10-slim
# WORKDIR /app

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 默认启动命令
CMD [“bash”]

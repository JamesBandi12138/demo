FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    pandas \
    numpy \
    fastapi \
    uvicorn \
    pydantic

COPY . /app

EXPOSE 5000

CMD ["python", "api.py"]

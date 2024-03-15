FROM python:3.9-slim

WORKDIR /workdir

COPY ./requirements.txt /workdir/requirements.txt

RUN pip install \
    torch==1.12.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/pip
    
RUN pip install --no-cache-dir --upgrade -r /workdir/requirements.txt

EXPOSE 9080

COPY ./server.py /workdir/

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9080"]
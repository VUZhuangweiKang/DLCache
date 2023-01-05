FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install nano
RUN mkdir app
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN chmod +x *
COPY /opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py /opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/fetch-bk.py
COPY fetch.py /opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py
COPY /opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/worker.py /opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/worker-bk.py
COPY worker.py /opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/worker.py
# CMD [ "python3", "main.py", "-j", "8", "--pretrained", "-e"]
CMD ["bash"]
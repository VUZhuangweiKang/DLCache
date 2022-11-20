FROM pytorch/pytorch:latest
RUN apt update 
RUN apt install -y vmtouch python3-pip nano
RUN mkdir hdd
RUN mkdir app
WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt
RUN chmod +x *
CMD ["bash"]
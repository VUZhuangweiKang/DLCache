FROM pytorch/pytorch:latest
RUN apt update && apt install -y python3-pip vmtouch nano
RUN mkdir app
WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt
RUN chmod +x *
CMD ["bash"]
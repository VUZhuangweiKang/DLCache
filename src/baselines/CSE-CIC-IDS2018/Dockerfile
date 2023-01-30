FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y nano vmtouch
RUN mkdir app
WORKDIR /app
COPY datasets/CSE-CIC-IDS2018/ .
COPY databus ./databus
COPY lib ./lib
RUN pip install -r requirements.txt
RUN chmod +x *
Run ./entrypoint.sh
CMD ["bash"]
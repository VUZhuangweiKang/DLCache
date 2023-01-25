FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y nano vmtouch
RUN mkdir app
WORKDIR /app
COPY datasets/ImageNet/ .
COPY databus ./databus
COPY lib ./lib
RUN pip install -r requirements.txt
RUN chmod +x *
ENTRYPOINT ["bash", "entrypoint.sh"]
CMD ["bash"]
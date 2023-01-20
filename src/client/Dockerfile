FROM ubuntu:20.04
RUN apt update && apt install -y python3-pip nano nfs-common nvidia-utils-460
RUN mkdir app
WORKDIR /app
COPY client ./
COPY databus ./databus
RUN pip3 install -r requirements.txt
RUN chmod +x *
CMD [ "bash" ]
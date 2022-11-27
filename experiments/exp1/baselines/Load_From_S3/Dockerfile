FROM pytorch/pytorch:latest
RUN apt update 
RUN apt install -y python3-pip
RUN mkdir app
WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt
RUN chmod +x *
CMD ["bash"]
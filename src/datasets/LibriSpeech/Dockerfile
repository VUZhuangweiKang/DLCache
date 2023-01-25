FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y nano git libsndfile1 sox liblzma-dev
RUN mkdir app
WORKDIR /app
COPY datasets/LibriSpeech/ .
COPY databus ./databus
COPY lib ./lib
RUN pip install -r requirements.txt
RUN pip install -e .
RUN bash -c entrypoint.sh
RUN chmod +x *
# CMD [ "python3", "train.py", "+configs", "librispeech"]
CMD [ "bash" ]

FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y nano git libsndfile1 sox liblzma-dev vmtouch
RUN mkdir app
WORKDIR /app
COPY datasets/LibriSpeech/ .
COPY databus ./databus
COPY lib ./lib
RUN pip3 install -r requirements.txt
RUN pip3 install -e .
RUN pip3 uninstall -y torch torchaudio
RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN chmod +x *
RUN ./entrypoint.sh
# CMD [ "python3", "train.py", "+configs", "librispeech"]
CMD [ "bash" ]

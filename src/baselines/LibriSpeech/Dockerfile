FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y nano git libsndfile1 sox liblzma-dev vmtouch
RUN mkdir app
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt
RUN pip3 install -e .
RUN pip3 uninstall -y torch torchaudio
RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN chmod +x *
CMD [ "bash" ]

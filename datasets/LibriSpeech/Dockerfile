FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y nano git libsndfile1 sox liblzma-dev
RUN mkdir deepspeech
WORKDIR /deepspeech
COPY . ./
RUN pip install -r requirements.txt
RUN pip install -e .
RUN chmod +x *
# CMD [ "python3", "train.py", "+configs", "librispeech"]
CMD [ "bash" ]

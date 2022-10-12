FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install nano
RUN mkdir app
WORKDIR /app
COPY src ./
RUN pip install -r requirements.txt
RUN pip install -e .
RUN chmod +x *
# CMD [ "python3", "main.py", "-j", "8", "--pretrained", "-e"]
CMD ["bash"]
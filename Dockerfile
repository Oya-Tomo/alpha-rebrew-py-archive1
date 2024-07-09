FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install torch torchvision numpy ray

WORKDIR /train

CMD [ "python3", "src/train.py"]
#FROM ubuntu:latest
# вы указываете, что ваш образ будет основан на образе ubuntu:latest

#RUN ["mkdir", "/my_super_dir"]
# создавая образ, docker должен будет также создать диреторию в корневой папке


#FROM ubuntu:latest
#RUN ["mkdir", "/my_super_dir"]
#RUN ["echo", "Hello world!",  “>”, "/my_super_dir/super_file.txt"]

#FROM python:3.11-slim
#LABEL author="mle-student"
#COPY . ./my_super_dir

FROM python:3.11-slim
LABEL author="mle-student"
COPY . ./my_super_dir
WORKDIR my_super_dir
CMD ["python3", "./main.py"]


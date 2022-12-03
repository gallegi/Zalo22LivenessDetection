# build docker
# docker build -t train4ever .

# py run
# sudo docker run -it -v /home/ntnam/Personal/Zalo22/Zalo22LivenessDetection/data:/data -v /home/ntnam/Personal/Zalo22/Zalo22LivenessDetection/private_result:/result  train4ever:latest /bin/bash /code/predict.sh

# notebook run
sudo docker run -it -p 9777:9777 -v /home/ntnam/Personal/Zalo22/Zalo22LivenessDetection/data:/data -v /home/ntnam/Personal/Zalo22/Zalo22LivenessDetection/private_result:/result  train4ever:latest /bin/bash /code/start_jupyter.sh
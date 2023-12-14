#학습용 도커파일

#이거 버전에 맞게 설정해주세여
FROM python:3.8

# 작업 디렉토리 설정
WORKDIR /app

# 현재 모델 폴더의 내용을 컨테이너로 복사
COPY ./BloGeeK-Data/PolarityRecognition /app

# requirements.txt 파일이 있다면 종속성 설치
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 모델 실행을 위한 명령어 설정
# 현정님이 맞게 짜세여
CMD ["python", "model/train.py"]

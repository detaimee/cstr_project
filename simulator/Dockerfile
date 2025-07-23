# Python 3.9 slim 이미지 사용 (가벼운 버전)
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
# numpy와 scipy는 과학 계산용, pymodbus는 Modbus 통신용
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pymodbus==3.5.2

# 시뮬레이터 파일 복사
COPY cstr_simulator.py .

# Modbus TCP 포트 노출
EXPOSE 5020

# 컨테이너 시작 시 시뮬레이터 실행
CMD ["python", "-u", "cstr_simulator.py"]
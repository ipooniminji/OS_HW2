# 가벼운 Python 3.11 slim 버전 사용 (ML 라이브러리 구동에 충분하며 이미지 크기를 최소화함)
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 파이썬 환경 변수 설정
# PYTHONDONTWRITEBYTECODE: .pyc 파일 생성 방지
# PYTHONUNBUFFERED: 파이썬 출력을 버퍼링하지 않아 로그를 즉시 컨테이너로 스트리밍
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 의존성 파일 복사
COPY requirements.txt /app/

# 패키지 설치: --no-cache-dir를 사용하여 이미지 크기를 획기적으로 줄힘
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 모델, UI 코드가 통합된 단일 앱 코드 복사
COPY main.py /app/

# 컨테이너 외부로 8000번 포트 노출
EXPOSE 8000

# 앱 실행 (로컬 환경이 아니므로 성능을 위해 --reload 옵션은 제외)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    espeak-ng \
    libespeak-ng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN find /usr -name "libespeak-ng.so" -print -quit | xargs -I {} ln -s {} /usr/lib/libespeak-ng.so

ENV PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/libespeak-ng.so

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/demo_phonetic.py"]
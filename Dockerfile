FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY src src
COPY config config
RUN pip install --no-cache-dir .
ENV HOST=0.0.0.0
EXPOSE 8000
CMD ["local-llm"]

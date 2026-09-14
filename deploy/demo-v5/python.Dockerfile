FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ANONYMIZED_TELEMETRY=False OMP_NUM_THREADS=4
WORKDIR /app/backend/python
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY backend/python/cluster-engine/pyproject.toml backend/python/cluster-engine/README.md ./cluster-engine/
COPY backend/python/cluster-engine/src ./cluster-engine/src
COPY backend/python/requirements.txt ./
RUN pip install -r requirements.txt
COPY backend/python/app ./app
COPY backend/python/samples ./samples
COPY backend/python/cluster-engine/configs ./cluster-engine/configs
COPY backend/python/cluster-engine/data ./cluster-engine/data
COPY backend/python/cluster-engine/models ./cluster-engine/models
COPY backend/python/cluster-engine/artifacts/knowledge_bases /opt/demo-knowledge-bases
COPY deploy/demo-v5/python-entrypoint.py /opt/python-entrypoint.py
COPY deploy/demo-v5/source-model.json deploy/demo-v5/relocate-kb.py /opt/
RUN python /opt/relocate-kb.py
EXPOSE 8000
CMD ["python", "/opt/python-entrypoint.py"]

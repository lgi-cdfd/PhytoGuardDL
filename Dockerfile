FROM python:3.11-slim

LABEL maintainer="Laboratory of Genome Informatics (LGI), BRIC-CDFD, Hyderabad, India"
LABEL description="PhytoGuardDL: Unified Plant Biotic Defense & Abiotic Stress Gene Discovery Web Server and REST API"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY webserver /app/webserver
COPY phytoguarddl_client.py /app/phytoguarddl_client.py
COPY examples /app/examples
COPY LICENSE /app/LICENSE
COPY README.md /app/README.md
RUN mkdir -p /app/models

EXPOSE 8000

CMD ["python", "webserver/server_phytoguarddl.py", "--host", "0.0.0.0", "--port", "8000"]

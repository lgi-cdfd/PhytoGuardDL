<div align="center">

# PhytoGuardDL

**Unified Two-Phase Deep Learning & Stacking Web Server for Plant Biotic Defense, Abiotic Stress & Dual Crosstalk Gene Discovery**

[![Web Server](https://img.shields.io/badge/Web%20Server-phytoguarddl.cdfd.org.in-0a3d2e?style=for-the-badge)](https://phytoguarddl.cdfd.org.in)
[![License: MIT](https://img.shields.io/badge/License-MIT-1a6b4a?style=for-the-badge)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-1a5276?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Container%20Ready-0db7ed?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![REST API](https://img.shields.io/badge/REST%20API-FastAPI-009688?style=for-the-badge)](phytoguarddl_client.py)
[![BRIC-CDFD](https://img.shields.io/badge/LGI-BRIC--CDFD%20Hyderabad-6c3483?style=for-the-badge)](https://www.cdfd.org.in)

</div>

---

## Overview

**PhytoGuardDL** ([https://phytoguarddl.cdfd.org.in](https://phytoguarddl.cdfd.org.in)) is a freely accessible, open-source two-phase deep learning and stacking ensemble web server, REST API, and containerized standalone tool for genome-wide prediction and functional decoding of **plant biotic defense**, **abiotic stress**, and **dual stress–defense crosstalk** genes across land plants.

- **No Login · No Registration · Zero Tracking Cookies** — Free for academic and non-commercial use under the [MIT License](LICENSE).
- **Two-Phase Architecture:**
  - **Phase I:** Calibrated 5-model deep stacking meta-ensemble (Deep MLP, XGBoost, HistGBM, ExtraTrees, Random Forest) on 496-D multi-scale sequence, 52 biotic/abiotic Pfam channel, and biophysical descriptors (with optional 5,120-D multi-pooled ESM-2 650M scaling).
  - **Phase II:** 3-axis biological decoding (`Biotic Defense`, `Abiotic Stress`, `Dual Crosstalk`), verified Pfam/InterPro, QuickGO, and KEGG plant pathway mapping, locus domain coordinates, 19-aa Kyte-Doolittle hydropathy profiles, and an interactive vector SVG / 600 DPI figure studio.

---

## Quick Start

### 1. Web Server (No Installation Required)
Access the interactive web portal directly at **[https://phytoguarddl.cdfd.org.in](https://phytoguarddl.cdfd.org.in)**:
- **Single Sequence** analysis with live interactive 4-tab workbench
- **Batch FASTA** paste (up to 1,000 sequences) with bookmarkable job URLs
- **Whole-Proteome Upload** (up to 50,000 sequences) with 31-column CSV/TSV, FASTA, and JSON exports

### 2. Automated Python REST API Client
Run predictions programmatically from any machine without installing dependencies (uses Python standard library only):
```bash
# Check server status
python phytoguarddl_client.py --health

# Predict a single protein sequence
python phytoguarddl_client.py --sequence "MNSFSAFSEMFGSDYEPQGGDYCPTLATSCPKKPAGRKKFRETRHPIYRGVRQRNSGKWVSEVREPNKKTRIWLGTFQTAEMAARAHDVAALALRGRSACLNFADSAWRLRIPESTCAKDIQKAAAEAALAFQDEMCDATTDHGFDMEETLVEAIYTAEQSENAFYMHDEAMFEMPSLLANMAEGMLLPLPSVQWNHNHEVDGDDDDVSLWSY" --name "DREB1A_AT4G25480"

# Batch predict from sample multi-FASTA and save parsed TSV
python phytoguarddl_client.py --fasta examples/sample_proteins.fasta --out predictions.tsv
```

### 3. Standalone Docker / Podman Container (Cross-Platform)
```bash
git clone https://github.com/lgi-cdfd/PhytoGuardDL.git
cd PhytoGuardDL

# Place production model weights (from Zenodo) in models/
docker build -t phytoguarddl .
docker run -p 8000:8000 phytoguarddl
```

### 4. Native Python Installation
```bash
pip install -r requirements.txt
python webserver/server_phytoguarddl.py --host 0.0.0.0 --port 8000
```

---

## Repository Contents

| Path | Description |
| :--- | :--- |
| `webserver/server_phytoguarddl.py` | FastAPI production web server & JSON REST API |
| `webserver/PhytoGuardDL.html` | Interactive 4-tab web workbench & vector SVG / 600 DPI studio |
| `phytoguarddl_client.py` | Standalone Python REST API template client |
| `examples/sample_proteins.fasta` | 5-gene multi-axis sample FASTA (Biotic, Abiotic, Dual Crosstalk, Control) |
| `Dockerfile` & `requirements.txt` | Cross-platform container and Python dependency specification |
| `LICENSE` | MIT open-source license |

---

## Citation & Contact

Developed by **Ramesh Eerapagula** and **Dr. Ajay Kumar Mahato** at the **Laboratory of Genome Informatics (LGI)**, BRIC – Centre for DNA Fingerprinting and Diagnostics (CDFD), Hyderabad 500039, Telangana, India.

For technical inquiries or collaborations, contact: `akmlgi2021@gmail.com`

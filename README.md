# Verisift

> **Find fit beyond keywords.**  
> *AI-powered resume screening, structured candidate intelligence, and recruiter shortlist export.*

[![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CPU-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19%20%2B%20TypeScript-61dafb.svg)](https://react.dev/)
[![Tests](https://img.shields.io/badge/Tests-70%2F70%20Passing-brightgreen.svg)]()
[![Accuracy](https://img.shields.io/badge/Held--Out%20Pair%20Accuracy-81.48%25-success.svg)]()

---

## Master Project Documentation

All research, benchmarks, audits, and engineering specifications across the project are consolidated into two comprehensive master reports:

1. **[Deep Learning Research, Modeling Experiments & Scientific Verification Report](reports/VERISIFT_RESEARCH_AND_MODELING_REPORT.md)**  
   *Covers dataset cleanup, baseline models, SBERT encoder fine-tuning and parameter shift audit, context-window truncation analysis, dual-branch `DenseCrossFeatureNet` architecture, the 81.48% accuracy milestone, cross-split leakage investigation, and zero-shot unseen job benchmarks.*

2. **[Complete Technical Project Dossier & Engineering Master Document](reports/VERISIFT_COMPLETE_PROJECT_DOSSIER.md)**  
   *Covers system requirements, multi-format resume extraction, OCR glued-heading resolution, contact disentanglement, explainability and verbatim skill evidence grounding, Excel and PDF shortlist export with formula injection protection, end-to-end 10-candidate demonstration, deliverables matrix, and the complete catalog of all 18 publication-quality figures and 20 statistical tables.*

---

## 1. Project Overview & Architecture

**Verisift** is an AI-powered candidate intelligence and resume screening web application developed for recruiter triage and candidate ranking. Traditional applicant tracking systems rely on brittle keyword frequency matching that rewards keyword-stuffing and penalizes qualified candidates who use alternative phrasing.

Verisift replaces keyword counters with:
1. **Section-Aware Resume Parsing**: Conservative candidate identity and contact extraction, normalization of 120+ section aliases into 12 canonical categories, glued-heading OCR boundary resolution, and token-bounded semantic chunking.
2. **Contextual Semantic Matching**: Pretrained dense sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`) generating 384-dimensional representations with context windows expanded to native 512 positions.
3. **Deep Pairwise Interaction Head**: A PyTorch neural network (`DenseCrossFeatureNet`) with dedicated dual branches decoupling base identity representations from differential interaction features ($[R, J, |R-J|, R \odot J, \cos(R, J)] \in \mathbb{R}^{1,537}$).
4. **Continuous Ordinal Relevance Scoring & Ranking**: Continuous ranking scores ($0\text{--}100$) that preserve intra-job candidate ordering with deterministic competency tie-breaking.
5. **Recruiter Shortlist Export**: Browser downloads of ranked shortlists as professionally styled Excel workbooks (`.xlsx` via `openpyxl` with auto-filters and frozen panes) or publication-ready PDFs (via `ReportLab` with running pagination), protected against spreadsheet formula injection.

```
+------------------------------------+        +-----------------------------------+
|  Candidate Resume Text (PDF/DOCX)  |        |    Job Description Text (Raw)     |
+------------------------------------+        +-----------------------------------+
                  |                                             |
                  v                                             v
     [ ResumeParser: Structured ]                  [ Text Preprocessor: Clean ]
  - Identity & Contact Disentangled                             |
  - Section Boundaries Detected (120+ aliases)                  |
  - Glued OCR Headings Resolved                                 |
  - Token-Bounded Chunking (256/512 tok)                        |
                  |                                             |
                  v                                             v
     [ SentenceTransformer Encoder ]               [ SentenceTransformer Encoder ]
       (all-MiniLM-L6-v2, 384-dim)                   (all-MiniLM-L6-v2, 384-dim)
                  |                                             |
                  +----------------------+----------------------+
                                         |
                                         v
                      [ Pairwise Feature Space (1,537d) ]
             Feature Vector = [ R, J, |R - J|, R (*) J, cos(R, J) ]
                                         |
                                         v
                         [ DenseCrossFeatureNet Engine ]
                 Branch 1: Identity (768d -> 256d)
                 Branch 2: Differential (769d -> 256d)
                 Gated Fusion: (512d -> 256d -> 64d -> 3d)
                                         |
                                         v
                              [ Softmax Distribution ]
                        P(No Fit), P(Potential Fit), P(Good Fit)
                                         |
                                         v
                       [ Continuous Ordinal Relevance Score ]
            Score = 100 * (0.0 * P_NoFit + 0.5 * P_PotentialFit + 1.0 * P_GoodFit)
                                         |
                                         v
                        [ Descending Candidate Ranking ]
                     Deterministic Competency Tie-Breaking
                                         |
                                         v
                     [ Shortlist Export (PDF & Excel) ]
            Top 100 / Top 200 / Custom with Formula Sanitization
```

---

## 2. Technology Stack

### Deep Learning & Inference Engine
- **Framework**: PyTorch 2.6 (`torch`, `torch.nn`)
- **Sentence Encoder**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings, L2 normalized)
- **Feature Engineering**: NumPy-vectorized pairwise symmetric distance, element-wise product, and cosine similarity (1,537-dim input)
- **Document Text Extraction**: `pypdf` (PDF stream parsing) and `python-docx` (OpenXML document parsing) with magic-byte validation
- **Structured Resume Intelligence**: `ResumeParser` (alias registry, contact disentanglement, glued heading regexes, semantic chunking)
- **Export Engine**: `openpyxl` (Excel `.xlsx` generation) and `ReportLab` (PDF generation with `NumberedCanvas`)

### Backend API & Storage
- **Framework**: Python 3.13 + FastAPI
- **Database**: SQLite 3 with SQLAlchemy 2.x ORM (`backend/data/verisift.db`)
- **Automatic Schema Migration**: Dynamic SQLite schema introspection and column additions (`PRAGMA table_info`)
- **Security & Auth**: Password hashing with `bcrypt` (cost factor 12), PyJWT (`HS256`) access tokens with server-side revocation
- **Testing**: `pytest` + `httpx` TestClient (**70 automated unit, integration, and API regression tests passing**)
- **Server**: Uvicorn ASGI

### Frontend Application
- **Framework**: React 19 + TypeScript (Vite bundler)
- **Routing**: React Router v7
- **UI Design System**: Arctic Signal Palette (`#12343B` Deep Teal, `#27D3B0` Signal Turquoise, `#F4F8FA` Ice White)
- **Typography**: Google Fonts (Sora headings, Inter interface text)
- **Iconography**: Lucide React

---

## 3. Project Directory Structure

```
DL_CP/
├── README.md                                                  # Primary project guide & execution manual
├── train.csv                                                  # Original raw training dataset (6,241 rows)
├── test.csv                                                   # Original raw holdout test dataset (1,759 rows)
├── artifacts/                                                 # Model checkpoints, cached embeddings & splits
│   ├── accuracy_experiments/                                  # Winning DenseCrossFeatureNet checkpoint & metrics
│   ├── finetuned_sbert/                                       # Production fine-tuned SBERT model & weights
│   ├── dataset_audit/                                         # Cleaned master dataset (7,987 rows) & split CSVs
│   ├── semantic_ranker/                                       # Baseline E4 PyTorch pairwise checkpoints
│   ├── baseline/                                              # Classical TF-IDF baseline models
│   └── model_comparison/                                      # Benchmark comparison JSON artifacts
├── backend/                                                   # FastAPI backend server
│   ├── app/                                                   # Core application code (api, core, db, models, schemas, services)
│   ├── data/                                                  # SQLite database (verisift.db) and uploaded resumes
│   ├── tests/                                                 # 70 automated pytest unit and integration tests
│   ├── requirements.txt                                       # Python dependencies
│   └── venv/                                                  # Backend virtual environment
├── frontend/                                                  # React 19 + TypeScript frontend application
│   ├── src/                                                   # React components, pages, services, styles
│   ├── public/                                                # Static public assets
│   ├── package.json                                           # Node dependencies
│   └── vite.config.ts                                         # Vite bundler configuration
├── notebooks/                                                 # Executed Jupyter capstone notebooks
│   ├── Verisift_Dataset_Audit_and_Split_Validation.ipynb      # Capstone Notebook 1: Provenance & leakage audit
│   ├── Verisift_Deep_Learning_Training_and_Evaluation.ipynb    # Capstone Notebook 2: Deep training & evaluation
│   ├── 01_dataset_exploration.ipynb                           # Exploratory data analysis
│   ├── 02_text_preprocessing.ipynb                            # Text normalization experiments
│   ├── 03_baseline_model.ipynb                                # TF-IDF baseline implementation
│   ├── 04_sentence_transformer.ipynb                          # Dense semantic sentence transformer
│   └── 05_long_context_transformer.ipynb                      # Long-context BGE-M3 investigation
├── reports/                                                   # Master reports, figures, and statistical tables
│   ├── VERISIFT_RESEARCH_AND_MODELING_REPORT.md               # Master Deep Learning Research Report
│   ├── VERISIFT_COMPLETE_PROJECT_DOSSIER.md                   # Master Technical Dossier & Engineering Report
│   ├── figures/                                               # 18 publication-quality figures (300 DPI PNG)
│   ├── tables/                                                # 20 statistical tables (CSV format)
│   └── logs/                                                  # Phase verification logs
├── sample_resumes/                                            # Real-world test resumes for demo & manual upload
└── scripts/                                                   # Reproducible pipeline & evaluation scripts
    ├── dataset_pipeline.py                                    # Cleans dataset & creates split subsets
    ├── cache_all_embeddings.py                                # Precomputes 512-token dense embeddings
    ├── run_accuracy_experiments.py                            # Trains & evaluates DenseCrossFeatureNet
    ├── train_finetuned_sbert.py                               # Fine-tunes SBERT layers 4 & 5
    ├── run_end_to_end_demo.py                                 # Executes 10-candidate live screening demo
    └── verify_model_and_metrics.py                            # Verifies all metrics against saved artifacts
```

---

## 4. Quickstart & Execution Guide

### Prerequisites
- **Python:** 3.11 – 3.13 (Python 3.13 recommended)
- **Node.js:** 18+ and npm
- **Hardware:** Intel Core Ultra 5 / modern x86 CPU, 16 GB RAM (CPU execution verified)

---

### Step 1: Start the Backend API Server

```powershell
# Navigate to backend directory and activate virtual environment
cd backend
.\venv\Scripts\Activate.ps1

# Start the FastAPI ASGI server with auto-reload
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive API Documentation (Swagger UI):** `http://127.0.0.1:8000/docs`
- **Alternative ReDoc Documentation:** `http://127.0.0.1:8000/redoc`
- **Health Check Endpoint:** `http://127.0.0.1:8000/health`

---

### Step 2: Start the Frontend Application

In a separate terminal:

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies (if not already installed)
npm install

# Start the Vite development server
npm run dev
```

- **Web Application URL:** `http://localhost:5173`
- **Build Production Bundle:** `npm run build` (`tsc -b && vite build`)

---

### Step 3: Run the Automated Backend Test Suite

The test suite contains **70 comprehensive unit, integration, and API regression tests** covering authentication, job management, multi-format resume parsing, deterministic tie-breaking, explainability evidence generation, and shortlist export in Excel and PDF:

```powershell
# Run from repository root
backend\venv\Scripts\python -m pytest backend/tests -v
```

*Expected Result: `70 passed in ~56s` (100% pass rate).*

---

### Step 4: Run the End-to-End Live Demonstration

Run the automated live screening simulation that ingests a Junior Python / ML Engineer job requisition, screens 10 diverse candidate resumes, and outputs continuous scores, predicted fit tiers, and grounded text evidence:

```powershell
backend\venv\Scripts\python scripts/run_end_to_end_demo.py
```

*Expected Result: 10/10 candidates parsed, scored, and ranked in ~8.3 seconds (~0.83s per candidate on CPU).*

---

### Step 5: Reproduce Model Metrics & Verifications

To independently verify model weights, accuracy metrics, and split leakage guarantees from repository artifacts:

```powershell
backend\venv\Scripts\python scripts/verify_model_and_metrics.py
```

---

### Step 6: Explore Jupyter Notebooks

Launch Jupyter to inspect the fully executed capstone notebooks:

```powershell
backend\venv\Scripts\python -m jupyter notebook
```

- Open `notebooks/Verisift_Dataset_Audit_and_Split_Validation.ipynb` for dataset provenance, deduplication, and leakage analysis.
- Open `notebooks/Verisift_Deep_Learning_Training_and_Evaluation.ipynb` for deep learning model training, validation leaderboards, and test set evaluations.

---

## 5. REST API Reference

| Method | Endpoint | Description | Auth Required |
|---|---|---|:---:|
| `GET` | `/health` | Server health check & service status | No |
| `POST` | `/api/auth/register` | Register a new recruiter account | No |
| `POST` | `/api/auth/login` | Recruiter login; returns JWT access token | No |
| `GET` | `/api/auth/me` | Fetch authenticated recruiter profile | Yes |
| `POST` | `/api/auth/logout` | Revoke current session token | Yes |
| `GET` | `/api/jobs` | List all job requisitions owned by recruiter | Yes |
| `POST` | `/api/jobs` | Create a new job requisition | Yes |
| `GET` | `/api/jobs/{job_id}` | Retrieve specific job requisition details | Yes |
| `PUT` | `/api/jobs/{job_id}` | Update job description or metadata | Yes |
| `DELETE` | `/api/jobs/{job_id}` | Archive / delete a job requisition | Yes |
| `GET` | `/api/jobs/{job_id}/candidates` | List ranked candidates for a job requisition | Yes |
| `POST` | `/api/jobs/{job_id}/candidates/upload` | Upload resume files (PDF / DOCX, multi-file batch) | Yes |
| `POST` | `/api/jobs/{job_id}/screen` | Trigger neural screening & ranking | Yes |
| `GET` | `/api/jobs/{job_id}/candidates/export` | Download ranked shortlist (`format=excel` or `format=pdf`) | Yes |
| `GET` | `/api/candidates/{candidate_id}` | Retrieve candidate profile & explainability evidence | Yes |
| `DELETE` | `/api/candidates/{candidate_id}` | Delete a candidate profile and purge file from disk | Yes |

---

## 6. Verified Metric Highlights

| Dimension | Metric | Score Achieved | Historical Baseline | Status |
|---|---|---:|---:|:---:|
| **Held-Out Pair Test Split** | Classification Accuracy | **81.48%** | 79.72% | **Exceeds $>80\%$ Milestone** |
| | Macro-$F_1$ Score | **0.7944** | 0.7765 | Multi-class balance verified |
| | Weighted-$F_1$ Score | **0.8134** | 0.7951 | High confidence on dominant classes |
| | Intra-Job nDCG@5 | **0.9654** | 0.9380 | Near-optimal top-5 ranking |
| | Pairwise Concordance | **88.49%** | 84.55% | Consistent candidate ordering |
| **Strict Unseen-Job Split** | Classification Accuracy | **54.46%** | 49.52% (TF-IDF) / 52.02% (MLP) | **Exceeds All Baselines** |
| | Macro-$F_1$ Score | **0.4960** | 0.4612 (TF-IDF) / 0.4839 (MLP) | Verified zero-shot transfer |
| | Intra-Job nDCG@5 | **0.7148** | 0.6527 (TF-IDF) | +9.5% relative ranking gain |
| **Cross-Split Leakage Audit** | `pair_train` vs `jd_test` Overlap | **78.74%** | N/A | Explains illusory 95.79% score |

---

## 7. License & Credits

- **Course:** Deep Learning (Recruiter-Facing Decision Support & Resume Matching Capstone)
- **Dataset:** Hugging Face [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
- **Base Embedding Model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by Hugging Face & UKPLab
- **License:** Educational and research use under MIT License.

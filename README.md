## Sustainability SME Agent Expert

This repository contains a multi-part pipeline and agent system for building a Sustainability SME (Subject Matter Expert) assistant, including data extraction, cleaning, chunking, indexing, and a web UI for interaction.

### Repository structure

- **app/** (not committed in this repo): FastAPI/Streamlit (or similar) web application for serving the SME agent as a website.
- **data_for_finetuning/**: Raw text files used for LLM fine-tuning or prompt engineering experiments.
- **data_json/**: JSON-converted documents extracted from PDFs and other sources, used as the base corpus.
- **dataset/**: Original documents (e.g., PDFs) and sample chapters used for experimentation and evaluation.
- **experiments/**: Prompt experiments, outputs, and notes for different agent and retrieval configurations.
- **generated_documents/**: Auto-generated reports, answers, and other content produced by the SME agent.
- **partb/**: PDF text extraction, cleaning, chunking, and preprocessing pipeline.
- **partc/**: Vector database (Milvus) setup and indexing utilities (includes local Milvus binaries and data).
- **partd/**, **parte/**, **partf/**, **partg/**, **parth/**, **parti/**: Iterative agent, retrieval, and orchestration components (multi-agent controllers, observation logs, routing logic, etc.).
- **elasticsearch_local/**: Local Elasticsearch distribution and configuration for search-based retrieval (ignored for git to avoid large binaries).
- **logs and \*.log files**: Operational logs for extraction, indexing, and agent runs.

For more detailed design notes, see:
- `ARCHITECTURE_AND_WORKFLOW.md`
- `PROMPT_BASED_LEARNING_IMPLEMENTATION.md`
- `partb/PROCESSING_PIPELINE_REPORT.md`

### Prerequisites

- Python 3.10+ recommended
- `pip` or `pipenv`/`poetry`
- (Optional) Local Milvus or Elasticsearch if you are not using the embedded/lightweight versions

### Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you need to run the extraction/indexing pipeline, make sure any local services (Milvus, Elasticsearch) referenced in the configs are running or adjust the settings accordingly.

### Running the website (SME Agent UI)

The web UI code lives in the `app/` folder, which is **not** part of this Git repository to keep app-specific secrets and heavy dependencies out of version control. On your local machine:

1. Ensure the `app/` directory is present alongside this repository (not tracked by git).
2. Create and populate any required environment files (for example `app/passwords.local.env` and/or `.env`) with your API keys, database URLs, and service endpoints.
3. Install app-specific dependencies (inside the same virtual environment or a separate one, depending on your setup).
4. Start the web server from inside `app/` (for example):

```bash
cd app
python run_server.py
```

or follow the instructions in `app/README.md` (if present) for the exact command (e.g. `uvicorn`, `fastapi`, or a Streamlit command).

Once the server is running, open the printed URL (typically the static address `http://localhost:8000`) in your browser to interact with the Sustainability SME Agent Expert.



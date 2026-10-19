---
title: Anforderungs-Check (Deploy-Test)
sdk: gradio
sdk_version: "6.30.0"
python_version: "3.11"
app_file: app.py
---

# Job Application Copilot

A local RAG (Retrieval-Augmented Generation) tool that matches a job posting
against your own application profile and produces a structured, evidence-based
requirement analysis using a large language model.

You paste in a job ad, and the tool tells you — for each requirement — whether
your profile covers it (met / partially / gap), backed by a concrete citation
from your own documents. It also suggests honest framing for the gaps, projects
worth highlighting, and likely interview questions.

The profile is built entirely from **your own PDF documents** (CV, transcripts,
papers, project descriptions). Nothing is hard-coded: swap the documents and the
tool works for a different person.

<!-- SCREENSHOT: add a screenshot of the Streamlit UI here -->

## How it works (the RAG pipeline)

The tool runs a three-step Retrieval-Augmented-Generation flow. First, Claude
reads the job ad and extracts the concrete requirements as a structured list
(**step A**). Then, for each requirement, the tool retrieves the top-k most
similar chunks from the profile using semantic search (**step B**). Finally,
Claude assesses each requirement using only those retrieved chunks as evidence,
marking it as met, partially met, or a gap (**step C**). The result is returned
as JSON and rendered in the UI as a traffic-light matrix.

The three letters of RAG map onto this directly: **Retrieval** is step B
(finding the right profile chunks), **Augmented** is enriching each requirement
with those chunks, and **Generation** is step C (Claude producing the
judgement). Because the judgement is restricted to the retrieved evidence, the
tool marks anything unsupported as an honest gap instead of inventing it.

Under the hood:

- **Retrieval (R):** PDFs are loaded, their text is extracted and split into
  chunks. Free-flowing text is chunked at natural boundaries; tabular grade data
  gets a dedicated treatment (one module per chunk). Chunks are embedded with
  `sentence-transformers` and searched via cosine similarity (top-k) using an
  in-memory NumPy index.
- **Augmented (A):** each extracted requirement is enriched with the profile
  chunks retrieved for it.
- **Generation (G):** Claude judges each requirement **only** on the supplied
  chunks. If there is no evidence, it is honestly marked as a gap — nothing is
  invented.

## Tech stack

Python · sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) ·
NumPy · Anthropic Claude API (REST) · pypdf · Streamlit

## Requirements

- Python 3.11+
- An Anthropic API key (the API is billed separately from any Claude.ai
  subscription — see https://console.anthropic.com)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key as an environment variable
# Windows (persistent):
setx ANTHROPIC_API_KEY "sk-ant-your-key"
# macOS/Linux:
export ANTHROPIC_API_KEY="sk-ant-your-key"
```

On Windows, after `setx` you need to open a **new** terminal for the key to be
available.

## Add your profile

Put your own PDF documents into the `profile/` folder — for example:

- your CV
- your transcript / list of grades
- papers or reports you have written
- short descriptions of your projects

The folder is git-ignored, so your personal documents are **never** committed or
uploaded. Only PDF files are read.

> The quality of the analysis depends directly on the documents you provide:
> the tool can only find evidence that actually exists in your profile.

## Usage

**Web UI (recommended):**

```bash
streamlit run app.py
```

Paste a job ad into the text box and click "Analysieren". The result is shown as
a traffic-light matrix (met / partially / gap) with citations, gap framing,
highlighted projects, and likely interview questions.

**Command line:**

```bash
python -m copilot.cli path/to/job_ad.txt
# or a PDF:
python -m copilot.cli path/to/job_ad.pdf
```

Each analysis makes two Claude API calls (extract requirements, then assess), so
it incurs small API costs.

## Inspecting the internals

Two diagnostic scripts help you see what the pipeline does:

```bash
python inspect_extraction.py   # shows the text extracted from each profile PDF
python inspect_retrieval.py    # shows top-k retrieval results for test questions
```

## Project structure

```
copilot/
  ingest.py      load PDFs, extract text, chunk (incl. grade-table handling)
  index.py       embeddings + retrieve(query, top_k) via cosine similarity
  llm.py         Claude calls (step A and C) + robust JSON parsing
  pipeline.py    wires the RAG flow together
  cli.py         command-line entry point
app.py           Streamlit UI
profile/         your own profile PDFs (git-ignored)
```

## Notes

- The retrieval index is held in memory (no vector database) — sufficient and
  exact for a personal profile of this size.
- The embedding model (~120 MB) is downloaded once on first run and cached
  locally.
- LLM assessments can vary slightly between runs; this is expected behaviour of
  the model and does not affect the tool's usefulness as a decision aid.

## Author

WxExLGHTZ — https://github.com/WxExLGHTZ

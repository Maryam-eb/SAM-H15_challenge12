# VisionVerse AI 

> An AI-powered visual intelligence  that transforms images into contextual, audience-aware marketing content.

## Overview

VisionVerse AI is an AI-powered visual designed to understand images and transform visual information into meaningful words.

The system combines computer vision, image captioning, visual feature extraction, candidate generation, ranking, and marketing-oriented reasoning to generate content based on:

- Visual content
- Industry
- Target audience
- Tone
- Language
- Marketing goals

The project is organized as a modular architecture so that vision models, reasoning components, datasets, and future fine-tuned models can be integrated independently.

## Team

Developed collaboratively by:

- **Name 1** Maryam Ebrahimi
- **Name 2** fatemeh zahra davari
- **Name 3** Mohadeseh Jarahzadeh

Developed as part of the **Innovers International Challenge (USA)**.



## Core Pipeline

```text
IMAGE
  │
  ▼
Visual Understanding
  │
  ├── BLIP
  ├── GIT
  └── ViT-GPT2 (optional)
  │
  ▼
Caption Candidates
  │
  ▼
Candidate Ranking
  │
  ▼
Marketing Context / Intent
  │
  ▼
Fusion / Generation Layer
  │
  ▼
Marketing Content
```

## Key Features

- Image understanding and caption generation
- Multiple vision-language model integrations
- Candidate generation and ranking
- Industry, audience, tone, language, and goal context
- Marketing-oriented output generation
- Modular FastAPI backend
- Frontend ↔ backend integration
- Dataset and training architecture
- Evaluation components
- Local model-weight support with configured Hub fallback

## Project Architecture

```text
VisionVerse_AI/
│
├── frontend/
├── backend/
│   ├── api/
│   ├── config/
│   ├── models/
│   ├── services/
│   ├── evaluation/
│   └── tests/
│
├── datasets/
│   └── marketing/
│
├── training/
│   └── configs/
│
├── weights/
│
├── README.md
├── INTEGRATION.md
├── requirements.txt
├── .env.example
├── .gitignore
├── run_server.py
├── start_server.bat
└── start_server.sh
```

## Output Schema

The marketing agent is designed around the following conceptual output:

```json
{
  "image": "...",
  "industry": "...",
  "audience": "...",
  "tone": "...",
  "language": "...",
  "goal": "...",
  "caption": "...",
  "hashtags": [],
  "cta": "...",
  "marketing_strategy": "..."
}
```

## Backend API

Main caption-generation endpoint:

```text
POST /api/generate-captions
```

The backend supports multipart image requests and image-URL based requests. Additional endpoints include:

```text
/health
/api/status
/docs
```

`/docs` provides the FastAPI Swagger interface.

See `INTEGRATION.md` for the request/response contract and integration details.

## Frontend

The frontend contains the main VisionVerse workspace together with Psychology and Custom Industry pages. The application can be served together with the FastAPI backend from one origin.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional environment configuration:

```bash
cp .env.example .env
```

Windows:

```text
start_server.bat
```

Linux/macOS:

```bash
./start_server.sh
```

Or:

```bash
python run_server.py
```

Then open:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

## Models and Weights

The repository reserves a dedicated `weights/` directory for VisionVerse model artifacts. Large model files should be managed with Git LFS or another artifact-storage solution rather than standard Git when appropriate.

If local fine-tuned weights are not present, the configured model loader can use its supported fallback behavior. See `INTEGRATION.md` for the current model-weight behavior.

## Dataset Architecture

```text
datasets/
├── captioning/
├── marketing/
├── intent/
└── fusion/
```

The current repository includes the marketing dataset schema/sample and the directory structure for future dataset expansion.

## Training Architecture

```text
training/
├── configs/
├── dataset_loader.py
├── train_captioning.py
├── train_intent.py
└── train_fusion.py
```

The training directory provides the project training architecture and pipeline scaffolding for captioning, intent, and fusion work. It should not be interpreted as a claim that all three training stages are already completed or that production fine-tuned models are included.

## Evaluation

The backend contains evaluation-related utilities and tests for model and API behavior. Evaluation is kept separate from inference so that model quality can be assessed independently from the production API.

## Security

The backend includes request validation and security-oriented handling for external image URLs and API requests, including URL validation, request limits, controlled model loading, and environment-based configuration.

## Current Status

### Implemented / Available

- Modular FastAPI backend
- Frontend ↔ backend integration
- Image caption-generation pipeline
- Vision-language model integrations
- Candidate generation and ranking
- Marketing-oriented output structure
- Dataset architecture
- Training architecture
- Evaluation structure
- API documentation
- Local model-weight support and configured fallback behavior

### Planned / In Development

- Larger domain-specific datasets
- Full captioning fine-tuning
- Marketing intent model training
- Fusion/reasoning model training
- Additional domain-specific models
- Full Psychology/EEG model integration
- Full Digital Twin model integration
- Expanded evaluation benchmarks

## Technology Stack

- Python
- FastAPI
- Uvicorn
- PyTorch
- Hugging Face Transformers
- BLIP
- GIT
- ViT-GPT2
- CLIP
- DETR
- HTML / CSS / JavaScript

## Project Goal

VisionVerse AI aims to move beyond generic image captioning by combining:

```text
Visual Understanding
        +
Context
        +
Audience
        +
Tone
        +
Marketing Goal
        ↓
Meaningful Marketing Content
```

The long-term goal is to build an intelligent visual marketing agent capable of adapting generated content to different industries, audiences, languages, and marketing objectives.

## Competition Project

Developed collaboratively as part of an international innovation challenge.

## License

This project is provided for research, educational, and competition purposes.

---

**VisionVerse AI — See the image. Understand the context. Create meaningful marketing intelligence.**

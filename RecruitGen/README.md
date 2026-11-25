
# RecruitGen AI

### A Multi-Model Generative AI Framework for Resume Enhancement, Candidate Synthesis, and Role Classification

RecruitGen is an AI-powered recruitment application designed to assist recruiters in the initial resume-screening process. It combines image enhancement, OCR, natural language processing, and multiple deep-learning architectures to analyse resumes, predict suitable job roles, calculate role-fit scores, and generate synthetic candidate profiles.

The application provides an interactive Streamlit interface for resume analysis, candidate insights, model demonstrations, screening history, and downloadable reports.

> RecruitGen is a decision-support system. It assists recruiters with initial screening, while the final hiring decision remains with the recruiter.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [AI Models](#ai-models)
- [Application Modules](#application-modules)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results and Evaluation](#results-and-evaluation)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Ethical Considerations](#ethical-considerations)

## Overview

Traditional recruitment involves manually reviewing large numbers of resumes to identify candidates with relevant skills, education, and experience. This process can be time-consuming and may lead to inconsistent screening decisions.

RecruitGen addresses this challenge through an integrated AI-based workflow that:

1. Processes resumes in different document formats.
2. Enhances scanned or degraded resume images.
3. Extracts text using OCR and document-processing techniques.
4. Identifies structured candidate features.
5. Predicts suitable job roles using a Transformer-based classifier.
6. Calculates role-fit scores using a Conditional Variational Autoencoder (CVAE).
7. Generates synthetic candidate profiles for training and evaluation experiments.
8. Produces candidate analysis reports.

## Key Features

- Resume upload and text-based resume analysis.
- Support for multiple document and image formats.
- AI-based resume image enhancement using Autoencoder and Diffusion models.
- OCR-based text extraction using RapidOCR.
- Candidate feature extraction and job-role classification.
- Role-fit scoring and best-match identification.
- Synthetic candidate generation using GAN and Hybrid approaches.
- Interactive charts and candidate analysis dashboards.
- Recruiter authentication and screening history.
- Admin panel for user and activity management.
- Downloadable PDF reports.
- Bulk resume processing through ZIP uploads and consolidated Excel reporting.
- AI Assistant for recruitment-related questions and candidate insights.
- Lazy model loading to reduce unnecessary model initialization.

## System Workflow

```text
Resume Upload
      |
      v
File Format Detection
      |
      v
Document Processing
      |
      v
Image Enhancement (when required)
      |
      v
OCR / Text Extraction
      |
      v
Candidate Feature Extraction
      |
      v
Transformer V3 - Job Role Classification
      |
      v
CVAE - Role-Fit Scoring
      |
      v
Candidate Analysis and Visualisation
      |
      v
PDF / Excel Report Generation
```

The Autoencoder and Diffusion models are used for image enhancement when required. The GAN and Hybrid pipelines primarily operate offline for synthetic candidate generation and are not part of the normal live resume-screening workflow.

## AI Models

RecruitGen integrates multiple AI architectures, each designed for a different task.

| Model | Purpose |
|---|---|
| Convolutional Autoencoder | Enhances scanned and degraded resume images |
| Conditional Diffusion Model | Provides an alternative image-restoration approach |
| RapidOCR | Extracts text from scanned documents and images |
| Transformer V3 (DistilBERT) | Understands resume text and predicts job roles |
| Conditional Variational Autoencoder (CVAE) | Evaluates candidate features under different job-role conditions and calculates role-fit scores |
| WGAN-GP | Generates synthetic candidate feature profiles |
| Hybrid Generation Pipeline | Combines GAN-based generation with language-model-assisted seeding |

### 1. Convolutional Autoencoder

The Autoencoder uses an encoder-decoder architecture to reconstruct cleaner resume images from degraded inputs.

The project includes two versions:

- **Autoencoder V1:** The original image-enhancement model.
- **Autoencoder V2:** An improved model fine-tuned using degraded real resume pages and a loss function that gives greater importance to text pixels.

Autoencoder V2 is the default image-enhancement model in RecruitGen.

### 2. Diffusion Model

The conditional Diffusion model provides an alternative approach to resume image restoration.

It learns to reverse a noise-adding process and uses deterministic DDIM sampling during inference.

The model is optional because it requires more processing time than the Autoencoder.

### 3. Transformer V3 — DistilBERT

Transformer V3 uses a fine-tuned DistilBERT architecture for resume text understanding and job-role classification.

The supported job roles are:

- AI/ML Engineer
- Cloud Engineer
- Data Analyst
- General IT
- Software Engineer

The model uses contextual relationships between words to understand resume content and predict a suitable job role.

### 4. Conditional Variational Autoencoder (CVAE)

The CVAE evaluates candidate feature profiles under different job-role conditions.

It combines a reconstruction-based score with predefined role-relevance weights to calculate role-fit scores.

The scores help identify candidate profiles that match different job roles. They are intended as screening assistance rather than validated hiring decisions.

### 5. WGAN-GP

The project uses a Wasserstein Generative Adversarial Network with Gradient Penalty to generate synthetic candidate profiles.

The Generator creates artificial candidate features, while the Critic evaluates the generated data against real training data.

The synthetic profiles are labelled as synthetic and are primarily used for training and evaluation experiments.

### 6. Hybrid Synthetic Candidate Generation

The Hybrid pipeline combines GAN-based generation with a language-model-assisted seeding process.

It explores an additional approach to synthetic candidate generation and enables comparison with other generative models.

## Application Modules

| Module | Description |
|---|---|
| Dashboard | Provides an overview of the application |
| Resume Analyzer | Uploads, processes, and analyses resumes |
| Autoencoder | Demonstrates Autoencoder-based image enhancement |
| Diffusion | Demonstrates Diffusion-based image restoration |
| CVAE/VAE | Provides role-fit scoring and generative model demonstrations |
| GAN | Demonstrates synthetic candidate generation |
| Transformer V3 | Demonstrates resume classification and job-role prediction |
| AI Assistant | Supports recruitment-related questions and candidate insights |
| History | Displays previous screening activity |
| Admin Panel | Supports user management and activity monitoring |

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Application | Streamlit |
| Deep Learning | PyTorch, TensorFlow, Keras |
| NLP | Hugging Face Transformers, DistilBERT |
| OCR | RapidOCR |
| Document Processing | PyMuPDF |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, SciPy |
| Visualisation | Plotly |
| Database | SQLite |
| PDF Reporting | ReportLab |

## Project Structure

The main project files include:

```text
RecruitGen_AI/
├── app.py
├── README.md
├── requirements.txt
├── requirements-training.txt
├── .gitignore
├── core/
├── data/
├── models/
├── RecruitGen Models/
│   └── models/
│       ├── autoencoder/
│       ├── cvae/
│       └── transformer_v3/
├── pdf_downloads/
├── Sample Input/
└── sample_data/
```

The project also includes supporting application modules and model assets.

Virtual environments, Python cache files, temporary files, and private configuration files should not be committed to the repository.

## Installation and Setup

### Prerequisites

- Python 3.11 or another Python version compatible with the project's dependencies.
- Git.
- A GitHub account.
- A compatible environment for running the Streamlit application.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and project folder name.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to access RecruitGen.

### Model and Dataset Requirements

Some models require trained checkpoints and supporting assets. Ensure that the required model files and data are available before using their corresponding modules.

The training dependencies are listed separately in `requirements-training.txt`.

## Usage

1. Launch RecruitGen.
2. Create an account or log in.
3. Open the Resume Analyzer.
4. Upload a supported resume or paste resume text.
5. Allow the application to process the document.
6. Review the extracted candidate features.
7. View the predicted job role and role-fit scores.
8. Explore the analysis charts and candidate insights.
9. Download the PDF report or use the bulk-processing features where required.

The application also provides separate pages for demonstrating the individual AI models.

## Dataset

RecruitGen uses the Kaggle Resume Dataset for resume classification and related model experiments.

Different models require different training data:

- Transformer V3 uses resume text and job-role labels.
- Autoencoder and Diffusion models use clean and degraded resume-image pairs.
- CVAE uses candidate feature profiles and job-role conditions.
- GAN and Hybrid pipelines use candidate feature data for synthetic generation.

The project uses preprocessing and training, validation, and testing splits to prepare and evaluate the models.

## Results and Evaluation

The project evaluates each model according to its intended purpose.

- **Autoencoder V1:** Image enhancement sometimes increased OCR errors.
- **Autoencoder V2:** Improved image enhancement and became the default model.
- **Diffusion:** Achieved the best image-quality measurements, but required more processing time. Its OCR performance was statistically indistinguishable from Autoencoder V2 in the reported comparison.
- **CVAE:** Implemented successfully for role-fit scoring, but its standalone classification performance was weak. The shipped checkpoint requires further validation or retraining before standalone classification use.
- **Transformer V3:** Achieved strong classification performance and became the primary job-role classifier.
- **WGAN-GP:** Generated synthetic candidate data with better distributional similarity than the CVAE baseline.
- **Hybrid:** Explored an additional approach to synthetic candidate generation.

The results highlight that model complexity alone does not guarantee better performance. Each model must be evaluated according to its intended task.

## Limitations

- Image-enhancement evaluation uses a relatively small test set.
- Artificially degraded images may not represent every real-world scanned document or photograph.
- Rule-based feature extraction may miss important candidate information.
- CVAE role-fit scores have not been validated against real recruiter judgements.
- Synthetic candidate generation requires further evaluation for the application's job roles.
- Fairness and bias require further investigation.
- Reported training and timing results are based on CPU execution and may differ on GPU hardware.

## Future Scope

- Support additional languages, including Indian regional languages.
- Integrate with Applicant Tracking Systems (ATS).
- Evaluate image-enhancement models on more real-world resume images.
- Improve candidate feature extraction.
- Validate role-fit scores using recruiter feedback.
- Perform more extensive fairness and bias testing.
- Improve synthetic candidate generation for specific job roles.
- Evaluate model performance on GPU hardware.

## Ethical Considerations

RecruitGen is designed to assist recruiters rather than make hiring decisions independently.

- **Human involvement:** Recruiters retain responsibility for final hiring decisions.
- **Transparency:** Role-fit scores use predefined role-relevance weights.
- **Fairness:** The project focuses on skills and experience rather than names, gender, age, or photographs.
- **Privacy:** Synthetic profiles are labelled, and uploaded resumes are subject to the application's retention and deletion process.
- **Security:** The application includes authentication and role-based access permissions.
- **Responsible evaluation:** The project reports limitations and negative experimental findings alongside positive results.

## Author and Project Information

**Project:** RecruitGen AI  
**Category:** Generative AI, Deep Learning, Natural Language Processing, Recruitment Automation  
**Application:** Streamlit

Developed as an academic project to explore the integration of multiple AI architectures for resume analysis and recruitment support.

## Disclaimer

RecruitGen is an academic project and a decision-support tool. Its predictions and role-fit scores should not be treated as definitive measures of a candidate's ability or suitability. Recruiters should independently review candidate qualifications and make final hiring decisions.

---

**Live Demo:** To be added after deployment.

**GitHub Repository:** To be added after upload.

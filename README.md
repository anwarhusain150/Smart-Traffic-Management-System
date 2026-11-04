# AI-Based Smart Traffic Management and Congestion Prediction System

> CSE570 – Machine Learning with Python | Skill Based Assignment (Academic Task 2)
> Author: **Anwar Husain** ([@anwarhusain150](https://github.com/anwarhusain150))

## Problem Statement
Traffic congestion wastes time and fuel and increases pollution. Traffic authorities need to know
**how much traffic will be on a road at a given hour** and **how congested it will be**, so they can plan
signal timings, diversions and police deployment in advance.

This project builds two machine-learning modules on **historical** hourly traffic data:

| Module | ML Task | Target |
|---|---|---|
| 1. Traffic Volume Prediction | Regression | `traffic_volume` (vehicles per hour) |
| 2. Congestion Level Classification | Multi-class Classification | `congestion_level` – Low / Medium / High |

## Dataset
**Metro Interstate Traffic Volume** – UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
(Hogue, J. (2019). Metro Interstate Traffic Volume. UCI ML Repository. https://doi.org/10.24432/C5X60B)

Hourly westbound traffic volume on I-94 (Minneapolis–St Paul, USA), 2012–2018, with weather and holiday features.

## Project Structure
```
Smart-Traffic-Management-System/
├── Dataset/        raw and cleaned data
├── Notebooks/      Jupyter notebooks (cleaning, EDA, models)
├── src/            reusable Python scripts
├── models/         trained models (created by training script)
├── app/            Streamlit dashboard
├── Images/         saved plots / figures
├── Reports/        final project report
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Progress
- [x] Project setup and GitHub repository
- [ ] Data collection and cleaning
- [ ] Exploratory Data Analysis
- [ ] Feature engineering
- [ ] Regression models
- [ ] Classification models
- [ ] Model evaluation
- [ ] Streamlit dashboard
- [ ] Report

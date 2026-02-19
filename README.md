# Formula-1-Telemetry-Performance-Modeling-Race-Strategy-Optimization

*Work in Progress — Ongoing Research Project*
##  Overview
This repository contains an ongoing research project focused on modeling Formula 1 race performance using telemetry-derived and race-level features. The current phase investigates how qualifying position, team strength, and driver form relate to finishing position and top-10 outcomes using statistical analysis and logistic regression. Future work will extend this framework using advanced machine learning and deep learning models.

## Research Motivation
Qualifying performance is widely believed to be one of the strongest determinants of Formula 1 race outcomes, yet race-day execution, team strength, and driver form introduce substantial variability. This project aims to quantify these relationships, assess their statistical significance, and build predictive models that move beyond descriptive analysis toward actionable race performance forecasting.

## Key Research Questions (Current Phase)
- Quantify the relationship between qualifying position and finishing position
- Evaluate the impact of driver form and team strength on race outcomes
- Develop predictive models for race performance and top-10 finishes
- Extend analysis toward advanced machine learning and deep learning approaches

## Current Progress
- Exploratory data analysis and correlation studies
- Statistical evaluation of key predictors (qualifying position, driver form, team strength)
- Binary logistic regression model to predict top-10 finishes

## Repository Contents
- `analysis/` — PDFs summarizing exploratory analysis, modeling ideas, and research notes
- `interactive/` — Interactive HTML visualizations examining relationships between qualifying position, driver form, team strength, and finishing position
  - Interactive plots are hosted via GitHub Pages:
  - [Qualifying Position vs Finishing Position](https://ropillay.github.io/Formula-1-Telemetry-Performance-Modeling-Race-Strategy-Optimization/interactive/F1_2024_Qual_vs_Finish.html)
  - [Driver Form vs Finishing Position](https://ropillay.github.io/Formula-1-Telemetry-Performance-Modeling-Race-Strategy-Optimization/interactive/DriverForm_vs_Finish_PerRace.html)
  - [Team Strength vs Finishing Position](https://ropillay.github.io/Formula-1-Telemetry-Performance-Modeling-Race-Strategy-Optimization/interactive/TeamStrength_vs_Finish_PerRace.html)
- `src/` — Python scripts used for data collection, feature engineering, statistical analysis, and model development

## Data
- Official Formula 1 timing and session data via the **FastF1 API**
- Race results, qualifying data, and race-level context features
- Local caching enabled for reproducibility

## Planned Extensions
- Plackett- Luce Ranking Model
- Support Vector Machines (SVMs)
- Correlation-driven feature selection
- Deep learning models for race outcome prediction
- Integration of richer telemetry and strategy features

## Research Paper
A full academic-style LaTeX manuscript documenting the methodology, analysis, and results will be linked here upon completion.

## Contact
**Rohan Pillay**  
Statistics & Data Science — UC Davis

**Professor Maxime Guiffo Pouokam**  
UC Davis Department of Statistics

# ML-based Circuit Delay Prediction for CMOS Timing Analysis

## Motivation
Accurate circuit delay estimation using SPICE simulations is computationally expensive.
This project explores machine learning models to predict CMOS inverter timing parameters
across voltage corners, enabling faster early-stage timing analysis.

## Problem Statement
Given circuit parameters such as capacitive load and supply voltage, predict:
- Rise Delay
- Fall Delay
- Average Delay (DAVG)

using supervised machine learning models.

## Approach
- Feature engineering on capacitive load and voltage parameters
- Baseline Linear Regression vs Gradient Boosting comparison
- Per-voltage-corner modeling
- 5-fold cross-validation for robustness
- Derived average delay (DAVG) from predicted rise/fall delays

## Results
- Gradient Boosting consistently outperformed Linear Regression
- Achieved strong R² scores with RMSE reduced by ~28% compared to linear regression
- Demonstrated feasibility of ML-based timing estimation

## Tech Stack
- Python  
- scikit-learn  
- pandas, numpy  
- matplotlib

## Project Structure

circuit-delay-prediction/
 src/
  metrics.py
  models.py
  train.py
  utils.py
 data/
  data.csv
 results/
 requirements.txt
 README.md

## How to Run
```bash
pip install -r requirements.txt  
python src/train.py

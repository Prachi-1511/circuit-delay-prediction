# ML-based Circuit Delay Prediction for CMOS Timing Analysis

## Motivation
Accurate circuit delay estimation using SPICE simulations is computationally expensive.
This project explores machine learning models to predict CMOS inverter timing parameters
across voltage corners, enabling faster early-stage timing analysis.

## Approach
- Feature engineering on capacitive load and voltage parameters
- Baseline Linear Regression vs Gradient Boosting comparison
- Per-voltage-corner modeling
- 5-fold cross-validation for robustness
- Derived average delay (DAVG) from predicted rise/fall delays

## Results
- Gradient Boosting consistently outperformed Linear Regression
- Achieved strong R² scores with low RMSE across voltage corners
- Demonstrated feasibility of ML-based timing estimation

## Tech Stack
Python, scikit-learn, pandas, numpy, matplotlib

## How to Run
pip install -r requirements.txt  
python src/train.py
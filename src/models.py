from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def get_gbr_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42
        ))
    ])

def get_lr_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from models import get_gbr_model, get_lr_model
from metrics import evaluate
from utils import add_features, plot_actual_vs_pred, ensure_dirs

# ================= CONFIG =================
DATA_PATH = "data/data.csv"

COL_CLOAD = "CLOAD (pF)"
COL_VMAX  = "VOUT_MAX (V)"
COL_TPHL  = "TPHL (ns)"
COL_TPLH  = "TPLH (ns)"
COL_DAVG  = "DAVG (ns)"
COL_VMIN  = "VOUT_MIN (V)"

TARGETS = [COL_TPHL, COL_TPLH, COL_VMIN]
# =========================================


def main():
    ensure_dirs()
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    rows = []
    corners = sorted(df[COL_VMAX].unique())

    for v in corners:
        corner_df = df[df[COL_VMAX] == v].copy()

        # -------- Features --------
        X = corner_df[[COL_CLOAD, "CLOAD_sq", COL_VMAX]]
        idx = corner_df.index.values

        Xtr, Xte, idx_tr, idx_te = train_test_split(
            X, idx, test_size=0.2, random_state=42, shuffle=True
        )

        train_df = corner_df.loc[idx_tr]
        test_df  = corner_df.loc[idx_te]

        # Store predictions for DAVG derivation
        preds_tr = {}
        preds_te = {}

        for target in TARGETS:
            ytr = train_df[target].values
            yte = test_df[target].values

            models = {
                "LinearRegression": get_lr_model(),
                "GradientBoosting": get_gbr_model()
            }

            for model_name, model in models.items():
                model.fit(Xtr, ytr)

                yhat_tr = model.predict(Xtr)
                yhat_te = model.predict(Xte)

                tr_metrics = evaluate(ytr, yhat_tr)
                te_metrics = evaluate(yte, yhat_te)

                rows.append({
                    "VOUT_MAX": v,
                    "Target": target,
                    "Model": model_name,
                    "Split": "Train",
                    **tr_metrics
                })

                rows.append({
                    "VOUT_MAX": v,
                    "Target": target,
                    "Model": model_name,
                    "Split": "Test",
                    **te_metrics
                })

                # -------- Cross-validation --------
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_r2 = cross_val_score(
                    model, X, corner_df[target], cv=cv, scoring="r2"
                )

                rows.append({
                    "VOUT_MAX": v,
                    "Target": target,
                    "Model": model_name,
                    "Split": "CV",
                    "R2": cv_r2.mean(),
                    "RMSE": np.nan,
                    "MAE": np.nan
                })

                # -------- Save predictions for DAVG (GBR only) --------
                if model_name == "GradientBoosting":
                    preds_tr[target] = yhat_tr
                    preds_te[target] = yhat_te

                # -------- Plot --------
                plot_actual_vs_pred(
                    yte,
                    yhat_te,
                    f"{model_name} | {target} | Vmax={v}",
                    f"results/plots/{model_name}_{target}_V{v}.png"
                )

        # -------- Derived DAVG (Engineering logic) --------
        if COL_TPHL in preds_tr and COL_TPLH in preds_tr:
            d_tr_true = train_df[COL_DAVG].values
            d_te_true = test_df[COL_DAVG].values

            d_tr_pred = 0.5 * (preds_tr[COL_TPHL] + preds_tr[COL_TPLH])
            d_te_pred = 0.5 * (preds_te[COL_TPHL] + preds_te[COL_TPLH])

            d_tr_metrics = evaluate(d_tr_true, d_tr_pred)
            d_te_metrics = evaluate(d_te_true, d_te_pred)

            rows.append({
                "VOUT_MAX": v,
                "Target": "DAVG (derived)",
                "Model": "GradientBoosting",
                "Split": "Train",
                **d_tr_metrics
            })

            rows.append({
                "VOUT_MAX": v,
                "Target": "DAVG (derived)",
                "Model": "GradientBoosting",
                "Split": "Test",
                **d_te_metrics
            })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv("results/metrics.csv", index=False)

    print("Training completed successfully.")
    print("Metrics saved to results/metrics.csv")
    print("Plots saved to results/plots/")


if __name__ == "__main__":
    main()
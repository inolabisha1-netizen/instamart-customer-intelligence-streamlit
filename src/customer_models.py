"""Part B: rubric-aligned RFM segmentation and leakage-safe lapse modelling."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SEED = 42

NUMERIC = ["recency", "frequency", "monetary", "avg_discount",
           "cancellation_rate", "avg_delivery", "age", "tenure_days"]
CATEGORICAL = ["favourite_category", "city", "gender"]
FEATURES = NUMERIC + CATEGORICAL


def load():
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")
    return customers, orders


def rfm_and_segments(k=4):
    """Return RFM table, k=2..8 diagnostics, segment profile, PCA ratios."""
    if k not in (3, 4, 5):
        raise ValueError("The rubric requires final k in {3, 4, 5}.")
    _, orders = load()
    completed = orders.loc[
        orders["status"].eq("completed") & orders["order_ts"].lt(SNAPSHOT)
    ]
    rfm = (completed.groupby("customer_id")
           .agg(last_order=("order_ts", "max"),
                frequency=("order_id", "nunique"),
                monetary=("order_amount", "sum"))
           .reset_index())
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm = rfm.drop(columns="last_order")
    assert rfm["customer_id"].is_unique
    assert rfm["recency"].ge(0).all()

    scaled = StandardScaler().fit_transform(
        rfm[["recency", "frequency", "monetary"]]
    )
    diagnostics = []
    for n in range(2, 9):
        km = KMeans(n_clusters=n, n_init=10, random_state=SEED)
        labels = km.fit_predict(scaled)
        diagnostics.append({"k": n, "inertia": km.inertia_,
                            "silhouette": silhouette_score(scaled, labels)})
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    rfm["segment_id"] = km.fit_predict(scaled)
    profile = (rfm.groupby("segment_id")
               .agg(size=("customer_id", "size"),
                    recency=("recency", "mean"),
                    frequency=("frequency", "mean"),
                    monetary=("monetary", "mean"))
               .reset_index())

    # Highest mean monetary is Premium, regardless of arbitrary cluster ID.
    premium = int(profile.loc[profile["monetary"].idxmax(), "segment_id"])
    remaining = profile.loc[profile["segment_id"].ne(premium)]
    frequent = int(remaining.loc[remaining["frequency"].idxmax(), "segment_id"])
    remaining = remaining.loc[remaining["segment_id"].ne(frequent)]
    at_risk = int(remaining.loc[remaining["recency"].idxmax(), "segment_id"])
    labels = {premium: "Premium", frequent: "Frequent", at_risk: "At Risk"}
    for sid in profile["segment_id"]:
        labels.setdefault(int(sid), "Regular")
    actions = {
        "Premium": "Offer VIP benefits and personalised retention outreach.",
        "Frequent": "Reward repeat purchases and offer cross-category bundles.",
        "At Risk": "Send a timely reactivation offer.",
        "Regular": "Encourage the next purchase with relevant recommendations.",
    }
    rfm["segment_name"] = rfm["segment_id"].map(labels)
    rfm["is_premium"] = rfm["segment_id"].eq(premium)
    profile["segment_name"] = profile["segment_id"].map(labels)
    profile["recommended_action"] = profile["segment_name"].map(actions)

    pca = PCA(n_components=2).fit(scaled)
    OUT.mkdir(parents=True, exist_ok=True)
    rfm[["customer_id", "recency", "frequency", "monetary",
         "segment_id", "segment_name", "is_premium"]].to_csv(
             OUT / "customer_segments.csv", index=False)
    return rfm, pd.DataFrame(diagnostics), profile, pca.explained_variance_ratio_


def make_cohort():
    """Features use pre-cutoff orders only; September orders define label only."""
    customers, orders = load()
    hist = orders.loc[orders["order_ts"].lt(CUTOFF)].copy()
    done = hist.loc[hist["status"].eq("completed")].copy()
    cohort = (done.groupby("customer_id")
              .agg(last_order=("order_ts", "max"),
                   frequency=("order_id", "nunique"),
                   monetary=("order_amount", "sum"),
                   avg_discount=("discount_pct", "mean"),
                   avg_delivery=("delivery_minutes", "mean"))
              .reset_index())
    cohort["recency"] = (CUTOFF - cohort["last_order"]).dt.days
    cohort = cohort.drop(columns="last_order")

    final_status = hist.loc[hist["status"].isin(["completed", "cancelled"])]
    counts = (final_status.groupby("customer_id")["status"]
              .agg(total="count",
                   cancelled=lambda s: s.eq("cancelled").sum())
              .reset_index())
    cohort = cohort.merge(counts, on="customer_id", how="left",
                          validate="one_to_one")
    cohort["cancellation_rate"] = cohort["cancelled"] / cohort["total"]

    products = pd.read_csv(CLEAN / "products.csv")
    cats = (done.merge(products[["product_id", "category"]],
                       on="product_id", validate="many_to_one")
            .groupby(["customer_id", "category"]).size()
            .rename("n").reset_index())
    favourite = (cats.sort_values(["customer_id", "n", "category"],
                                  ascending=[True, False, True])
                 .drop_duplicates("customer_id")
                 [["customer_id", "category"]]
                 .rename(columns={"category": "favourite_category"}))
    cohort = cohort.merge(favourite, on="customer_id", how="left",
                          validate="one_to_one")
    cohort = cohort.merge(customers, on="customer_id", how="left",
                          validate="one_to_one")
    cohort["signup_date"] = pd.to_datetime(cohort["signup_date"])
    cohort["tenure_days"] = (CUTOFF - cohort["signup_date"]).dt.days

    september = orders.loc[
        orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(SNAPSHOT)
        & orders["status"].eq("completed"), "customer_id"
    ]
    cohort["lapsed"] = (~cohort["customer_id"].isin(september)).astype(int)
    assert cohort["customer_id"].is_unique
    assert cohort["recency"].ge(0).all()
    assert cohort["lapsed"].isin([0, 1]).all()
    return cohort


def make_pipeline(estimator):
    """All learned imputations, scaling and encoding are inside CV."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def scores(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]).tolist(),
    }


def build_lapse_model():
    cohort = make_cohort()
    # Raw missing values are allowed at this point: imputers learn from
    # training folds only. No learned preprocessing sees test rows.
    X, y = cohort[FEATURES].copy(), cohort["lapsed"].copy()
    assert X.shape[1] == len(FEATURES)
    assert y.nunique() == 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    assert set(X_train.index).isdisjoint(X_test.index)

    configurations = {
        "logistic": (
            LogisticRegression(max_iter=1000),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "forest": (
            RandomForestClassifier(n_estimators=100, random_state=SEED),
            {"model__max_depth": [3, 5, 8],
             "model__min_samples_leaf": [5, 10, 20]},
        ),
    }
    searches, metrics = {}, {}
    for name, (estimator, grid) in configurations.items():
        search = GridSearchCV(
            make_pipeline(estimator), grid, cv=5, scoring="f1",
            n_jobs=-1, error_score="raise")
        search.fit(X_train, y_train)
        searches[name] = search

    # Only after both searches finish do we evaluate the untouched test set.
    for name, search in searches.items():
        pred = search.predict(X_test)
        metrics[name] = {
            "best_params": search.best_params_,
            "cv_f1": float(search.best_score_),
            **scores(y_test, pred),
        }

    # Choose using training cross-validation only; tie goes to forest.
    selected_name = max(
        searches, key=lambda name: (
            searches[name].best_score_, name == "forest"))
    selected = searches[selected_name].best_estimator_

    baseline_pred = np.zeros(len(y_test), dtype=int)
    baseline = {
        "accuracy": float(accuracy_score(y_test, baseline_pred)),
        **scores(y_test, baseline_pred),
    }

    # Training-only resampling: keep all original rows and duplicate
    # minority (lapsed=1) training rows until class counts match.
    train_0 = y_train.index[y_train.eq(0)]
    train_1 = y_train.index[y_train.eq(1)]
    if len(train_1) >= len(train_0):
        raise ValueError("Expected lapsed=1 to be the minority class.")
    rng = np.random.RandomState(SEED)
    extra = rng.choice(train_1.to_numpy(),
                       size=len(train_0) - len(train_1), replace=True)
    balanced_indices = np.concatenate([X_train.index.to_numpy(), extra])
    X_balanced = X_train.loc[balanced_indices].reset_index(drop=True)
    y_balanced = y_train.loc[balanced_indices].reset_index(drop=True)
    assert y_balanced.value_counts().nunique() == 1
    assert len(X_balanced) == len(y_balanced)

    best_logistic = searches["logistic"].best_estimator_
    balanced_logistic = make_pipeline(
        LogisticRegression(
            max_iter=1000,
            C=searches["logistic"].best_params_["model__C"]))
    balanced_logistic.fit(X_balanced, y_balanced)
    balance_experiment = {
        "balanced_class_counts": y_balanced.value_counts().sort_index().to_dict(),
        "original": scores(y_test, best_logistic.predict(X_test)),
        "oversampled": scores(y_test, balanced_logistic.predict(X_test)),
    }

    # Interpret fitted pipelines using transformed feature names.
    forest = searches["forest"].best_estimator_
    feature_names = forest.named_steps["prep"].get_feature_names_out()
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": forest.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).head(5)
    logistic = searches["logistic"].best_estimator_
    log_names = logistic.named_steps["prep"].get_feature_names_out()
    coefficients = pd.DataFrame({
        "feature": log_names,
        "coefficient": logistic.named_steps["model"].coef_[0],
    })
    # Three easy-to-interpret numeric signs.
    coefficient_signs = coefficients.loc[
        coefficients["feature"].isin([
            "num__recency", "num__frequency", "num__monetary"
        ])].copy()
    coefficient_signs["sign"] = np.where(
        coefficient_signs["coefficient"].ge(0), "positive", "negative")

    # Scaler learned from training rows only, not the full cohort.
    fitted_scaler = selected.named_steps["prep"].named_transformers_[
        "num"].named_steps["scaler"]
    frequency_position = NUMERIC.index("frequency")
    train_frequency_mean = float(fitted_scaler.mean_[frequency_position])
    cohort_frequency_mean = float(cohort["frequency"].mean())
    assert np.isclose(train_frequency_mean,
                      X_train["frequency"].mean())
    # Confirm transformed features have no missing values.
    transformed_train = selected.named_steps["prep"].transform(X_train)
    transformed_test = selected.named_steps["prep"].transform(X_test)
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_test).all()

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, OUT / "lapse_model.joblib")
    reloaded = joblib.load(OUT / "lapse_model.joblib")
    assert np.array_equal(
        selected.predict(X_test), reloaded.predict(X_test))

    # Rubric explicitly requests POST-HOC whole-window Premium flags.
    # These are ONLY for illustrative filtering, never model features.
    segments, _, _, _ = rfm_and_segments(k=4)
    premium_ids = set(segments.loc[
        segments["is_premium"], "customer_id"])
    probs = selected.predict_proba(X_test)[
        :, list(selected.classes_).index(1)]
    retention = pd.DataFrame({
        "customer_id": cohort.loc[X_test.index, "customer_id"].to_numpy(),
        "lapse_probability": probs,
        "actually_lapsed": y_test.to_numpy(),
    })
    retention = (retention.loc[
        retention["customer_id"].isin(premium_ids)]
        .sort_values("lapse_probability", ascending=False)
        .head(10).reset_index(drop=True))
    retention.to_csv(OUT / "retention_list.csv", index=False)

    return {
        "cohort_rows": int(len(cohort)),
        "lapsed": int(y.sum()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_accuracy": baseline["accuracy"],
        "baseline": baseline,
        "models": metrics,
        "selected_model": selected_name,
        "selected_test_f1": metrics[selected_name]["f1"],
        "frequency_mean_training": train_frequency_mean,
        "frequency_mean_full_cohort": cohort_frequency_mean,
        "balance_experiment": balance_experiment,
        "forest_top_features": importance,
        "logistic_coefficient_signs": coefficient_signs,
        "retention_list": retention,
        "retention_truly_lapsed": int(retention["actually_lapsed"].sum()),
        "retention_limitation": (
            "Premium uses whole-window RFM including September; "
            "the retention list is illustrative, not leakage-free evaluation."
        ),
    }


if __name__ == "__main__":
    rfm, diagnostics, profile, ratios = rfm_and_segments(k=4)
    print("\n=== RFM ===")
    print("Customers:", len(rfm))
    print("Recency median/min/max:",
          rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
    print("Frequency mean/max:", rfm.frequency.mean(), rfm.frequency.max())
    print("Monetary mean/max:", rfm.monetary.mean(), rfm.monetary.max())
    print("\nK diagnostics:\n", diagnostics.to_string(index=False))
    print("\nSegment profiles and actions:\n", profile.to_string(index=False))
    print("\nPCA explained variance:", ratios, "total:", ratios.sum())
    print("Interpretation: Increasing K can only reduce or maintain inertia; "
          "K=8's lower inertia alone does not establish business usefulness. "
          "K=4 balances the elbow, silhouette and actionable segment profiles.")
    summary = build_lapse_model()
    print("\n=== LAPSE MODEL ===")
    for key, value in summary.items():
        if isinstance(value, pd.DataFrame):
            print(f"\n{key}:\n{value.to_string(index=False)}")
        else:
            print(f"\n{key}: {value}")
    print("\nFrequency means differ because the scaler was fitted on "
          "training rows only, while the cohort mean includes test rows.")
    print("\nNo encoder, scaler, or resampler saw test rows during fitting; "
          "features use only pre-September orders and exclude segment labels.")

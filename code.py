import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

FILE_NAME = "dataset.csv"
df = pd.read_csv(FILE_NAME)

target = "Outcome"

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Bagging(DT)": BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=200,
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

def run_case(features, case_name):
    X = df[features].copy()
    y = df[target].astype(int).copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rows = []
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred) * 100
        prec = precision_score(y_test, pred, zero_division=0) * 100
        rec = recall_score(y_test, pred, zero_division=0) * 100
        f1 = f1_score(y_test, pred, zero_division=0) * 100
        mae = mean_absolute_error(y_test, pred) * 100

        rows.append([name, acc, prec, rec, f1, mae])

    result_df = pd.DataFrame(rows, columns=[
        "Model", "Accuracy (%)", "Precision (%)", "Recall (%)",
        "F1-score (%)", "MAE (%)"
    ]).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print(case_name)
    print("Features:", features)
    print("="*80)
    display(result_df)
    return result_df

features_case1 = ["Age", "Gender", "Area", "AreaType", "HouseType", "NS1"]
res_case1 = run_case(features_case1, "CASE-1: Environmental + NS1")

features_case2 = ["Age", "Gender", "Area", "AreaType", "HouseType", "IgG"]
res_case2 = run_case(features_case2, "CASE-2: Environmental + IgG")

features_case3 = ["Age", "Gender", "Area", "AreaType", "HouseType", "IgM"]
res_case3 = run_case(features_case3, "CASE-3: Environmental + IgM")
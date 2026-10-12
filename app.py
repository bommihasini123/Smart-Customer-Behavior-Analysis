import os
import uuid
import math
from itertools import combinations
from collections import Counter

import pandas as pd
import numpy as np

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash
)

from werkzeug.utils import secure_filename

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================
# 1. FLASK CONFIGURATION
# =========================================

app = Flask(__name__)

app.secret_key = "customer_data_mining_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CHART_FOLDER = os.path.join(BASE_DIR, "static", "charts")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv", "arff"}


# =========================================
# 2. DATASET FUNCTIONS
# =========================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def read_dataset(filepath):

    extension = filepath.rsplit(".", 1)[1].lower()

    if extension == "csv":

        try:
            df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="latin1")

    elif extension == "arff":

        from scipy.io import arff

        data, metadata = arff.loadarff(filepath)

        df = pd.DataFrame(data)

        for column in df.columns:

            if df[column].dtype == object:

                df[column] = df[column].apply(
                    lambda value:
                    value.decode("utf-8", errors="replace")
                    if isinstance(value, bytes)
                    else value
                )

    else:
        raise ValueError("Only CSV and ARFF files are supported.")

    df.columns = [str(column).strip() for column in df.columns]

    if df.empty:
        raise ValueError("The dataset is empty.")

    if len(df.columns) < 2:
        raise ValueError("Dataset must contain at least two columns.")

    return df


def get_current_dataset():

    filename = session.get("dataset_file")

    if not filename:
        return None

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.isfile(filepath):
        return None

    return read_dataset(filepath)


def make_preview(df, limit=15):

    preview = df.head(limit).copy()

    preview = preview.replace(
        [np.inf, -np.inf],
        np.nan
    )

    preview = preview.astype(object).where(
        pd.notna(preview),
        ""
    )

    return {
        "columns": preview.columns.tolist(),
        "rows": preview.values.tolist()
    }


# =========================================
# 3. DATA PREPROCESSING
# =========================================

def clean_dataset(df):

    cleaned = df.copy()

    cleaned = cleaned.drop_duplicates()

    cleaned = cleaned.dropna(how="all")

    for column in cleaned.columns:

        if cleaned[column].isnull().any():

            if pd.api.types.is_numeric_dtype(cleaned[column]):

                median_value = cleaned[column].median()

                if pd.notna(median_value):
                    cleaned[column] = cleaned[column].fillna(
                        median_value
                    )

            else:

                mode_values = cleaned[column].mode(dropna=True)

                if not mode_values.empty:
                    cleaned[column] = cleaned[column].fillna(
                        mode_values.iloc[0]
                    )

    return cleaned


# =========================================
# 4. MACHINE LEARNING PREPROCESSOR
# =========================================

def create_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(transformers=transformers)


# =========================================
# 5. CLASSIFICATION ALGORITHMS
# =========================================

def run_classification(df):

    results = []

    if len(df) < 6:
        return [{
            "name": "Classification",
            "status": "At least 6 rows are recommended."
        }]

    # Last column is the prediction target.
    target = df.columns[-1]

    model_df = df.dropna(subset=[target]).copy()

    if len(model_df) < 6:
        return [{
            "name": "Classification",
            "status": "Not enough valid target records."
        }]

    X = model_df.drop(columns=[target])
    y = model_df[target].astype(str)

    if X.empty or y.nunique() < 2:
        return [{
            "name": "Classification",
            "status": "Target must contain at least two classes."
        }]

    stratify_values = (
        y if y.value_counts().min() >= 2 else None
    )

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify_values
        )

    except Exception as error:

        return [{
            "name": "Classification",
            "status": str(error)
        }]

    algorithms = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "Naive Bayes": GaussianNB(),

        "KNN": KNeighborsClassifier(
            n_neighbors=min(3, len(X_train))
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    for name, classifier in algorithms.items():

        try:

            preprocessor = create_preprocessor(X)

            # GaussianNB requires dense input.
            if name == "Naive Bayes":

                model = Pipeline([
                    ("preprocessing", preprocessor),

                    (
                        "to_dense",
                        FunctionTransformer(
                            lambda data:
                            data.toarray()
                            if hasattr(data, "toarray")
                            else data,
                            validate=False
                        )
                    ),

                    ("classifier", classifier)
                ])

            else:

                model = Pipeline([
                    ("preprocessing", preprocessor),
                    ("classifier", classifier)
                ])

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            ) * 100

            labels = sorted(
                set(y_test.tolist())
                | set(predictions.tolist())
            )

            matrix = confusion_matrix(
                y_test,
                predictions,
                labels=labels
            ).tolist()

            results.append({
                "name": name,
                "status": "Success",
                "accuracy": round(float(accuracy), 2),
                "target": str(target),
                "matrix": matrix
            })

        except Exception as error:

            results.append({
                "name": name,
                "status": str(error)
            })

    return results


# =========================================
# 6. DATA VISUALIZATION
# =========================================

def create_charts(df):

    charts = []

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column for column in df.columns
        if column not in numeric_columns
    ]

    # Histogram
    if numeric_columns:

        column = numeric_columns[0]

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if not values.empty:

            plt.figure(figsize=(7, 4))

            plt.hist(values, bins=10, edgecolor="black")

            plt.title("Histogram - " + column)
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.tight_layout()

            filename = "hist_" + uuid.uuid4().hex + ".png"

            plt.savefig(os.path.join(CHART_FOLDER, filename))
            plt.close()

            charts.append({
                "title": "Histogram - " + column,
                "url": url_for(
                    "static",
                    filename="charts/" + filename
                )
            })

    # Bar Chart
    if categorical_columns:

        column = categorical_columns[0]

        counts = (
            df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(10)
        )

        plt.figure(figsize=(7, 4))

        counts.plot(kind="bar")

        plt.title("Bar Chart - " + column)
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.tight_layout()

        filename = "bar_" + uuid.uuid4().hex + ".png"

        plt.savefig(os.path.join(CHART_FOLDER, filename))
        plt.close()

        charts.append({
            "title": "Bar Chart - " + column,
            "url": url_for(
                "static",
                filename="charts/" + filename
            )
        })

    return charts


# =========================================
# 7. K-MEANS CLUSTERING
# =========================================

def run_clustering(df):

    try:

        X = df.dropna(axis=1, how="all").copy()

        if len(X) < 2 or X.empty:
            return {"status": "Not enough data for clustering."}

        transformed = create_preprocessor(X).fit_transform(X)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        if transformed.shape[1] == 0:
            return {"status": "No usable features were found."}

        k = min(3, len(X))

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(transformed)

        output = X.copy()
        output["Cluster"] = labels

        counts = Counter(labels.tolist())

        if transformed.shape[1] >= 2:

            coordinates = PCA(
                n_components=2
            ).fit_transform(transformed)

            x_values = coordinates[:, 0]
            y_values = coordinates[:, 1]

        else:

            x_values = transformed[:, 0]
            y_values = np.zeros(len(transformed))

        plt.figure(figsize=(7, 4))

        plt.scatter(x_values, y_values, c=labels)

        plt.title("K-Means Customer Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.tight_layout()

        filename = "cluster_" + uuid.uuid4().hex + ".png"

        plt.savefig(os.path.join(CHART_FOLDER, filename))
        plt.close()

        return {
            "status": "Success",
            "number_of_clusters": k,
            "counts": dict(counts),
            "preview": make_preview(output),
            "url": url_for(
                "static",
                filename="charts/" + filename
            )
        }

    except Exception as error:

        plt.close("all")

        return {"status": str(error)}


# =========================================
# 8. APRIORI ASSOCIATION RULES
# =========================================

def run_apriori(df):

    try:

        transactions = []

        for _, row in df.iterrows():

            items = set()

            for column, value in row.items():

                if pd.isna(value):
                    continue

                items.add(str(column) + "=" + str(value))

            transactions.append(items)

        total = len(transactions)

        if total < 2:
            return {
                "status": "At least two rows are required.",
                "rules": []
            }

        minimum_support = 0.10
        minimum_confidence = 0.50

        minimum_count = max(
            1,
            math.ceil(minimum_support * total)
        )

        item_counts = Counter()

        for transaction in transactions:

            for item in transaction:
                item_counts[frozenset([item])] += 1

        frequent = {
            itemset: count
            for itemset, count in item_counts.items()
            if count >= minimum_count
        }

        unique_items = sorted({
            item
            for itemset in frequent
            for item in itemset
        })

        for size in [2, 3]:

            if len(unique_items) < size:
                break

            for group in combinations(unique_items, size):

                itemset = frozenset(group)

                if itemset in frequent:
                    continue

                count = sum(
                    itemset.issubset(transaction)
                    for transaction in transactions
                )

                if count >= minimum_count:
                    frequent[itemset] = count

        rules = []

        for itemset, count in frequent.items():

            if len(itemset) < 2:
                continue

            for left_size in range(1, len(itemset)):

                for left in combinations(
                    sorted(itemset),
                    left_size
                ):

                    antecedent = frozenset(left)
                    consequent = itemset - antecedent

                    antecedent_count = frequent.get(
                        antecedent,
                        0
                    )

                    if antecedent_count == 0:
                        continue

                    confidence = count / antecedent_count

                    if confidence < minimum_confidence:
                        continue

                    support = count / total

                    consequent_count = sum(
                        consequent.issubset(transaction)
                        for transaction in transactions
                    )

                    consequent_support = consequent_count / total

                    lift = (
                        confidence / consequent_support
                        if consequent_support > 0
                        else 0
                    )

                    rules.append({
                        "if": ", ".join(sorted(antecedent)),
                        "then": ", ".join(sorted(consequent)),
                        "support": round(support * 100, 2),
                        "confidence": round(confidence * 100, 2),
                        "lift": round(lift, 2)
                    })

        rules.sort(
            key=lambda rule: rule["confidence"],
            reverse=True
        )

        return {
            "status": "Success" if rules else "No association rules found.",
            "rules": rules[:30]
        }

    except Exception as error:

        return {"status": str(error), "rules": []}


# =========================================
# 9. DASHBOARD
# =========================================

def generate_dashboard(df):

    cleaned = clean_dataset(df)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),

        "missing_count": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),

        "cleaned_row_count": len(cleaned),
        "cleaned_missing_count": int(cleaned.isnull().sum().sum()),
        "cleaned_duplicate_count": int(cleaned.duplicated().sum()),

        "preview": make_preview(df),
        "cleaned_preview": make_preview(cleaned),

        "classification_results": run_classification(cleaned),
        "charts": create_charts(cleaned),
        "clustering": run_clustering(cleaned),
        "association": run_apriori(cleaned)
    }


# =========================================
# 10. HOME PAGE
# =========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        title="Smart Customer Behavior Analysis Using Data Mining"
    )


# =========================================
# 11. DATASET UPLOAD
# =========================================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        if "file" not in request.files:

            flash("Please select a dataset.", "error")

            return redirect(url_for("upload"))

        file = request.files["file"]

        if file.filename == "":

            flash("Please choose a file.", "error")

            return redirect(url_for("upload"))

        if not allowed_file(file.filename):

            flash("Upload only CSV or ARFF files.", "error")

            return redirect(url_for("upload"))

        original_name = secure_filename(file.filename)

        unique_name = uuid.uuid4().hex + "_" + original_name

        filepath = os.path.join(UPLOAD_FOLDER, unique_name)

        try:

            file.save(filepath)

            df = read_dataset(filepath)

            session["dataset_file"] = unique_name
            session["original_filename"] = original_name

            dashboard = generate_dashboard(df)

            return render_template(
                "upload.html",
                dashboard=dashboard,
                dataset_name=original_name
            )

        except Exception as error:

            if os.path.exists(filepath):
                os.remove(filepath)

            flash("Dataset error: " + str(error), "error")

            return redirect(url_for("upload"))

    df = get_current_dataset()

    if df is not None:

        dashboard = generate_dashboard(df)

        return render_template(
            "upload.html",
            dashboard=dashboard,
            dataset_name=session.get("original_filename", "Dataset")
        )

    return render_template(
        "upload.html",
        dashboard=None,
        dataset_name=None
    )


# =========================================
# 12. OTHER MENU LINKS
# =========================================

@app.route("/preprocessing")
@app.route("/classification")
@app.route("/visualization")
@app.route("/clustering")
@app.route("/association")
def other_pages():

    return redirect(url_for("upload"))


# =========================================
# 13. RUN APPLICATION
# =========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
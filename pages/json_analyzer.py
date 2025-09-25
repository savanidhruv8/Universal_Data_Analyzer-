import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import io
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score

# ---------- Cleaning Function ----------
def clean_data(df, imputation_methods=None, remove_outliers=False, outlier_method=None, encoding_method=None, 
               encoding_column=None, remove_duplicates=True, handle_datetime_nulls=False, dtype_conversions=None,
               columns_to_remove=None):
    summary = {
        "initial_shape": df.shape,
        "initial_nulls": df.isnull().sum(),
        "removed_duplicate_rows": 0,
        "removed_duplicate_columns": 0,
        "dropped_columns": [],
        "user_removed_columns": [],  # Field for user-specified removed columns
        "converted_dates": [],
        "converted_to_numeric": [],
        "skipped_outlier_removal": [],
        "standardized_types": {},
        "replaced_nulls": 0,
        "imputed_columns": [],
        "outliers_detected": {},
        "outliers_removed": 0,
        "encoded_columns": [],
        "datetime_nulls_handled": [],
        "dtype_conversion_success": [],
        "dtype_conversion_failed": []
    }

    # Strip whitespace from headers
    df.columns = df.columns.str.strip()

    # Replace common string nulls
    df.replace(["NA", "na", "n/a", "N/A", "null", "NULL", "NaN", "nan", ""], np.nan, inplace=True)
    summary["replaced_nulls"] = df.isnull().sum().sum() - summary["initial_nulls"].sum()

    # Strip and lowercase string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Convert date-like columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='raise')
                summary["converted_dates"].append(col)
            except:
                continue

    # Identify columns with numeric content
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    potential_numeric_cols = []
    for col in df.select_dtypes(include='object').columns:
        try:
            temp = pd.to_numeric(df[col], errors='coerce')
            if temp.notnull().sum() > 0:
                df[col] = temp
                if temp.dtype in ['float64', 'int64', 'Float64', 'Int64']:
                    potential_numeric_cols.append(col)
                    summary["converted_to_numeric"].append(col)
                    numeric_cols.append(col)
                else:
                    summary["skipped_outlier_removal"].append(f"{col}: Non-numeric after conversion")
            else:
                summary["skipped_outlier_removal"].append(f"{col}: No valid numeric values")
        except:
            summary["skipped_outlier_removal"].append(f"{col}: Failed numeric conversion")
            continue

    # Handle null/empty cells in datetime columns
    if handle_datetime_nulls:
        datetime_cols = df.select_dtypes(include='datetime64').columns
        for col in datetime_cols:
            if df[col].isnull().any():
                df[col] = df[col].where(df[col].notnull(), np.nan)
                summary["datetime_nulls_handled"].append(col)

    # Remove user-specified columns
    if columns_to_remove:
        valid_columns = [col for col in columns_to_remove if col in df.columns]
        if valid_columns:
            df.drop(columns=valid_columns, inplace=True)
            summary["user_removed_columns"] = valid_columns

    # Remove duplicate rows and columns
    if remove_duplicates:
        before_rows = df.shape[0]
        df = df.drop_duplicates()
        after_rows = df.shape[0]
        summary["removed_duplicate_rows"] = before_rows - after_rows

        before_cols = df.shape[1]
        df = df.T.drop_duplicates().T
        after_cols = df.shape[1]
        summary["removed_duplicate_columns"] = before_cols - after_cols

    # Remove empty and constant columns
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1 or df[col].isnull().all():
            summary["dropped_columns"].append(col)
    df.drop(columns=summary["dropped_columns"], inplace=True)

    # Detect and remove outliers
    if remove_outliers and numeric_cols:
        before_rows = df.shape[0]
        for col in numeric_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64', 'Float64', 'Int64']:
                if outlier_method == "Z-score":
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers = df[col][z_scores >= 3]
                    if not outliers.empty:
                        summary["outliers_detected"][col] = outliers.tolist()
                    df = df[df[col].isin(df[col][z_scores < 3]) | df[col].isna()]
                elif outlier_method == "IQR":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
                    if not outliers.empty:
                        summary["outliers_detected"][col] = outliers.tolist()
                    df = df[(df[col].between(lower_bound, upper_bound)) | df[col].isna()]
            elif col in df.columns:
                summary["skipped_outlier_removal"].append(f"{col}: Non-numeric dtype ({df[col].dtype})")
        summary["outliers_removed"] = before_rows - df.shape[0]

    # Impute null values in selected numeric columns
    if imputation_methods:
        for col, method in imputation_methods.items():
            if col in df.columns and df[col].isnull().any():
                summary["imputed_columns"].append(f"{col} ({method})")
                if method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                if df[col].dropna().apply(float.is_integer).all():
                    df[col] = df[col].astype('int64')
                    summary["standardized_types"][col] = 'int64'

    # Encoding categorical columns
    if encoding_method and encoding_column:
        if encoding_method == "Label Encoding":
            le = LabelEncoder()
            df[encoding_column] = le.fit_transform(df[encoding_column].astype(str))
            summary["encoded_columns"].append(f"{encoding_column} (Label Encoding)")
        elif encoding_method == "One-Hot Encoding":
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_cols = pd.DataFrame(ohe.fit_transform(df[[encoding_column]]),
                                        columns=[f"{encoding_column}_{cat}" for cat in ohe.categories_[0]])
            df = pd.concat([df.drop(columns=[encoding_column]), encoded_cols], axis=1)
            summary["encoded_columns"].append(f"{encoding_column} (One-Hot Encoding)")

    # Apply user-specified data type conversions
    if dtype_conversions:
        for col, target_dtype in dtype_conversions.items():
            try:
                if target_dtype == 'int':
                    if df[col].isnull().any():
                        summary["dtype_conversion_failed"].append(f"{col}: Contains null values")
                        continue
                    if not df[col].apply(float.is_integer).all():
                        summary["dtype_conversion_failed"].append(f"{col}: Contains non-integer values")
                        continue
                    df[col] = df[col].astype('int64')
                else:
                    df[col] = df[col].astype(target_dtype)
                summary["dtype_conversion_success"].append(f"{col}: {target_dtype}")
            except Exception as e:
                summary["dtype_conversion_failed"].append(f"{col}: {str(e)}")

    # Infer and standardize remaining data types
    df = df.convert_dtypes()
    for col in df.columns:
        if col not in summary["standardized_types"]:
            summary["standardized_types"][col] = str(df[col].dtype)

    summary["final_shape"] = df.shape
    summary["final_nulls"] = df.isnull().sum()

    return df, summary

# ---------- ML Algorithm Subtype Detection ----------
def detect_ml_subtype(df, algorithm_type, target_column=None):
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include='object').columns
    n_samples, n_features = df.shape

    if algorithm_type == "Classification":
        if target_column and target_column in df.columns:
            n_classes = df[target_column].nunique()
            if n_classes == 2:
                return "Logistic Regression" if n_features < 50 else "Gradient Boosting Classifier"
            elif n_classes <= 10:
                return "Random Forest Classifier" if n_samples > 1000 else "Support Vector Classifier"
            else:
                return "K-Nearest Neighbors Classifier"
        return "Random Forest Classifier"

    elif algorithm_type == "Regression":
        return "Linear Regression" if n_features < 20 else "Random Forest Regressor" if n_samples > 1000 else "Gradient Boosting Regressor"

    elif algorithm_type == "Clustering":
        return "K-Means" if n_samples < 10000 else "DBSCAN" if n_features > 10 else "Agglomerative Clustering"

    return "No subtype detected"

# ---------- Streamlit App ----------
st.title("🧹 JSON Analyzer")

# Initialize session state for dataset and CSV
if 'df' not in st.session_state:
    st.session_state.df = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None

# File uploader for JSON only
uploaded_file = st.file_uploader("Upload a JSON file", type=["json"], key="file_uploader")

if uploaded_file:
    # Load and process JSON file
    try:
        json_data = json.load(uploaded_file)
        st.session_state.df = pd.json_normalize(json_data)
        st.write("✅ JSON file loaded and converted to DataFrame.")
        # Convert to CSV in-memory
        csv_buffer = io.StringIO()
        st.session_state.df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.session_state.csv_data = csv_buffer.getvalue()
        st.download_button("📥 Download Converted CSV",
                          st.session_state.csv_data,
                          file_name="converted_dataset.csv",
                          mime="text/csv")
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        st.stop()

if st.session_state.df is not None:
    df = st.session_state.df.copy()

    st.subheader("📌 Original Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Null values per column:**")
    st.write(df.isnull().sum())
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.dataframe(df.head())

    # User input for cleaning options
    st.subheader("⚙️ Cleaning Options")
    remove_outliers = st.checkbox("Detect and remove outliers")
    outlier_method = None
    if remove_outliers:
        outlier_method = st.radio("Outlier removal method", ["Z-score", "IQR"])
    remove_duplicates = st.checkbox("Remove duplicate rows and columns", value=True)
    handle_datetime_nulls = st.checkbox("Replace null/empty datetime cells with NaN")
    
    # Option to remove specific columns
    columns_to_remove = st.multiselect("Select columns to remove (optional)", df.columns.tolist())

    # Imputation options for numeric columns
    st.subheader("📈 Imputation for Numeric Columns")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    potential_numeric_cols = []
    for col in df.select_dtypes(include='object').columns:
        try:
            temp = pd.to_numeric(df[col], errors='coerce')
            if temp.notnull().sum() > 0:
                potential_numeric_cols.append(col)
        except:
            continue
    all_numeric_cols = numeric_cols + potential_numeric_cols

    imputation_methods = {}
    if all_numeric_cols:
        num_imputations = st.number_input("Number of columns to impute", min_value=0, max_value=len(all_numeric_cols), step=1)
        for i in range(num_imputations):
            col = st.selectbox(f"Select column {i+1} for imputation", all_numeric_cols, key=f"impute_col_{i}")
            method = st.selectbox(f"Select imputation method for {col}", ["Mean", "Median", "Mode"], key=f"impute_method_{i}")
            imputation_methods[col] = method
    else:
        st.write("No numeric columns available for imputation.")

    # Encoding options
    categorical_cols = df.select_dtypes(include='object').columns
    encoding_column = None
    encoding_method = None
    if len(categorical_cols) > 0:
        encoding_column = st.selectbox("Select column for encoding", ["None"] + list(categorical_cols))
        if encoding_column != "None":
            encoding_method = st.radio("Encoding method", ["Label Encoding", "One-Hot Encoding"])

    # Data type conversion options
    st.subheader("🔄 Data Type Conversion")
    dtype_conversions = {}
    num_conversions = st.number_input("Number of columns to convert", min_value=0, max_value=len(df.columns), step=1)
    for i in range(num_conversions):
        col = st.selectbox(f"Select column {i+1} for type conversion", df.columns, key=f"col_{i}")
        target_dtype = st.selectbox(f"Select target data type for {col}",
                                    ["int", "float", "string", "datetime"], key=f"dtype_{i}")
        dtype_conversions[col] = target_dtype

    # Machine Learning Algorithm Selection
    st.subheader("🤖 Machine Learning Algorithm Selection")
    algorithm_type = st.selectbox("Select ML Algorithm Type",
                                  ["None", "Classification", "Regression", "Clustering"])
    target_column = None
    if algorithm_type != "None":
        if algorithm_type in ["Classification", "Regression"]:
            available_columns = [col for col in df.columns if col not in columns_to_remove]
            if available_columns:
                target_column = st.selectbox("Select target column", available_columns)
            else:
                st.error("No columns available for target selection after column removal.")
                st.stop()
        suggested_subtype = detect_ml_subtype(df, algorithm_type, target_column)
        st.write(f"**Suggested Subtype:** {suggested_subtype}")

    if st.button("🚀 Clean and Analyze Dataset"):
        cleaned_df, summary = clean_data(
            df.copy(),
            imputation_methods,
            remove_outliers,
            outlier_method,
            encoding_method,
            encoding_column if encoding_column != "None" else None,
            remove_duplicates,
            handle_datetime_nulls,
            dtype_conversions,
            columns_to_remove
        )

        # Update session state with cleaned dataset
        st.session_state.df = cleaned_df
        csv_buffer = io.StringIO()
        cleaned_df.to_csv(csv_buffer, index=False)
        st.session_state.csv_data = csv_buffer.getvalue()

        st.subheader("✅ Cleaned Dataset Overview")
        st.write(f"**Shape:** {summary['final_shape'][0]} rows × {summary['final_shape'][1]} columns")
        st.write("**Null values per column after cleaning:**")
        st.write(summary["final_nulls"])
        st.write("**Data Types after standardization:**")
        st.write(summary["standardized_types"])

        st.subheader("🔍 Summary of Changes")
        st.write(f"Removed duplicate rows: **{summary['removed_duplicate_rows']}**")
        st.write(f"Removed duplicate columns: **{summary['removed_duplicate_columns']}**")
        st.write(f"Replaced null-like strings with NaN: **{summary['replaced_nulls']}** replacements")
        st.write(f"Converted date columns: **{summary['converted_dates']}**")
        st.write(f"Converted to numeric columns: **{summary['converted_to_numeric']}**")
        if summary["user_removed_columns"]:
            st.write(f"User-specified removed columns: **{summary['user_removed_columns']}**")
        if summary["skipped_outlier_removal"]:
            st.write(f"Skipped outlier removal for columns: **{summary['skipped_outlier_removal']}**")
        st.write(f"Dropped empty or constant columns: **{summary['dropped_columns']}**")
        if summary["imputed_columns"]:
            st.write(f"Imputed columns: **{summary['imputed_columns']}**")
        if summary["outliers_detected"]:
            st.write("**Outliers detected in columns:**")
            for col, outliers in summary["outliers_detected"].items():
                st.write(f"- {col}: {outliers}")
        st.write(f"Outliers removed: **{summary['outliers_removed']}** rows")
        st.write(f"Encoded columns: **{summary['encoded_columns']}**")
        if summary["datetime_nulls_handled"]:
            st.write(f"Datetime columns with nulls replaced: **{summary['datetime_nulls_handled']}**")
        if summary["dtype_conversion_success"]:
            st.write(f"Successful data type conversions: **{summary['dtype_conversion_success']}**")
        if summary["dtype_conversion_failed"]:
            st.write(f"Failed data type conversions: **{summary['dtype_conversion_failed']}**")

        st.subheader("🧾 Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head())

        # Download cleaned CSV
        st.download_button("📥 Download Cleaned CSV",
                          st.session_state.csv_data,
                          file_name="cleaned_dataset.csv",
                          mime="text/csv")

        # Download cleaned JSON
        json_buffer = io.StringIO()
        cleaned_df.to_json(json_buffer, orient='records', lines=True)
        st.download_button("📥 Download Cleaned JSON",
                          json_buffer.getvalue(),
                          file_name="cleaned_dataset.json",
                          mime="application/json")

        # ML Algorithm Evaluation
        if algorithm_type != "None" and target_column:
            st.subheader("🤖 ML Algorithm Evaluation")
            try:
                X = cleaned_df.drop(columns=[target_column])
                y = cleaned_df[target_column]
                X = X.select_dtypes(include=np.number)  # Use only numeric features
                if X.empty:
                    st.error("No numeric features available for ML evaluation.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    if algorithm_type == "Classification":
                        if suggested_subtype == "Logistic Regression":
                            model = LogisticRegression(random_state=42)
                        elif suggested_subtype == "Random Forest Classifier":
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                        elif suggested_subtype == "Support Vector Classifier":
                            model = SVC(random_state=42)
                        elif suggested_subtype == "Gradient Boosting Classifier":
                            model = GradientBoostingClassifier(random_state=42)
                        elif suggested_subtype == "K-Nearest Neighbors Classifier":
                            model = KNeighborsClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.write("**Classification Report:**")
                        st.text(classification_report(y_test, y_pred))

                    elif algorithm_type == "Regression":
                        if suggested_subtype == "Linear Regression":
                            model = LinearRegression()
                        elif suggested_subtype == "Random Forest Regressor":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif suggested_subtype == "Gradient Boosting Regressor":
                            model = GradientBoostingRegressor(random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")

                    elif algorithm_type == "Clustering":
                        if suggested_subtype == "K-Means":
                            model = KMeans(n_clusters=3, random_state=42)
                        elif suggested_subtype == "DBSCAN":
                            model = DBSCAN(eps=0.5, min_samples=5)
                        elif suggested_subtype == "Agglomerative Clustering":
                            model = AgglomerativeClustering(n_clusters=3)
                        labels = model.fit_predict(X)
                        st.write("**Cluster Labels (first 10):**")
                        st.write(labels[:10])
            except Exception as e:
                st.error(f"Error in ML evaluation: {str(e)}")
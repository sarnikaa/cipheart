import streamlit as st
import tenseal as ts
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

#################################################### STREAMLIT SETUP ###############################################################
st.title("Cipheart: Heart Disease Classification with Homomorphic Encryption")
st.markdown("""
    Cipheart uses machine learning to predict heart disease using the UCI Heart Disease dataset. 
    It provides options to choose between Logistic Regression and Naive Bayes models, 
    and offers the use of CKKS homomorphic encryption for data privacy.
""")

numerical_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']

# PRE-PROCESSING
def preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    missing_values_before = df.isnull().sum()
    num_duplicates_before = df.duplicated().sum()

    df = df.drop_duplicates()

    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0]).astype('category')

    if 'num' not in df.columns:
        raise ValueError("Column 'num' is not found in the data.")

    df['num_binary'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    data_encoded = df.copy()
    data_encoded['sex'] = data_encoded['sex'].map({'Male': 1, 'Female': 0})
    data_encoded = pd.get_dummies(data_encoded,drop_first=True)

    bool_columns = data_encoded.select_dtypes(include=['bool']).columns
    data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

    scaler = StandardScaler()
    numerical_features = data_encoded.select_dtypes(include=['number']).columns
    data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

    missing_values_after = df.isnull().sum()

    X = data_encoded.drop(columns=['num', 'num_binary'])
    y = data_encoded['num_binary'].astype(int)
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test,data_encoded, missing_values_before, missing_values_after, num_duplicates_before

# DETECT AND REMOVE OUTLIERS 
def remove_outliers(df, numerical_features, multiplier=1.5):
    for column in numerical_features:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        median = df[column].median()
        df.loc[df[column] < lower_bound, column] = median
        df.loc[df[column] > upper_bound, column] = median

    return df


def plot_boxplot(data, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

def plot_metrics(y_true, y_pred,y_proba=None, model_name=""):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
    }

#  METRICS PLOT 

    st.subheader(f"Metrics for {model_name}")
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title(f'{model_name} - Classification Metrics')
    plt.ylim(0, 1)
    plt.ylabel('Score')
    st.pyplot(plt)

    st.subheader(f"Confusion Matrix for {model_name}")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='d')
    st.pyplot(plt)

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot(plt)

# SIDEBAR 
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if st.checkbox("Pre-process Data"):
    if uploaded_file is not None:
        try:
            X_train, X_test, y_train, y_test,data_encoded, missing_values_before, missing_values_after, num_duplicates_before = preprocess_data(uploaded_file)
            st.write("Data preprocessed successfully.")

            st.write("Processed Data:")
            st.dataframe(data_encoded)

            st.write("Missing Values (Before Cleaning):")
            st.write(missing_values_before)  
            st.write("Number of Duplicates (Before Cleaning):", num_duplicates_before)

            st.write("Missing Values (After Cleaning):")
            st.write(missing_values_after)  
            st.write("Number of Duplicates (After Cleaning):", data_encoded.duplicated().sum())

            st.write("Columns in Processed Data:")
            st.write(data_encoded.columns.tolist())

            target_column_name = 'num'  
            if target_column_name not in data_encoded.columns:
                st.error(f"Target column '{target_column_name}' not found in data.")
            else:
                X = data_encoded.drop(target_column_name, axis=1)
                y = data_encoded[target_column_name]

            if st.checkbox("Remove Outliers"):
                before_outlier_removal = data_encoded.copy()
                data_encoded = remove_outliers(data_encoded, numerical_features)

                st.subheader("Boxplot Comparison Before and After Outlier Removal")
                plot_boxplot(before_outlier_removal[numerical_features], "Boxplot Before Outlier Removal")
                plot_boxplot(data_encoded[numerical_features], "Boxplot After Outlier Removal")

        except Exception as e:
            st.error(f"An error occurred during preprocessing: {e}")

# LOGISTIC REGRESSION
if st.sidebar.checkbox("Use Logistic Regression", value=False):

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("The uploaded CSV file is empty. Please upload a valid file.")
            else:
                st.write(df.head()) 
                if 'num' in df.columns:  
                    df['num_binary'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
                    df_encoded = pd.get_dummies(df, drop_first=True)

                    X = df_encoded.drop(['num', 'num_binary'], axis=1)
                    y = df_encoded['num_binary']  
               
                    imputer = SimpleImputer(strategy='mean')  
                    X_imputed = imputer.fit_transform(X) 

                    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
                    log_reg.fit(X_train_scaled, y_train)
                    y_pred = log_reg.predict(X_test_scaled)

                    st.subheader("Logistic regression results")

                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                    y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]  
                    plot_metrics(y_test, y_pred, y_proba, model_name="Logistic Regression")

                else:
                    st.error("The 'num' column does not exist in the uploaded file. Please check your CSV.")
        
        except pd.errors.EmptyDataError:
            st.error("Uploaded file is empty. Please upload a valid CSV file.")
        except pd.errors.ParserError as e:
            st.error(f"Parsing error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# NAIVE BAYES
if st.sidebar.checkbox("Use Naive Bayes"):
    if uploaded_file is not None:

        X = df_encoded.drop(['num', 'num_binary'], axis=1)  
        y = df_encoded['num_binary']  
    
        imputer = SimpleImputer(strategy='mean')  
        X_imputed = imputer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        nb_model = GaussianNB()
        nb_model.fit(X_train_scaled, y_train)

        y_pred = nb_model.predict(X_test_scaled)

        st.subheader("Naive Bayes results")

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        y_proba = nb_model.predict_proba(X_test_scaled)[:, 1]  
        plot_metrics(y_test, y_pred, y_proba, model_name="Naive Bayes")

encrypted_X_test = None
encrypted_y_proba_lr = []
encrypted_y_proba_nb = []

# CKKS HOMOMORPHIC ENCRYPTION
if st.sidebar.checkbox("Enable CKKS Encryption"):
    st.sidebar.subheader("Select Models for Encrypted Data")
    use_encrypted_lr = st.sidebar.checkbox("Use Logistic Regression on Encrypted Data")
    use_encrypted_nb = st.sidebar.checkbox("Use Naive Bayes on Encrypted Data")


    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    encrypted_X_test = [ts.ckks_vector(context, row) for row in X_test] 
    st.write(encrypted_X_test) 
  
# LOG REG ON ENCRYPTED DATA
    if use_encrypted_lr:
            encrypted_weights = [ts.ckks_vector(context, coef) for coef in log_reg.coef_.tolist()]
            encrypted_bias = ts.ckks_vector(context, log_reg.intercept_.tolist())

            print('Encrypted weights')
            print(encrypted_weights)

            print('\n\nEncrypted Bias')
            print(encrypted_bias)

            log_reg = LogisticRegression(multi_class='ovr') 
            log_reg.fit(X_train, y_train)

            encrypted_weights = [ts.ckks_vector(context, log_reg.coef_[0].tolist())]  
            encrypted_bias = ts.ckks_vector(context, [log_reg.intercept_[0]])       

            print('Encrypted weights:', encrypted_weights)
            print('Encrypted Bias:', encrypted_bias)

            def encrypted_predict(encrypted_sample, encrypted_weights, encrypted_bias):
                encrypted_dot_product = encrypted_sample.dot(encrypted_weights[0])
                encrypted_result = encrypted_dot_product + encrypted_bias
                return encrypted_result

            encrypted_predictions = [encrypted_predict(sample, encrypted_weights, encrypted_bias) for sample in encrypted_X_test]

            decrypted_predictions = [enc_pred.decrypt() for enc_pred in encrypted_predictions]
            for i, pred in enumerate(decrypted_predictions): 
                print(f"Decrypted Prediction for sample {i+1}: {pred[0]}") 

            threshold = 0.5

            binary_predictions = []

            for i, pred in enumerate(decrypted_predictions): 
                prediction_score = 1 / (1 + np.exp(-pred[0])) 
                classification = 1 if prediction_score >= threshold else 0 
                binary_predictions.append(classification)
                print(f"Sample {i+1}: Score = {prediction_score:.4f} -> Classification = {'Heart Disease' if classification == 1 else 'No Heart Disease'}")

            accuracy = accuracy_score(y_test, binary_predictions)
            print(f"\nAccuracy: {accuracy*100:.2f}")
        
            st.subheader("Logistic regression results for encrypted data")
            st.write(f"Logistic Regression Accuracy on encrypted data: {accuracy * 100:.2f}%")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            plot_metrics(y_test, y_pred, model_name="Logistic regression on encrypted data")


    encrypted_X_test = [ts.ckks_vector(context, sample.tolist()) for sample in X_test]

# NAIVE BAYES ON ENCRYPTED DATA
    if use_encrypted_nb:
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2 ** 40
        context.generate_galois_keys()

        encrypted_X_test = [ts.ckks_vector(context, sample.tolist()) for sample in X_test]  

        means = nb_model.theta_  
        variances = nb_model.var_  

        encrypted_means = [ts.ckks_vector(context, mean.tolist()) for mean in means]
        encrypted_variances = [ts.ckks_vector(context, variance.tolist()) for variance in variances]

        print("Encrypted means:", [mean.decrypt() for mean in encrypted_means])
        print("Encrypted variances:", [variance.decrypt() for variance in encrypted_variances])

        def encrypted_nb_predict(encrypted_sample, encrypted_means, encrypted_variances):
     
            decrypted_sample = encrypted_sample.decrypt()

            log_probs = []
            for mean, variance in zip(encrypted_means, encrypted_variances):
                
                decrypted_mean = np.array(mean.decrypt())
                decrypted_variance = np.array(variance.decrypt())

                log_prob = -0.5 * np.sum((decrypted_sample - decrypted_mean) ** 2 / decrypted_variance)
                log_probs.append(log_prob)

            return np.argmax(log_probs)

        encrypted_predictions = [encrypted_nb_predict(sample, encrypted_means, encrypted_variances) for sample in encrypted_X_test]

        for i, pred in enumerate(encrypted_predictions):
            print(f"Decrypted Prediction for sample {i + 1}: {pred}")
            print(f"Sample {i + 1}: Score = {pred:.4f} -> Classification = {'Heart Disease' if pred == 1 else 'No Heart Disease'}")

        decrypted_predictions = np.array(encrypted_predictions)  
        accuracy = accuracy_score(y_test, decrypted_predictions)  
        print(f"Accuracy: {accuracy:.4f}") 

        st.subheader("Naive Bayes results for encrypted data")
        st.write(f"Naive Bayes Accuracy on encrypted data: {accuracy * 100:.2f}%")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        plot_metrics(y_test, y_pred, model_name="Naive Bayes on encrypted data")


    





# Credit Card Fraud Detection

This project uses **machine learning ensembles** to detect fraudulent credit card transactions from an imbalanced dataset. By leveraging **Random Forest, XGBoost, LightGBM**, and **stacking classifiers**, we address the challenges of **class imbalance** and deliver a robust solution.

---

## Folder Structure

## Folder Structure

| Folder/File            |         Description                                     |
|------------------------|---------------------------------------------------------|
| `data/`                | Contains the output (zipped)                            |
| ├── `results.zip`      | Credit card fraud detection outputs                     |
| `main_pipeline.ipynb`  | Jupyter Notebook with full training pipeline            |
| `requirements.txt`     | List of required Python libraries with versioning       |
| `LICENSE`              | License file (MIT License)                              |
| `README.md`            | Project documentation, installation steps and dataset   |
| ├── `EDA Report`       | Detailed Pandas Profiling EDA report (Google Drive link)|
| ├── `Sweetviz Report`  | Automated Sweetviz EDA report (Google Drive link)       |
| ├── `Dataset`          | Credit Card Fraud Dataset (Linked)                      |

---

## Dataset
The dataset is provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and consists of:
- **284,807 transactions** with 30 features (`V1` to `V28`, `Amount`, and `Time`).
- **Highly imbalanced**:
  - Fraudulent: 0.17%
  - Non-Fraudulent: 99.83%
- Stored in: `dataset/creditcard.csv`

---

## Project Pipeline
1. **Exploratory Data Analysis (EDA)**:
   - Detailed insights using:
     - `credit_card_fraud_eda_report.html` (Pandas Profiling)
     - `sweetviz_report.html` (Sweetviz)

2. **Data Preprocessing**:
   - Handle **class imbalance** with SMOTE & undersampling.
   - Feature scaling with `StandardScaler`.

3. **Model Training**:
   - Train multiple ensemble models:
     - **Random Forest**
     - **XGBoost**
     - **LightGBM**
   - Combine models using **StackingClassifier** for improved performance.

4. **Evaluation**:
   - Key metrics:
     - **AUC-ROC** for classification performance.
     - **Confusion Matrix** to visualize errors.
     - **Precision-Recall Curve** for handling imbalanced data.

---

## 📂 Reports
You can find detailed EDA reports in the `eda_reports/` folder:
- **Pandas Profiling Report**: `credit_card_fraud_eda_report.html`
- **Sweetviz Report**: `sweetviz_report.html`

Due to GitHub’s file size limit, these files are available in the repository as `.zip' files. 

Please find their Google Drive links here: 

[Credit Card Fraud EDA Report](https://drive.google.com/file/d/1--IGCaNMtc4iftLoQSb873tqKfC-GUYc/view?usp=drivesdk)

[Credit Card Fraud Sweetviz Report](https://drive.google.com/file/d/131x7lTeph8pfJ_jK4CQVnZZRzLr27fWF/view?usp=drivesdk)


---

## Key Results
| **Model**      | **AUC-ROC** | **Precision** | **Recall** |
|----------------|-------------|---------------|------------|
| Random Forest  | 0.96        | 0.92          | 0.84       |
| XGBoost        | 0.98        | 0.95          | 0.89       |
| LightGBM       | 0.97        | 0.93          | 0.87       |
| Stacking       | **0.99**    | **0.97**      | **0.91**   |

---

## 📦 Installation
To set up the environment, clone the repository and install the dependencies:

```bash
# Clone the repo
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

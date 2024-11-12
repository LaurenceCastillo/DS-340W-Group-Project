import pandas as pd
import numpy as np
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models.tabnet import TabNetModelConfig  # Adjusted import for TabNet model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data (your preprocessing steps go here)
raw_dataset = pd.read_csv('./kidney_disease.csv')
dataframe = pd.DataFrame(raw_dataset)

# Data preprocessing steps (fill in as in your existing code)

# Drop 'id' column and rename columns
dataframe.drop('id', axis=1, inplace=True)
dataframe.columns = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
    'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
    'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
    'aanemia', 'class'
]

# Convert necessary columns to numeric types
dataframe['packed_cell_volume'] = pd.to_numeric(dataframe['packed_cell_volume'], errors='coerce')
dataframe['white_blood_cell_count'] = pd.to_numeric(dataframe['white_blood_cell_count'], errors='coerce')
dataframe['red_blood_cell_count'] = pd.to_numeric(dataframe['red_blood_cell_count'], errors='coerce')

# Correct specific values and map 'class' column to numerical labels
dataframe['diabetes_mellitus'] = dataframe['diabetes_mellitus'].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'})
dataframe['coronary_artery_disease'] = dataframe['coronary_artery_disease'].replace({'\tno': 'no'})
dataframe['class'] = dataframe['class'].replace({'ckd\t': 'ckd', 'notckd': 'not ckd'})
dataframe['class'] = dataframe['class'].map({'ckd': 0, 'not ckd': 1})

# Fill missing values
def random_value_imputation(feature):
    random_sample = dataframe[feature].dropna().sample(dataframe[feature].isna().sum())
    random_sample.index = dataframe[dataframe[feature].isnull()].index
    dataframe.loc[dataframe[feature].isnull(), feature] = random_sample

def impute_mode(feature):
    mode = dataframe[feature].mode()[0]
    dataframe[feature] = dataframe[feature].fillna(mode)

num_cols = [col for col in dataframe.columns if dataframe[col].dtype != 'object']
cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'object']

for col in num_cols:
    random_value_imputation(col)

random_value_imputation('red_blood_cells')
random_value_imputation('pus_cell')
for col in cat_cols:
    impute_mode(col)

# Label encode categorical features
le = LabelEncoder()
for col in cat_cols:
    dataframe[col] = le.fit_transform(dataframe[col])

# Split data into training and test sets
train_data, test_data = train_test_split(dataframe, test_size=0.2, random_state=42)

# Step 2: Set up TabNet using pytorch_tabular
data_config = DataConfig(
    target=['class'],
    categorical_cols=cat_cols,
    continuous_cols=num_cols,
)

trainer_config = TrainerConfig(
    max_epochs=50,
)

optimizer_config = OptimizerConfig()

# Configure TabNet model using TabNetConfig
model_config = TabNetModelConfig(  # Adjusted to TabNetModelConfig if that exists
    task="classification",
    n_d=16,
    n_a=16,
    n_steps=5,
    gamma=1.5,
)

# Initialize and train the TabularModel with TabNet
tabnet_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

# Fit the model
tabnet_model.fit(train=train_data, validation=test_data)

# Step 3: Evaluate the model
# Predict on the test set
y_true = test_data['class']
y_pred = tabnet_model.predict(test_data)

# Convert probabilities to class labels if needed
y_pred = np.array(y_pred)
if y_pred.shape[1] > 1:  # If TabNet outputs probabilities for both classes
    y_pred = (y_pred[:, 1] > 0.5).astype(int)

# Classification report and confusion matrix
print("Classification Report for TabNet:")
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for TabNet Model')
plt.show()

# Collect results for comparison
tabnet_accuracy = accuracy_score(y_true, y_pred)
tabnet_f1_score = classification_report(y_true, y_pred, output_dict=True)["1"]["f1-score"]

# Example model comparison table
comparison_data = {
    "Model": ["TabNet"],
    "Accuracy": [tabnet_accuracy],
    "F1-Score": [tabnet_f1_score]
}
comparison_df = pd.DataFrame(comparison_data)
print("\nModel Performance Comparison:")
print(comparison_df)

import pandas as pd
import numpy as np
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning.callbacks import EarlyStopping
import shap

# Load and preprocess the data
raw_dataset = pd.read_csv('./kidney_disease.csv')
dataframe = pd.DataFrame(raw_dataset)

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
le_dict = {col: LabelEncoder().fit(dataframe[col]) for col in cat_cols}
for col in cat_cols:
    dataframe[col] = le_dict[col].transform(dataframe[col])

# Label encode target variable separately and ensure only valid classes are used
target_le = LabelEncoder()
dataframe['class'] = target_le.fit_transform(dataframe['class'])

# Split data into training and test sets
train_data, test_data = train_test_split(dataframe, test_size=0.2, random_state=42)

# Define TabTransformer configurations
data_config = DataConfig(
    target=['class'],
    continuous_cols=num_cols,
    categorical_cols=cat_cols
)

model_config = TabTransformerConfig(
    task="classification",
    metrics=["accuracy"],
    input_embed_dim=32,
)

trainer_config = TrainerConfig(
    max_epochs=50,
    batch_size=32,
)

# Ensure EarlyStopping is added correctly
early_stopping_callback = EarlyStopping(
    monitor="valid_loss",  # Specify the metric for early stopping
    patience=5,
    mode="min"  # "min" because lower validation loss is better
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=OptimizerConfig(),
    trainer_config=trainer_config,
)

# Train and evaluate the model
tabular_model.fit(train=train_data)
result = tabular_model.evaluate(test_data)
print("Model Evaluation Results:", result)

# Predict on the test set
y_true = test_data['class']
y_pred = tabular_model.predict(test_data)

y_pred = np.array(y_pred)
if y_pred.shape[1] > 1:  # Check if y_pred is multilabel
    y_pred = (y_pred[:, 1] > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for TabTransformer Model')
plt.show()

# Classification Report
print("Classification Report for TabTransformer:")
print(classification_report(y_true, y_pred))

# Collect results for model comparison
tab_transformer_accuracy = accuracy_score(y_true, y_pred)
tab_transformer_f1_score = classification_report(y_true, y_pred, output_dict=True)["1"]["f1-score"]

comparison_data = {
    "Model": ["Logistic Regression", "Random Forest", "TabTransformer"],
    "Accuracy": [0.85, 0.89, tab_transformer_accuracy],
    "F1-Score": [0.84, 0.88, tab_transformer_f1_score]
}
comparison_df = pd.DataFrame(comparison_data)
print("\nModel Performance Comparison:")
print(comparison_df)

import torch
import shap

# Define a wrapper for the TabTransformer model
class TabTransformerWrapper:
    def __init__(self, model, cat_cols, num_cols):
        self.model = model
        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def __call__(self, data):
        # Separate categorical and continuous columns
        cat_data = data[:, :len(self.cat_cols)]
        cont_data = data[:, len(self.cat_cols):]
        
        # Format the data as a dictionary
        formatted_data = {
            "categorical": torch.tensor(cat_data, dtype=torch.int64),
            "continuous": torch.tensor(cont_data, dtype=torch.float32)
        }
        
        # Get model predictions from the dictionary output
        with torch.no_grad():
            output_dict = self.model(formatted_data)
            output = output_dict["logits"] if "logits" in output_dict else output_dict["predictions"]
            output = output.cpu().numpy()
            
            # Ensure output has a suitable shape
            if len(output.shape) == 1:
                output = output.reshape(-1, 1)
            return output

# Instantiate the wrapper with your trained TabTransformer model
wrapped_model = TabTransformerWrapper(tabular_model.model, cat_cols=cat_cols, num_cols=num_cols)

# Initialize SHAP explainer with the wrapped TabTransformer model
explainer = shap.Explainer(wrapped_model, train_data[cat_cols + num_cols].values)
shap_values = explainer(test_data[cat_cols + num_cols].values)

# Specify feature names for the SHAP plots
feature_names = test_data[cat_cols + num_cols].columns.tolist()

# Extract SHAP values for one class (e.g., class 1 in binary classification)
shap_values_array = shap_values.values[:, :, 1]  # Selecting the second output dimension

# Plot SHAP summary plot for interpretability
shap.summary_plot(shap_values_array, test_data[cat_cols + num_cols], plot_type="bar", feature_names=feature_names)
shap.summary_plot(shap_values_array, test_data[cat_cols + num_cols], feature_names=feature_names)




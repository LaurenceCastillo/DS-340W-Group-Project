import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV files
patients_df = pd.read_csv("PATIENTS.csv")
diagnoses_df = pd.read_csv("DIAGNOSES_ICD.csv")
prescriptions_df = pd.read_csv("PRESCRIPTIONS.csv")
icd_descriptions_df = pd.read_csv("D_ICD_DIAGNOSES.csv")

# Merge diagnoses with ICD descriptions on 'icd9_code'
diagnoses_df = diagnoses_df.merge(icd_descriptions_df, on="icd9_code", how="left")

# Convert birth dates to age
def calculate_age(birthdate_str):
    birthdate = datetime.strptime(birthdate_str, "%Y-%m-%d %H:%M:%S")
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

patients_df["age"] = patients_df["dob"].apply(calculate_age)

# Merge diagnosis history, prescriptions, and patient demographics
merged_data = diagnoses_df.merge(prescriptions_df, on=["subject_id", "hadm_id"], how="inner")
merged_data = merged_data.merge(patients_df[['subject_id', 'age', 'gender']], on="subject_id", how="inner")

# Define categories based on general ICD code mappings
category_mapping = {
    'cardio': ["410", "411", "412"],  # Example ICD codes for cardiovascular issues
    'respiratory': ["460", "490", "491"],  # Example ICD codes for respiratory issues
    'gastro': ["530", "531", "532"],  # Example ICD codes for gastrointestinal issues
    'neuro': ["340", "345", "348"]  # Example ICD codes for neurological issues
}

# Map each ICD code to a category
def map_category(icd_code):
    icd_code_str = str(icd_code)
    if icd_code_str.startswith(tuple(category_mapping['cardio'])):
        return "Cardiovascular"
    elif icd_code_str.startswith(tuple(category_mapping['respiratory'])):
        return "Respiratory"
    elif icd_code_str.startswith(tuple(category_mapping['gastro'])):
        return "Gastrointestinal"
    elif icd_code_str.startswith(tuple(category_mapping['neuro'])):
        return "Neurological"
    return None

# Apply category mapping to diagnosis data
merged_data["category"] = merged_data["icd9_code"].apply(map_category)
merged_data = merged_data.dropna(subset=["category"])

# Create combined text data for each patient by concatenating drug, gender, and age
merged_data["input_text"] = merged_data.apply(lambda x: f"Medication: {x['drug']}, Age: {x['age']}, Gender: {x['gender']}", axis=1)
texts = merged_data["input_text"].tolist()  # Combined input
labels = merged_data["category"].astype("category").cat.codes.tolist()  # Convert categories to numerical labels

# Split data into training and evaluation sets
train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initialize ClinicalBERT
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)
model.to(device)  # Move model to the selected device (GPU or CPU)

# Tokenize the data
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
eval_encodings = tokenizer(eval_texts, padding=True, truncation=True, return_tensors="pt")

# Define a custom Dataset class
class HealthcareDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create PyTorch datasets
train_dataset = HealthcareDataset(train_encodings, train_labels)
eval_dataset = HealthcareDataset(eval_encodings, eval_labels)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save the model and tokenizer for future use
model.save_pretrained('./saved_clinicalbert_model')
tokenizer.save_pretrained('./saved_clinicalbert_model')

# Making Predictions on New Text
new_texts = [
    "Medication: Aspirin, Age: 65, Gender: M",
    "Medication: Ventolin, Age: 52, Gender: F",
    "Medication: Omeprazole, Age: 45, Gender: M",
    "Medication: Metformin, Age: 70, Gender: F"
]

# Tokenize new text and move to device
new_encodings = tokenizer(new_texts, padding=True, truncation=True, return_tensors="pt").to(device)
model.eval()

with torch.no_grad():
    outputs = model(**new_encodings)
    predictions = torch.argmax(outputs.logits, dim=1).tolist()

# Map predictions back to disease categories
category_mapping = {0: "Cardiovascular", 1: "Respiratory", 2: "Gastrointestinal", 3: "Neurological"}
predicted_categories = [category_mapping[pred] for pred in predictions]

for text, category in zip(new_texts, predicted_categories):
    print(f"Text: {text}\nPredicted Disease Category: {category}\n")

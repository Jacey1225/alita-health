import pandas as pd
import numpy as np
from transformers.models.t5 import T5Tokenizer
import torch
import spacy
from sklearn.preprocessing import MinMaxScaler

nlp = spacy.load("en_core_web_sm")

class DataProcessor:
    def __init__(self, data):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.data = data
        self.text_input = self.data["text input"]
        self.desired_output = self.data["desired response"]
        self.nlp = nlp
        self.scaler = MinMaxScaler()

    def process_text_input(self):
        input_ids= []
        attention_mask = []
        for text in self.text_input:
            if pd.notna(text):
                encodings = self.tokenizer(text, padding='max_length', truncation=True, max_length=300, return_tensors='pt')
                input_ids.append(encodings["input_ids"].squeeze(0))
                attention_mask.append(encodings["attention_mask"].squeeze(0))
            else:
                input_ids.append(torch.zeros(300))
                attention_mask.append(torch.zeros(300))
        return torch.stack(input_ids), torch.stack(attention_mask) #shape = [batch_size, 150, 768]
    
    def process_label_output(self):
        label_ids = []
        for text in self.desired_output:
            encodings = self.tokenizer(text, padding='max_length', truncation=True, max_length=300, return_tensors='pt')
            label_ids.append(encodings["input_ids"].squeeze(0))

        return torch.stack(label_ids)
    
    def process_numerical_features(self):
        columns_to_process = ['Age', 'Gender', 'Diagnosis', 'Symptom Severity (1-10)', 'Mood Score (1-10)', 'Sleep Quality (1-10)', 'Physical Activity (hrs/week)' , 
                              'Medication', 'Therapy Type', 'Treatment Start Date', 'Treatment Duration (weeks)', 'Stress Level (1-10)', 'Outcome', 
                              'Treatment Progress (1-10)', 'AI-Detected Emotional State', 'Adherence to Treatment (%)']
        self.numerical_features = self.data[columns_to_process].copy()

        non_numerics = ['Gender', 'Diagnosis', 'Medication', 'Therapy Type', 'Treatment Start Date', 
                        'Outcome', 'AI-Detected Emotional State']
        for col in non_numerics:
            if col in self.numerical_features.columns:
                self.numerical_features[col] = self.numerical_features[col].apply(
                    lambda x: np.mean(self.nlp(str(x)).vector) if pd.notnull(x) else 0) #type: ignore
            
        self.numerical_features.fillna(0, inplace=True)
        self.numerical_features = torch.tensor(self.numerical_features.values, dtype=torch.float32)  # Convert to PyTorch tensor
        return self.numerical_features
    
    
    def scale_numerical_features(self):
        if self.numerical_features is None:
            raise ValueError("Numerical features have not been processed yet. Call process_numerical_features() first.")
        else:
            self.scaler.fit(self.numerical_features) #type: ignore
            scaled_features = self.scaler.transform(self.numerical_features) #type: ignore
            return torch.tensor(scaled_features, dtype=torch.float32)

    

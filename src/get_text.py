from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
import torch
import torch.nn as nn
from src.data_processing import DataProcessor
from src.model_setup import ModelSetup


model_name = "t5-base"
class GenerateText(nn.Module):
    def __init__(self, model_path="model/medical_t5_model.pth", model_name=model_name):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

    def generate(self, input_text, exclusive_parameters=None):
        try:
            if exclusive_parameters is not None:
                data = {
                    "text input": [input_text],
                    "desired response": [""],  
                    "Age": exclusive_parameters["Age"] if "Age" in exclusive_parameters else None, 
                    "Gender": exclusive_parameters["Gender"] if "Gender" in exclusive_parameters else None,
                    "Diagnosis": exclusive_parameters["Diagnosis"] if "Diagnosis" in exclusive_parameters else None,
                    "Symptom Severity (1-10)": exclusive_parameters["Symptom Severity (1-10)"] if "Symptom Severity (1-10)" in exclusive_parameters else None,
                    "Mood Score (1-10)": exclusive_parameters["Mood Score (1-10)"] if "Mood Score (1-10)" in exclusive_parameters else None,
                    "Sleep Quality (1-10)": exclusive_parameters["Sleep Quality (1-10)"] if "Sleep Quality (1-10)" in exclusive_parameters else None,
                    "Physical Activity (hrs/week)": exclusive_parameters["Physical Activity (hrs/week)"] if "Physical Activity (hrs/week)" in exclusive_parameters else None,
                    "Medication": exclusive_parameters["Medication"] if "Medication" in exclusive_parameters else None,
                    "Therapy Type": exclusive_parameters["Therapy Type"] if "Therapy Type" in exclusive_parameters else None,
                    "Treatment Start Date": exclusive_parameters["Treatment Start Date"] if "Treatment Start Date" in exclusive_parameters else None,
                    "Treatment Duration (weeks)": exclusive_parameters["Treatment Duration (weeks)"] if "Treatment Duration (weeks)" in exclusive_parameters else None,
                    "Stress Level (1-10)": exclusive_parameters["Stress Level (1-10)"] if "Stress Level (1-10)" in exclusive_parameters else None,
                    "Outcome": exclusive_parameters["Outcome"] if "Outcome" in exclusive_parameters else None,
                    "Treatment Progress (1-10)": exclusive_parameters["Treatment Progress (1-10)"] if "Treatment Progress (1-10)" in exclusive_parameters else None,
                    "AI-Detected Emotional State": exclusive_parameters["AI-Detected Emotional State"] if "AI-Detected Emotional State" in exclusive_parameters else None,
                    "Adherence to Treatment (%)": exclusive_parameters["Adherence to Treatment (%)"] if "Adherence to Treatment (%)" in exclusive_parameters else None
                }
                processor = DataProcessor(data)
                model_setup = ModelSetup(processor)
                model_setup.load_state_dict(torch.load(self.model_path))
                model_setup.eval()

                input_ids, attention_mask = model_setup.input_ids, model_setup.attention_mask
                numerical_features = model_setup.numerical_features
                with torch.no_grad():
                    output = model_setup.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        numerical_features=numerical_features,
                        labels=None
                    ) #expected output {"final state", "atention mask"}

                if output:
                    final_state = output["final state"]
                    final_attention_mask = output["attention mask"]

                    encoder = BaseModelOutput(
                        last_hidden_state=final_state,
                        attentions=None,
                        hidden_states=None,
                    )

                    generated_ids = model_setup.t5_model.generate(
                        input_ids=input_ids,
                        attention_mask=final_attention_mask,
                        encoder_outputs=encoder,
                        max_length=150,
                        num_beams=5
                    )

                    if generated_ids:
                        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        return generated_text
                    else:
                        print("No generated text found")
                        return None

        except Exception as e:
            print(f"Error initializing DataProcessor: {e}")
            return None
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
import torch
import torch.nn as nn
import pandas as pd  # ADDED: Import pandas
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
                data = pd.DataFrame({
                    "text input": [input_text],
                    "desired response": [""],  
                    "Age": [exclusive_parameters.get("Age", 30)], 
                    "Gender": [exclusive_parameters.get("Gender", "Unknown")],
                    "Diagnosis": [exclusive_parameters.get("Diagnosis", "General")],
                    "Symptom Severity (1-10)": [exclusive_parameters.get("Symptom Severity (1-10)", 5)],
                    "Mood Score (1-10)": [exclusive_parameters.get("Mood Score (1-10)", 5)],
                    "Sleep Quality (1-10)": [exclusive_parameters.get("Sleep Quality (1-10)", 5)],
                    "Physical Activity (hrs/week)": [exclusive_parameters.get("Physical Activity (hrs/week)", 3)],
                    "Medication": [exclusive_parameters.get("Medication", "None")],
                    "Therapy Type": [exclusive_parameters.get("Therapy Type", "None")], 
                    "Treatment Start Date": [exclusive_parameters.get("Treatment Start Date", "2024-01-01")],
                    "Treatment Duration (weeks)": [exclusive_parameters.get("Treatment Duration (weeks)", 4)],
                    "Stress Level (1-10)": [exclusive_parameters.get("Stress Level (1-10)", 5)],
                    "Outcome": [exclusive_parameters.get("Outcome", "Unknown")],
                    "Treatment Progress (1-10)": [exclusive_parameters.get("Treatment Progress (1-10)", 5)],
                    "AI-Detected Emotional State": [exclusive_parameters.get("AI-Detected Emotional State", "Neutral")],
                    "Adherence to Treatment (%)": [exclusive_parameters.get("Adherence to Treatment (%)", 80)]
                })
                
                print(f"🔍 Created DataFrame with shape: {data.shape}")
                print(f"📊 Sample data: {data.iloc[0][['text input', 'Age', 'Gender', 'Diagnosis']].to_dict()}")
                
                processor = DataProcessor(data)
                model_setup = ModelSetup(processor)
                
                try:
                    model_setup.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                    print("✅ Model weights loaded successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load model weights: {e}")
                    print("Using untrained model")
                
                model_setup.eval()

                input_ids, attention_mask = processor.process_text_input()
                numerical_features = processor.process_numerical_features()
                
                print(f"📊 Processed features:")
                print(f"   - Input IDs shape: {input_ids.shape}")
                print(f"   - Attention mask shape: {attention_mask.shape}")
                print(f"   - Numerical features shape: {numerical_features.shape}")
                
                with torch.no_grad():
                    output = model_setup.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        numerical_features=numerical_features,
                        labels=None
                    )

                if output and "final state" in output:
                    final_state = output["final state"]
                    final_attention_mask = output["attention mask"]

                    print(f"✅ Forward pass successful:")
                    print(f"   - Final state shape: {final_state.shape}")
                    print(f"   - Attention mask shape: {final_attention_mask.shape}")

                    encoder = BaseModelOutput(
                        last_hidden_state=final_state,
                        attentions=None,
                        hidden_states=None,
                    )

                    generated_ids = model_setup.t5_model.generate(
                        input_ids=input_ids,
                        attention_mask=final_attention_mask,
                        encoder_outputs=encoder,
                        max_length=500,
                        num_beams=2,
                        early_stopping=True,
                        do_sample=True,
                        temperature=0.7,
                        repetition_penalty=1.4,
                        pad_token_id=model_setup.t5_model.config.pad_token_id,
                        eos_token_id=model_setup.t5_model.config.eos_token_id,
                    )

                    if generated_ids is not None and len(generated_ids) > 0:
                        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        print(f"🤖 Generated text: {generated_text}")
                        return generated_text
                    else:
                        print("❌ No text generated")
                        return "I'm sorry, I couldn't generate a response."
                else:
                    print("❌ Forward pass failed")
                    return "Error: Forward pass failed"
            else:
                print("❌ No parameters provided")
                return "Please provide medical parameters for personalized response."

        except Exception as e:
            print(f"❌ Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
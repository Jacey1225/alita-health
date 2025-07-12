import pandas as pd
from transformers.models.t5 import T5ForConditionalGeneration, T5Config
import torch
import torch.nn as nn
from src.data_processing import DataProcessor
model_name = "t5-base"

class ModelSetup(nn.Module):
    def __init__(self, processor: DataProcessor, model_name=model_name):
        super().__init__()
        self.model_name = model_name
        self.data = processor
        self.input_ids, self.attention_mask = processor.process_text_input()
        self.label_ids = processor.process_label_output()
        self.numerical_features = processor.process_numerical_features()
        self.numerical_features = processor.scale_numerical_features()

        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.numerical_feature_layer = nn.Sequential( #For numerical Feature processing
            nn.Linear(self.numerical_features.shape[1], 128), #layer type
            nn.ReLU(), #Activation function for better performance 
            nn.Dropout(0.2), #regularization function to help model generalize
            nn.Linear(128, 768),  #Hidden layer size
            nn.LayerNorm(768) #Normalization layer
        )

        self.combined_layer = nn.Sequential(
            nn.Linear(768 * 2, 768), # Combine T5 and numerical features
            nn.ReLU(), # Activation function for better performance
            nn.Dropout(0.1) # Lower regularization function to help model generalize
        )
        
    def shift_right(self, labels):
        try:
            shifted_labels = []
            for input_ids in labels:
                shifted = [0] + input_ids[:-1].tolist()  # Shift right and add <START> token
                shifted_labels.append(torch.tensor(shifted))
            return torch.stack(shifted_labels)
        except Exception as e:
            print(f"Error shifting labels: {e}")
            return None

    def forward(self, input_ids, attention_mask, numerical_features, labels=None):
        try:
            encoder_outputs = self.t5_model.encoder( #model forward pass for text input
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        except Exception as e:
            print(F"Error during encoding text input: {e}") 

        try:
            mean_state = torch.mean(encoder_outputs.last_hidden_state, dim=1) #Evaluate the overall output of the text input
            numerical_outputs = self.numerical_feature_layer(numerical_features) #model forward pass for numerical features
            combined_input = self.combined_layer(torch.cat((mean_state, numerical_outputs), dim=1)) #Combine the information output from both layers
            
        except Exception as e:
            print(f"Error combining text and numerical feature inputs: {e}")
        
        try: #FIXME: Error here - size mismatch 
            batch_size, seq_length, hidden_layer_size = encoder_outputs.last_hidden_state.shape #Ensure that we get the correct shape for the output [4, 150, 768]
            combined_output = combined_input.unsqueeze(1).expand(batch_size, seq_length, hidden_layer_size) # should look like [text_output(eg. [0.124, 0.481, 0.601]), numerical_output(eg. [0.567, 0.678, ...]) --> [0.124, 0.481, 0.601, 0.567, 0.678, ...]
            final_state = encoder_outputs.last_hidden_state + combined_output #match overall expected output shape
            
        except Exception as e:
            print(f"Error during output layer processing: {e}")

        try:
            if labels is not None:
                next_label = self.shift_right(labels) #Allow model to predict the next best token
                if next_label is None:
                    print("Error shifting labels, returning None")
                    return None
            
                encoder_outputs.last_hidden_state = final_state  
                outputs = self.t5_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=next_label,
                    labels=labels,
                    encoder_outputs=encoder_outputs,
                )
                if outputs:
                    return {
                        "loss": outputs.loss,
                        "scores": outputs.logits,
                        "final state": final_state
                    }
                else:
                    return None
            else:
                print("No labels provided, returning enhanced encoder states for generation")
                return {
                    "final state": final_state,
                    "attention mask": attention_mask
                }
        except Exception as e:
            print(f"Error during final layer processing: {e}")
            return None

class Training():
    def __init__(self, batch_size, epochs, filename="data/conversational_dataset.csv", training_size=0.8, validation_size=0.1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.filename = filename
        self.training_size = training_size
        self.validation_size = validation_size

        self.data = pd.read_csv(self.filename) 
        self.training_data = self.data.sample(frac=self.training_size).reset_index(drop=True)
        self.processor = DataProcessor(self.training_data)
        self.model = ModelSetup(
            processor=self.processor
        )
        self.validation_data = self.data.sample(frac=self.validation_size).reset_index(drop=True)
        self.validation_processor = DataProcessor(self.validation_data)
        self.validation_model = ModelSetup(
            processor=self.validation_processor
        )

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=4e-5,
            weight_decay=0.001 #L2 regularization
        )
        self.model.train()

        for epoch in range(self.epochs):
            print(f"\n\n🏋️‍♂️ Training epoch {epoch + 1}/{self.epochs}...")
            index = 0
            average_loss = 0.0
            while index < len(self.model.input_ids):
                print(f"\rProcessing batch {index // self.batch_size + 1}...", end="", flush=True)

                optimizer.zero_grad() #reset gradients to zero before each batch
                outputs = self.model.forward(
                    input_ids=self.model.input_ids[index:index+self.batch_size],
                    attention_mask=self.model.attention_mask[index:index+self.batch_size],
                    numerical_features=self.model.numerical_features[index:index+self.batch_size],
                    labels=self.model.label_ids[index:index+self.batch_size]
                )

                try:
                    loss = outputs['loss']
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(f"Error during backprop: {e}")
                    continue

                average_loss += loss.item()
                index += self.batch_size
            average_loss /= (len(self.model.input_ids) // self.batch_size)
            print(f"\r🎯 Average Loss for epoch {epoch}: {average_loss}", end="", flush=True)
            self.validate() 
            if average_loss < 1.5:
                print(f"\n🎉 Early stopping at epoch {epoch + 1} due to low loss.")
                break

    def validate(self):
        index = 0
        average_loss = 0.0
        while index < len(self.validation_model.input_ids):
            outputs = self.validation_model.forward(
                input_ids=self.validation_model.input_ids[index:index+self.batch_size],
                attention_mask=self.validation_model.attention_mask[index:index+self.batch_size],
                numerical_features=self.validation_model.numerical_features[index:index+self.batch_size],
                labels=self.validation_model.label_ids[index:index+self.batch_size]
            )

            try:
                loss = outputs['loss']
                average_loss += loss.item()
            except Exception as e:
                print(f"Error during validation: {e}")
                continue

            index += self.batch_size
        
        average_loss /= (len(self.model.input_ids) // self.batch_size)
        print(f"🎯 Average Loss for validation: {average_loss}")

    def save_model(self, path='model/medical_t5_model.pth'):
        try:
            torch.save(self.model.state_dict(), path)
            print(f"✅ Model saved successfully to {path}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
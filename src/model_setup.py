import pandas as pd
from transformers.models.t5 import T5ForConditionalGeneration, T5Config
import torch
import torch.nn as nn
from src.data_processing import DataProcessor

dataset = DataProcessor()
model_name = "t5-base"

class ModelSetup(nn.Module):
    def __init__(self, num_numerical_features: int, processor: DataProcessor, model_name=model_name):
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.model_name = model_name
        self.data = processor
        self.input_ids, self.attention_mask = processor.process_text_input()
        self.label_ids = processor.process_label_output()
        self.numerical_features = processor.process_numerical_features()
        self.numerical_features = processor.scale_numerical_features()

        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.numerical_feature_layer = nn.Sequential( #For numerical Feature processing
            nn.Linear(self.num_numerical_features, 128), #layer type
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
            print(f"Shape of encoder_outputs: {encoder_outputs.last_hidden_state.shape}")
        except Exception as e:
            print(F"Error during encoding text input: {e}") 

        try:
            mean_state = torch.mean(encoder_outputs.last_hidden_state, dim=1) #Evaluate the overall output of the text input
            numerical_outputs = self.numerical_feature_layer(numerical_features) #model forward pass for numerical features
            combined_input = self.combined_layer(torch.cat((mean_state, numerical_outputs), dim=1)) #Combine the information output from both layers
            print(f"Shape of combined_input: {combined_input.shape}")
            print(f"Expected shape: [batch_size, 768]")
            print(f"Example Feature Output in Forward Pass{combined_input[0]}")
            
        except Exception as e:
            print(f"Error combining text and numerical feature inputs: {e}")
        
        try: #FIXME: Error here - size mismatch 
            batch_size, seq_length, hidden_layer_size = encoder_outputs.last_hidden_state.shape #Ensure that we get the correct shape for the output [4, 150, 768]
            combined_output = combined_input.unsqueeze(1).expand(batch_size, seq_length, hidden_layer_size) # should look like [text_output(eg. [0.124, 0.481, 0.601]), numerical_output(eg. [0.567, 0.678, ...]) --> [0.124, 0.481, 0.601, 0.567, 0.678, ...]
            final_state = encoder_outputs.last_hidden_state + combined_output #match overall expected output shape
            print(f"Shape of combined_output: {final_state.shape}")
            print(f"Expected shape: {batch_size, seq_length, hidden_layer_size}")
            print(f"Example Combined Output in Forward Pass: {final_state[0]}")
            
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
                print(f"Shape of outputs: {outputs.logits.shape}")
                print(f"Loss value: {outputs.loss.item()}")

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


def train(batch_size, epochs):
    processor = DataProcessor()
    num_numerical_features = processor.numerical_features.shape[1]

    input_ids, attention_masks = processor.process_text_input()
    label_ids = processor.process_label_output()
    numerical_features = processor.process_numerical_features()
    numerical_features = processor.scale_numerical_features()

    model = ModelSetup(
        num_numerical_features=num_numerical_features,
        processor=processor
    )

    optimizer = torch.optim.Adam(
       model.parameters(),
       lr=4e-5,
       weight_decay=0.001 #L2 regularization 
    )
    model.train()

    for epoch in range(epochs):
        for index in range(batch_size):
            optimizer.zero_grad() #reset gradients to zero before each batch
            outputs = model.forward(
                input_ids=input_ids[index:index+1],
                attention_mask=attention_masks[index:index+1],
                numerical_features=numerical_features[index:index+1],
                labels=label_ids[index:index+1]
            )

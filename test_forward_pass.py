#!/usr/bin/env python3
"""
Forward Pass Tester for ModelSetup

This script tests the forward pass function and logs all output to both
the terminal and a log file for debugging purposes.
"""

import sys
import os
import logging
from datetime import datetime
import torch
import pandas as pd

# Add the project root to Python path
sys.path.append('/Users/jaceysimpson/Vscode/Alita')

from src.data_processing import DataProcessor
from src.model_setup import ModelSetup

class ForwardPassTester:
    def __init__(self, log_to_file=True, batch_size=4):
        """Initialize the tester with logging configuration."""
        self.setup_logging(log_to_file)
        self.processor = None
        self.model = None
        self.input_ids = None
        self.attention_mask = None
        self.numerical_features = None
        self.labels = None  # ADDED: Initialize labels
        self.batch_size = batch_size  # Small batch size for testing
        
    def setup_logging(self, log_to_file=True):
        """Setup logging to both console and file."""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('ForwardPassTester')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(f'logs/forward_pass_test_{timestamp}.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.info("=" * 60)
        self.logger.info("FORWARD PASS TESTER INITIALIZED")
        self.logger.info("=" * 60)

    def prepare_data(self):
        """Prepare and validate a SMALL subset of data to avoid memory issues."""
        self.logger.info(f"🔄 Preparing SMALL test data (batch_size={self.batch_size})...")
        
        try:
            # Initialize data processor
            self.processor = DataProcessor()
            self.logger.info("✅ DataProcessor initialized successfully")
            
            # Process ALL data first (we need this for model initialization)
            self.logger.info("📝 Processing text input...")
            full_input_ids, full_attention_mask = self.processor.process_text_input()
            self.logger.info(f"✅ Full dataset text processing complete:")
            self.logger.info(f"   - Full Input IDs shape: {full_input_ids.shape}")
            self.logger.info(f"   - Full Attention mask shape: {full_attention_mask.shape}")
            
            # Process numerical features
            self.logger.info("🔢 Processing numerical features...")
            full_numerical_features = self.processor.process_numerical_features()
            self.logger.info(f"✅ Full dataset numerical processing complete:")
            self.logger.info(f"   - Full Numerical features shape: {full_numerical_features.shape}")
            
            # ADDED: Process label output for training mode
            self.logger.info("🏷️ Processing label output...")
            full_labels = self.processor.process_label_output()
            self.logger.info(f"✅ Full dataset label processing complete:")
            self.logger.info(f"   - Full Labels shape: {full_labels.shape}")
            
            # ⚡ MEMORY OPTIMIZATION: Take only small subset for actual testing
            self.logger.info(f"✂️ Extracting small test subset ({self.batch_size} samples)...")
            self.input_ids = full_input_ids[:self.batch_size]
            self.attention_mask = full_attention_mask[:self.batch_size]
            self.numerical_features = full_numerical_features[:self.batch_size]
            self.labels = full_labels[:self.batch_size]  # ADDED: Extract labels subset
            
            self.logger.info(f"✅ Small test data prepared:")
            self.logger.info(f"   - Test Input IDs shape: {self.input_ids.shape}")
            self.logger.info(f"   - Test Attention mask shape: {self.attention_mask.shape}")
            self.logger.info(f"   - Test Numerical features shape: {self.numerical_features.shape}")
            self.logger.info(f"   - Test Labels shape: {self.labels.shape}")  # ADDED
            self.logger.info(f"   - Memory reduction: {full_input_ids.shape[0]} → {self.batch_size} samples")
            
            # Log sample data
            self.logger.info("📊 Sample test data:")
            self.logger.info(f"   - First input_ids sample: {self.input_ids[0][:10]}...")  # First 10 tokens
            self.logger.info(f"   - First numerical features: {self.numerical_features[0]}")
            self.logger.info(f"   - First labels sample: {self.labels[0][:10]}...")  # ADDED: First 10 label tokens
            
        except Exception as e:
            self.logger.error(f"❌ Error in data preparation: {str(e)}")
            raise

    def initialize_model(self):
        """Initialize the ModelSetup."""
        self.logger.info("🤖 Initializing model...")
        
        try:
            if self.processor is None:
                raise ValueError("Processor not initialized. Call prepare_data() first.")

            self.model = ModelSetup(
                num_numerical_features=16,
                processor=self.processor
            )
            self.logger.info("✅ Model initialized successfully")
            
            # Log model architecture info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"📈 Model information:")
            self.logger.info(f"   - Total parameters: {total_params:,}")
            self.logger.info(f"   - Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in model initialization: {str(e)}")
            raise

    def test_training_mode(self):
        """Test the forward pass in TRAINING MODE with labels for loss calculation."""
        self.logger.info(f"🎯 Testing TRAINING MODE forward pass with {self.batch_size} samples...")
        
        try:
            if self.model is None:
                raise ValueError("Model not initialized. Call initialize_model() first.")
            if self.input_ids is None or self.attention_mask is None or self.numerical_features is None:
                raise ValueError("Data not prepared. Call prepare_data() first.")
            if not hasattr(self, 'labels') or self.labels is None:
                raise ValueError("Labels not prepared. Check prepare_data() method.")
                
            # Log memory info before forward pass
            if torch.cuda.is_available():
                self.logger.info(f"🔥 GPU Memory before forward pass:")
                self.logger.info(f"   - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                self.logger.info(f"   - Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            # Capture the model's print statements by temporarily redirecting stdout
            import io
            from contextlib import redirect_stdout
            
            captured_output = io.StringIO()
            
            self.logger.info(f"⚡ Starting TRAINING MODE forward pass execution with batch size {self.batch_size}...")
            self.logger.info(f"   - Input tensor shapes: {self.input_ids.shape}, {self.attention_mask.shape}, {self.numerical_features.shape}")
            self.logger.info(f"   - Labels tensor shape: {self.labels.shape}")
            self.logger.info(f"   - Providing labels to trigger training mode with loss calculation")
            
            with redirect_stdout(captured_output):
                # Run the forward pass WITH labels for training mode
                training_output = self.model.forward(
                    self.input_ids, 
                    self.attention_mask, 
                    self.numerical_features,
                    labels=self.labels  # Include labels for training mode with loss
                )
            
            # Log memory info after forward pass
            if torch.cuda.is_available():
                self.logger.info(f"🔥 GPU Memory after forward pass:")
                self.logger.info(f"   - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                self.logger.info(f"   - Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            # Get the captured print statements
            model_prints = captured_output.getvalue()
            
            # Log the captured output
            self.logger.info("📋 Model forward pass print statements (TRAINING MODE):")
            for line in model_prints.strip().split('\n'):
                if line.strip():
                    self.logger.info(f"   MODEL: {line}")
            
            # Log training mode results
            if training_output is not None:
                self.logger.info("✅ TRAINING MODE forward pass completed successfully!")
                self.logger.info(f"📊 Training output information:")
                
                if isinstance(training_output, dict):
                    self.logger.info(f"   - Output type: {type(training_output)}")
                    self.logger.info(f"   - Output keys: {list(training_output.keys())}")
                    
                    # Log loss information (MAIN GOAL!)
                    if 'loss' in training_output:
                        loss = training_output['loss']
                        self.logger.info(f"🎯 TRAINING LOSS: {loss.item():.6f}")
                        self.logger.info(f"   - Loss tensor shape: {loss.shape}")
                        self.logger.info(f"   - Loss dtype: {loss.dtype}")
                        self.logger.info(f"   - Loss requires_grad: {loss.requires_grad}")
                        self.logger.info(f"   - Loss device: {loss.device}")
                    
                    # Log scores/logits information
                    if 'scores' in training_output:
                        scores = training_output['scores']
                        self.logger.info(f"🎯 SCORES/LOGITS: {scores.shape}")
                        self.logger.info(f"   - Scores dtype: {scores.dtype}")
                        self.logger.info(f"   - Scores device: {scores.device}")
                        self.logger.info(f"   - Scores memory usage: {scores.numel() * scores.element_size() / 1024**2:.2f} MB")
                        self.logger.info(f"   - Sample scores (first 5 vocab): {scores[0, 0, :5]}")
                        self.logger.info(f"   - Max score: {scores.max().item():.4f}")
                        self.logger.info(f"   - Min score: {scores.min().item():.4f}")
                    
                    # Log final state information  
                    if 'final state' in training_output:
                        final_state = training_output['final state']
                        self.logger.info(f"🎯 FINAL STATE: {final_state.shape}")
                        self.logger.info(f"   - Final state dtype: {final_state.dtype}")
                        self.logger.info(f"   - Final state device: {final_state.device}")
                        self.logger.info(f"   - Sample final state features: {final_state[0, 0, :5]}")
                        
                elif isinstance(training_output, torch.Tensor):
                    self.logger.info(f"   - Output type: {type(training_output)}")
                    self.logger.info(f"   - Output shape: {training_output.shape}")
                    self.logger.info(f"   - Output dtype: {training_output.dtype}")
                    self.logger.info(f"   - Output device: {training_output.device}")
                    self.logger.info(f"   - Memory usage: {training_output.numel() * training_output.element_size() / 1024**2:.2f} MB")
                    self.logger.info(f"   - Sample values: {training_output[0, 0, :5] if len(training_output.shape) >= 3 else training_output[:5]}")
                else:
                    self.logger.info(f"   - Output type: {type(training_output)}")
                    self.logger.info(f"   - Output: {training_output}")
            else:
                self.logger.warning("⚠️ TRAINING MODE forward pass returned None")
                
        except Exception as e:
            self.logger.error(f"❌ Error in TRAINING MODE forward pass: {str(e)}")
            self.logger.error(f"❌ Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise

    def test_inference_mode(self):
        """Test the forward pass in inference mode (without labels)."""
        self.logger.info(f"🚀 Testing INFERENCE MODE forward pass with {self.batch_size} samples...")
        
        try:
            if self.model is None:
                raise ValueError("Model not initialized. Call initialize_model() first.")
            if self.input_ids is None or self.attention_mask is None or self.numerical_features is None:
                raise ValueError("Data not prepared. Call prepare_data() first.")
            
            import io
            from contextlib import redirect_stdout
            captured_output = io.StringIO()
            
            self.logger.info(f"⚡ Starting INFERENCE MODE forward pass...")
            
            with redirect_stdout(captured_output):
                # Run forward pass WITHOUT labels for inference mode
                output = self.model.forward(
                    self.input_ids, 
                    self.attention_mask, 
                    self.numerical_features
                    # No labels = inference mode
                )
            
            # Log the captured output
            model_prints = captured_output.getvalue()
            self.logger.info("📋 Inference mode print statements:")
            for line in model_prints.strip().split('\n'):
                if line.strip():
                    self.logger.info(f"   MODEL: {line}")
            
            # Log inference results
            if output is not None and isinstance(output, dict):
                self.logger.info("✅ Inference mode completed successfully!")
                
                if 'enhanced_encoder_states' in output:
                    enhanced_states = output['enhanced_encoder_states']
                    self.logger.info(f"   - Enhanced encoder states shape: {enhanced_states.shape}")
                    self.logger.info(f"   - Enhanced states device: {enhanced_states.device}")
                    self.logger.info(f"   - Sample enhanced features: {enhanced_states[0, 0, :5]}")
        
        except Exception as e:
            self.logger.error(f"❌ Error in inference mode: {str(e)}")
            raise

    def run_comprehensive_test(self):
        """Run a comprehensive test of both training and inference modes."""
        try:
            self.logger.info(f"🎯 Starting comprehensive forward pass test (batch_size={self.batch_size})...")
            
            # Step 1: Prepare data (now includes labels)
            self.prepare_data()
            
            # Step 2: Initialize model
            self.initialize_model()
            
            # Step 3: Test training mode (with labels)
            self.logger.info("🏋️ Testing TRAINING MODE...")
            self.test_training_mode()
            
            # Step 4: Test inference mode (without labels)
            self.logger.info("🔮 Testing INFERENCE MODE...")
            self.test_inference_mode()
            
            self.logger.info("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            self.logger.info(f"💡 Test used only {self.batch_size} samples out of full dataset to avoid memory issues")
            self.logger.info("✅ Both training and inference modes tested")
            
        except Exception as e:
            self.logger.error(f"💥 TEST FAILED: {str(e)}")
            sys.exit(1)
        
        finally:
            self.logger.info("=" * 60)
            self.logger.info("TEST SESSION ENDED")
            self.logger.info("=" * 60)

    def test_multiple_batch_sizes(self, batch_sizes=[2, 4, 8]):
        """Test forward pass with multiple batch sizes to find optimal size."""
        self.logger.info(f"🔬 Testing multiple batch sizes: {batch_sizes}")
        
        # Store original batch size
        original_batch_size = self.batch_size
        
        for test_batch_size in batch_sizes:
            try:
                self.logger.info(f"" + "="*50)
                self.logger.info(f"🧪 Testing with batch size: {test_batch_size}")
                self.logger.info(f"" + "="*50)
                
                # Update batch size
                self.batch_size = test_batch_size
                
                # Prepare fresh data with new batch size
                self.prepare_data()
                
                # Test training mode and inference mode
                self.test_training_mode()
                self.test_inference_mode()
                
                self.logger.info(f"✅ Batch size {test_batch_size} completed successfully!")
                
                # Clear GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("🧹 GPU cache cleared")
                
            except Exception as e:
                self.logger.error(f"❌ Batch size {test_batch_size} failed: {str(e)}")
                # Continue with next batch size
                continue
        
        # Restore original batch size
        self.batch_size = original_batch_size
        self.logger.info(f"🏁 Multiple batch size testing completed")

def main():
    """Main function to run the tester with configurable batch size."""
    print("🧪 Forward Pass Tester Starting...")
    print("📋 All output will be logged to terminal and log file.")
    print("🔧 Using small batch size to avoid memory issues")
    print("=" * 60)
    
    # Create and run tester with small batch size
    batch_size = 4  # Very small batch to avoid memory issues
    tester = ForwardPassTester(log_to_file=True, batch_size=batch_size)
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()

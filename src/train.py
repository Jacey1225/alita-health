import sys
import os

sys.path.append('/Users/jaceysimpson/Vscode/Alita')

from src.model_setup import Training

def main():
    print("🏋️ Starting Medical T5 Model Training with Improved Dataset...")
    print("=" * 50)
    
    try:
        trainer = Training(
            batch_size=4,  # Reduced for stability with enhanced model
            epochs=3      # Reduced for initial testing
        )
        print("✅ Trainer initialized successfully")
        print(f"📊 Training setup:")
        print(f"   - Batch size: {trainer.batch_size}")
        print(f"   - Epochs: {trainer.epochs}")
        print(f"   - Training data size: {len(trainer.training_data)}")
        print(f"   - Validation data size: {len(trainer.validation_data)}")
        print(f"   - Dataset: {trainer.filename}")
        print(f"   - Features: Enhanced with actionable medical advice")
        
        print(f"\n🚀 Starting training with improved conversational data...")
        trainer.train()
        trainer.save_model()
        
        print("\n🎉 Training completed successfully!")
        print("💡 Your model should now generate more helpful, actionable advice!")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

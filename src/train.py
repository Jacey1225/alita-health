import sys
import os

sys.path.append('/Users/jaceysimpson/Vscode/Alita')

from src.model_setup import Training

def main():
    print("🏋️ Starting Medical T5 Model Training...")
    print("=" * 50)
    
    try:
        trainer = Training(
            batch_size=32,
            epochs=10
        )
        print("✅ Trainer initialized successfully")
        print(f"📊 Training setup:")
        print(f"   - Batch size: {trainer.batch_size}")
        print(f"   - Epochs: {trainer.epochs}")
        print(f"   - Training data size: {len(trainer.training_data)}")
        print(f"   - Validation data size: {len(trainer.validation_data)}")
        
        trainer.train()
        trainer.save_model()
        
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

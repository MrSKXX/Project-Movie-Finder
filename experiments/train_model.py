from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class ModelTrainer:
    def __init__(self, base_model='all-MiniLM-L6-v2'):
        self.base_model = base_model
        
    def train(self, epochs=1): 
        print(f"🔄 Loading FRESH base model: {self.base_model}")
        model = SentenceTransformer(self.base_model)
        
        # 1. Charger les données (seulement des paires positives maintenant)
        df = pd.read_csv(config.TRAINING_DATA_PATH)
        if len(df) > 10000: # On limite la taille pour éviter l'overfitting
            df = df.sample(10000)
            
        train_examples = []
        for _, row in df.iterrows():
            # Note: Plus de label ici ! Juste [Query, Doc]
            train_examples.append(InputExample(texts=[str(row['query']), str(row['movie_text'])]))
            
        # 2. DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        
        # 3. La Loss Magique : MultipleNegativesRankingLoss
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        output_path = f"{config.MODELS_DIR}/movie_finder_mnrl_ep{epochs}"
        
        print(f"\n🚀 Training with MNRL for {epochs} epochs...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * 0.1),
            output_path=output_path,
            show_progress_bar=True
        )
        print(f"✅ Model saved to {output_path}")
        return output_path

def main():
    trainer = ModelTrainer()
    
    # On teste 1 et 3 époques. 
    # Avec cette méthode, 1 seule époque suffit souvent.
    for ep in [1, 3]:
        trainer.train(epochs=ep)

if __name__ == "__main__":
    main()
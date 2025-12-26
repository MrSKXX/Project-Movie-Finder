"""
Module d'entraînement du modèle avec MultipleNegativesRankingLoss
Inclut validation en temps réel et sauvegarde du meilleur modèle
"""

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config


class ModelTrainer:
    """Entraîne le modèle de recherche sémantique avec MNRL"""
    
    def __init__(self, base_model='all-MiniLM-L6-v2'):
        """
        Initialise le trainer
        
        Args:
            base_model: Modèle de base pré-entraîné à fine-tuner
        """
        self.base_model = base_model
        
    def load_data(self):
        """
        Charge les ensembles d'entraînement et de validation
        
        Returns:
            Tuple (train_df, val_df)
        """
        train_path = config.TRAINING_DATA_PATH.replace('.csv', '_train.csv')
        val_path = config.TRAINING_DATA_PATH.replace('.csv', '_val.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print(f"Erreur: Données d'entraînement introuvables")
            print(f"Exécutez: python -m training.data_generator")
            sys.exit(1)
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        print(f"Données d'entraînement: {len(train_df)} échantillons")
        print(f"Données de validation: {len(val_df)} échantillons")
        
        return train_df, val_df
    
    def create_evaluator(self, val_df):
        """
        Crée l'évaluateur pour le monitoring de validation
        
        Args:
            val_df: DataFrame de validation
        
        Returns:
            InformationRetrievalEvaluator configuré
        """
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        for idx, row in val_df.iterrows():
            query_id = f"q{idx}"
            doc_id = f"d{idx}"
            
            queries[query_id] = str(row['query'])
            corpus[doc_id] = str(row['movie_text'])
            relevant_docs[query_id] = {doc_id}
        
        evaluator = evaluation.InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name='movie-search-val',
            show_progress_bar=True
        )
        
        return evaluator
    
    def train(self, epochs=3, batch_size=32, learning_rate=2e-5, output_name='movie_finder_v1'):
        """
        Lance l'entraînement complet avec validation
        
        Args:
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille des batches
            learning_rate: Taux d'apprentissage
            output_name: Nom du dossier de sortie pour le modèle
        
        Returns:
            Chemin du modèle sauvegardé
        """
        
        print("\n" + "="*70)
        print("DÉMARRAGE DE L'ENTRAÎNEMENT")
        print("="*70 + "\n")
        
        print(f"Chargement du modèle de base: {self.base_model}")
        model = SentenceTransformer(self.base_model)
        
        train_df, val_df = self.load_data()
        
        print("\nCréation des exemples d'entraînement...")
        train_examples = []
        for _, row in train_df.iterrows():
            train_examples.append(
                InputExample(texts=[str(row['query']), str(row['movie_text'])])
            )
        
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        print("Configuration de la loss: MultipleNegativesRankingLoss")
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        print("Création de l'évaluateur de validation...")
        evaluator = self.create_evaluator(val_df)
        
        output_path = os.path.join(config.MODELS_DIR, 'fine_tuned', output_name)
        os.makedirs(output_path, exist_ok=True)
        
        warmup_steps = int(len(train_dataloader) * 0.1)
        
        print(f"\nConfiguration d'entraînement:")
        print(f"   Époques: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Warmup steps: {warmup_steps}")
        print(f"   Batches d'entraînement: {len(train_dataloader)}")
        print(f"   Loss: MultipleNegativesRankingLoss")
        print(f"   Sortie: {output_path}")
        print("\n" + "="*70 + "\n")
        
        start_time = time.time()
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=len(train_dataloader) // 2,
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True,
            optimizer_params={'lr': learning_rate}
        )
        
        elapsed = time.time() - start_time
        print(f"\nEntraînement terminé en {elapsed/60:.1f} minutes")
        print(f"Modèle sauvegardé: {output_path}")
        
        metadata = {
            'base_model': self.base_model,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'training_time_minutes': round(elapsed/60, 2),
            'output_path': output_path
        }
        
        with open(os.path.join(output_path, 'training_config.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Configuration sauvegardée: {output_path}/training_config.json")
        
        return output_path


def main():
    """Point d'entrée pour l'entraînement du modèle"""
    print("\n" + "#"*70)
    print("# FINE-TUNING AVEC MNRL - CINESPHERE")
    print("#"*70 + "\n")
    
    trainer = ModelTrainer()
    
    model_path = trainer.train(
        epochs=3,
        batch_size=32,
        learning_rate=2e-5,
        output_name='movie_finder_v1'
    )
    
    print("\n" + "#"*70)
    print("# ENTRAÎNEMENT TERMINÉ")
    print("#"*70)
    print(f"\nModèle disponible: {model_path}")
    print("\nProchaines étapes:")
    print("  1. python -m training.evaluate")
    print("  2. python -m src.movie_retriever")
    print("  3. python -m src.app\n")


if __name__ == "__main__":
    main()
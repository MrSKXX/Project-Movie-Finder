# Modèles CineSphere

Les modèles fine-tunés sont hébergés sur Hugging Face (trop volumineux pour GitHub).

## Télécharger les Modèles
```bash
# Option 1: Script automatique
python scripts/download_from_huggingface.py

# Option 2: Manuel
huggingface-cli download votre-username/cinesphere-minilm-v1 --local-dir models/fine_tuned/movie_finder_v1
```

## Modèles Disponibles

| Modèle | Lien Hugging Face | Taille | Performance |
|--------|-------------------|--------|-------------|
| all-MiniLM-L6-v2 | [🤗 Hub](https://huggingface.co/votre-username/cinesphere-minilm-v1) | 90 MB | MRR: 0.611 |
| BERT-base | [🤗 Hub](https://huggingface.co/votre-username/cinesphere-bert-base-v1) | 440 MB | MRR: TBD |
| BERT-LoRA | [🤗 Hub](https://huggingface.co/votre-username/cinesphere-bert-lora-v1) | 440 MB | MRR: TBD |

## Structure
```
models/
├── base/              # Modèles pré-entraînés (téléchargés automatiquement)
└── fine_tuned/        # Modèles fine-tunés (télécharger depuis Hugging Face)
    ├── movie_finder_v1/    # all-MiniLM-L6-v2
    ├── bert_base_v1/       # BERT-base
    └── bert_lora_v1/       # BERT-LoRA
```

## Pour le Développement

Si vous entraînez un nouveau modèle:

1. Il sera sauvegardé dans `models/fine_tuned/`
2. Uploadez-le sur Hugging Face: `python scripts/upload_to_huggingface.py`
3. Partagez le lien avec l'équipe

## Note

Les modèles ne sont PAS versionnés dans Git pour respecter la limite de 100 MB par fichier de GitHub.
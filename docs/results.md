# Méthodologie - CineSphere

## Vue d'ensemble

CineSphere utilise une approche de fine-tuning par curriculum learning pour adapter un modèle pré-entraîné (all-MiniLM-L6-v2) au domaine de la recherche cinématographique.

## 1. Collecte des Données

### Source
- API TMDB (The Movie Database)
- 3,230 films populaires
- Métadonnées: titre, résumé, genres, mots-clés, note, popularité

### Attributs extraits
```python
{
    'id': int,
    'title': str,
    'plot': str,           # Résumé du film
    'genres': str,         # Genres séparés par virgule
    'keywords': str,       # Mots-clés séparés par virgule
    'year': str,
    'rating': float,       # Note /10
    'popularity': float,
    'poster_path': str
}
```

## 2. Génération des Données d'Entraînement

### Curriculum Learning

Contrairement à une approche naïve qui génère des requêtes aléatoires, nous utilisons un curriculum en 4 niveaux de difficulté croissante:

#### Niveau 1: Compréhension des Genres (85% des données)
**Objectif**: Apprendre les associations genre ↔ descripteurs
```python
Genre: "Action"
Descripteurs: ['explosive', 'high-octane', 'intense', 'combat']
Requêtes générées:
  - "explosive action movie"
  - "intense film"
  - "combat action movie"
```

**Nombre de paires**: ~24,780

#### Niveau 2: Compréhension Thématique (1.4% des données)
**Objectif**: Identifier les thèmes narratifs
```python
Thèmes détectés dans le plot:
  - revenge, friendship, survival, redemption, etc.

Requêtes générées:
  - "movie about revenge"
  - "drama movie about redemption"
```

**Nombre de paires**: ~394

#### Niveau 3: Requêtes Basées sur le Plot (5.8% des données)
**Objectif**: Comprendre les éléments narratifs
```python
Plot: "A young wizard discovers he has magical powers..."

Requêtes générées:
  - "movie where a boy discovers magical powers"
  - "movie about a wizard learning magic"
```

**Nombre de paires**: ~1,688

#### Niveau 4: Requêtes Multi-Concepts (8% des données)
**Objectif**: Combiner plusieurs dimensions
```python
Requêtes générées:
  - "action movie set in space about survival"
  - "drama with revenge and family"
```

**Nombre de paires**: ~2,350

### Représentation Textuelle des Films

Chaque film est représenté par une concaténation pondérée:
```python
movie_text = (
    f"{title} {title} "                    # 2x titre
    f"{genres} {genres} {genres} "         # 3x genres
    f"{keywords} × 4 "                     # 4x keywords
    f"{plot[:400]} "                       # Plot tronqué
    f"{year} film rated {rating}"
)
```

**Rationale**: 
- Titre: Identifiant principal
- Genres: Catégorisation importante
- Keywords: Descripteurs riches
- Plot: Contexte sémantique

## 3. Entraînement

### Modèle de Base
**all-MiniLM-L6-v2**
- Pré-entraîné sur 1B+ paires de phrases
- 384 dimensions d'embedding
- 22M paramètres
- Optimisé pour la similarité sémantique

### Loss Function: MultipleNegativesRankingLoss

**Principe:**
```
Pour chaque batch de 32 exemples:
  Query: "romantic movie"
  Positive: Titanic
  Negatives (automatiques): 31 autres films du batch
  
Loss = -log(exp(sim(q, pos)) / Σ exp(sim(q, all)))
```

**Avantages vs CosineSimilarityLoss:**
- Pas besoin de paires négatives manuelles
- In-batch negatives automatiques
- Meilleure généralisation
- Pas d'overfitting

### Hyperparamètres
```python
{
    'epochs': 3,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'warmup_steps': 23,
    'optimizer': 'AdamW',
    'scheduler': 'WarmupLinear'
}
```

**Justification:**
- **3 époques**: Validation montre plateau à epoch 2-3
- **Batch 32**: Compromis GPU/CPU + 31 négatifs
- **LR 2e-5**: Standard pour fine-tuning BERT-like
- **Warmup**: Évite instabilité initiale

### Validation en Temps Réel

Évaluation tous les 118 batches (mi-époque):
- MRR (Mean Reciprocal Rank)
- Precision@1, @3, @5
- Recall@10

**Évolution observée:**
```
Epoch 0.5: MRR = 0.260
Epoch 1.0: MRR = 0.277 (+6.5%)
Epoch 1.5: MRR = 0.282 (+1.8%)
Epoch 2.0: MRR = 0.287 (+1.8%)
Epoch 2.5: MRR = 0.287 (plateau)
Epoch 3.0: MRR = 0.287
```

## 4. Indexation Vectorielle

### FAISS (Facebook AI Similarity Search)

**Index utilisé**: IndexFlatL2
- Recherche exacte par distance L2
- Pas de compression (qualité maximale)
- 3,230 vecteurs × 384 dimensions

**Processus:**
```python
1. Encoder tous les films avec le modèle fine-tuné
2. Créer l'index FAISS
3. Sauvegarder (faiss_index_trained.bin)
```

## 5. Système de Reranking Hybride

La similarité sémantique seule ne suffit pas. Nous utilisons un score hybride:
```python
final_score = (
    similarity_score × 0.65 +    # Pertinence sémantique
    rating_weight × 0.25 +       # Qualité du film
    popularity × 0.10            # Tendance actuelle
) × doc_penalty                  # Pénalité documentaires
```

**Justification:**
- **65% sémantique**: Priorité à la pertinence
- **25% rating**: Favorise les films de qualité
- **10% popularité**: Boost films connus
- **Pénalité doc**: -15% pour documentaires (biais dataset)

## 6. Évaluation

### Dataset de Test
20 requêtes avec ground truth manuelle:
```python
[
    ("romantic movie on a sinking cruise ship", "Titanic"),
    ("AI falls in love with lonely writer", "Her"),
    ...
]
```

### Métriques

**MRR (Mean Reciprocal Rank)**
```
MRR = (1/n) × Σ(1/rank_i)

Exemple:
  Query 1: Film attendu en position 1 → 1/1 = 1.0
  Query 2: Film attendu en position 3 → 1/3 = 0.33
  Query 3: Film attendu en position 2 → 1/2 = 0.5
  MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
```

**Precision@K**
```
P@K = Nombre de résultats pertinents dans top-K / K

Exemple (K=5):
  Top 5: [Titanic, Film A, Film B, Inception, Film C]
  Pertinents: Titanic, Inception
  P@5 = 2/5 = 40%
```

**Recall@K**
```
R@K = Nombre de résultats pertinents trouvés / Total pertinents

Pour requêtes avec 1 film pertinent:
  R@10 = 1 si trouvé dans top-10, sinon 0
```

### Résultats Obtenus

| Métrique | Base | Fine-tuné | Δ |
|----------|------|-----------|---|
| MRR | 0.407 | 0.611 | +50.1% |
| P@1 | 30% | 50% | +66.7% |
| P@3 | 55% | 65% | +18.2% |
| P@5 | 55% | 70% | +27.3% |
| R@10 | 60% | 85% | +41.7% |

## 7. Analyse des Échecs

### Requêtes Échouées

**"time loop comedy repeat same day" → Groundhog Day**
- Raison: Film absent ou peu représenté dans training
- Solution: Ajout d'exemples spécifiques ou data augmentation

**"underground fight club soap maker" → Fight Club**
- Raison: Concepts très spécifiques non vus à l'entraînement
- Solution: Anchoring avec requêtes iconiques

### Limitations Identifiées

1. **Noms propres**: WALL-E, Groundhog Day difficiles
2. **Concepts ultra-spécifiques**: "soap maker"
3. **Biais dataset**: Sur-représentation d'Action/Drama

## 8. Conclusion

L'approche par curriculum learning avec MNRL a permis:
- +50% MRR (amélioration substantielle)
- Compréhension sémantique robuste
- Généralisation sur requêtes non vues
- Temps d'entraînement raisonnable (6 min)

**Améliorations futures:**
- Anchoring pour films iconiques
- Hard negative mining
- Augmentation de données
- Utilisation de modèles plus larges (BERT-base)
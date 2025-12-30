# NeuralMate2 - Architecture Technique

## Vue d'Ensemble

NeuralMate2 est un moteur d'échecs par reinforcement learning qui combine :
1. **Pretraining supervisé** sur 334 MB de parties de maîtres
2. **Self-play** pour dépasser le niveau humain

---

## Innovations par rapport à AlphaZero Classique

### 1. Architecture Réseau : SE-ResNet + Attention

**Problème AlphaZero classique** : Les convolutions standard ont une réceptivité locale limitée.

**Solution** : Architecture hybride avec :
- **Squeeze-and-Excitation (SE)** : Recalibration adaptative des canaux
- **Spatial Attention** dans les blocs finaux : Vision globale du plateau
- **Group Normalization** au lieu de Batch Norm : Stable avec petits batches
- **GELU activation** : Gradients plus lisses que ReLU

```
Input (54 planes) → Conv 3x3 → [SE-ResBlock × 10] → [SE-ResBlock + Attention × 2]
    ↑                                                           ↓
    │                                             ┌─────────────┴─────────────┐
    │                                             ↓                           ↓
 3 positions                               Policy Head                 Value Head
 d'historique                              (1858 moves)               (score -1/+1)
```

**Pourquoi 54 planes (3 positions d'historique) ?**
- Détection des répétitions (règle des 3 coups)
- Meilleure compréhension du flux de la partie
- Gain estimé : +50-100 ELO vs 18 planes
- Coût mémoire acceptable pour RTX 3060

### 2. MCTS Amélioré : PUCT + Transpositions

**Innovations** :
- **Virtual losses** : Parallélisation efficace sur GPU
- **Transposition table** : Cache des positions déjà évaluées
- **Progressive widening** : Focus sur les coups prometteurs
- **First Play Urgency (FPU)** : Réduction pour coups non explorés
- **Gumbel noise** : Alternative au Dirichlet pour l'exploration

### 3. Pretraining Efficace

**Chunked PGN Processing** :
- Lecture par blocs de 10000 parties (évite OOM sur 334 MB)
- Filtrage ELO ≥ 2200
- Extraction position → (coup joué, résultat)
- Cache HDF5 pour rechargement rapide

**Curriculum Learning** :
- Phase 1 : Positions tactiques évidentes
- Phase 2 : Milieu de partie complexe
- Phase 3 : Finales et technique

### 4. Training Optimisé RTX 3060

- **Mixed Precision (FP16)** : 2× vitesse, 50% VRAM
- **Gradient Checkpointing** : Modèles plus grands possibles
- **Efficient DataLoader** : Prefetch async avec pinned memory
- **Cosine Annealing** : LR schedule avec warm restarts

---

## Structure des Modules

```
src/
├── alphazero/
│   ├── __init__.py          # Exports publics
│   ├── network.py           # DualHeadNetwork (SE-ResNet + Attention)
│   ├── mcts.py              # MCTS avec virtual losses et transpositions
│   ├── trainer.py           # AlphaZeroTrainer + TrainingConfig
│   ├── arena.py             # Évaluation : NetworkPlayer, RandomPlayer
│   ├── move_encoding.py     # Encodage/décodage des coups (1858 classes)
│   ├── spatial_encoding.py  # Encodage du plateau (18/54 planes)
│   └── replay_buffer.py     # Buffer circulaire avec sampling prioritaire
│
├── pretraining/
│   ├── __init__.py
│   ├── pgn_processor.py     # Lecture chunked du PGN
│   ├── dataset.py           # ChessPositionDataset pour PyTorch
│   └── pretrain.py          # Script de pretraining
│
├── chess_encoding/
│   ├── __init__.py
│   └── board_utils.py       # Utilitaires (material diff, etc.)
│
├── ui/                      # [EXISTANT - ne pas modifier]
│   ├── app.py
│   ├── board_widget.py
│   ├── styles.py
│   └── ...
│
└── play.py                  # Point d'entrée principal
```

---

## Encodage du Plateau (54 planes par défaut)

L'encodage utilise **54 planes** avec 3 positions d'historique pour une meilleure compréhension contextuelle.

### Structure des 54 Planes

```
┌─────────────────────────────────────────────────────────────────┐
│ POSITION ACTUELLE (T=0) - 12 planes                             │
├─────────────────────────────────────────────────────────────────┤
│ Planes 0-5   : Pièces blanches (P, N, B, R, Q, K)               │
│ Planes 6-11  : Pièces noires (P, N, B, R, Q, K)                 │
├─────────────────────────────────────────────────────────────────┤
│ POSITION T-1 (1 coup avant) - 12 planes                         │
├─────────────────────────────────────────────────────────────────┤
│ Planes 12-17 : Pièces blanches                                  │
│ Planes 18-23 : Pièces noires                                    │
├─────────────────────────────────────────────────────────────────┤
│ POSITION T-2 (2 coups avant) - 12 planes                        │
├─────────────────────────────────────────────────────────────────┤
│ Planes 24-29 : Pièces blanches                                  │
│ Planes 30-35 : Pièces noires                                    │
├─────────────────────────────────────────────────────────────────┤
│ MÉTADONNÉES - 18 planes                                         │
├─────────────────────────────────────────────────────────────────┤
│ Plane 36     : Couleur au trait (1=blanc, 0=noir)               │
│ Plane 37     : Compteur de coups totaux (normalisé)             │
│ Plane 38     : Roque roi blanc disponible                       │
│ Plane 39     : Roque dame blanc disponible                      │
│ Plane 40     : Roque roi noir disponible                        │
│ Plane 41     : Roque dame noir disponible                       │
│ Plane 42     : Compteur 50 coups (normalisé)                    │
│ Plane 43     : Case en passant (si applicable)                  │
│ Planes 44-45 : Répétition (1 si position vue 1×, 2×)            │
│ Planes 46-53 : Réservés pour extensions futures                 │
└─────────────────────────────────────────────────────────────────┘
```

### Normalisation

- Toutes les valeurs sont dans [0, 1]
- Les pièces : 1.0 si présente, 0.0 sinon
- Compteur 50 coups : valeur / 100
- Compteur de coups : min(fullmove_number, 200) / 200

### Perspective

L'encodage est **toujours du point de vue du joueur au trait** :
- Si c'est aux noirs de jouer, le plateau est "retourné" (flip vertical)
- Les pièces "amies" sont toujours dans les planes 0-5
- Les pièces "ennemies" sont toujours dans les planes 6-11

---

## Encodage des Coups (1858 classes)

Système "Queen-like moves" d'AlphaZero :
- 56 × 8 = 448 mouvements "queen" par case (8 directions × 7 distances)
- 64 cases de départ = 3584 possibilités
- Filtré aux 1858 coups légaux maximum

Encodage compact :
```python
move_index = from_square * 73 + direction * 7 + (distance - 1) + knight_offset + promo_offset
```

---

## Hyperparamètres Recommandés

### Réseau
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **Input planes** | **54** | 3 positions d'historique (défaut) |
| Residual blocks | 12 | Bon ratio performance/vitesse |
| Filters | 192 | Optimal pour RTX 3060 |
| SE reduction | 8 | Standard pour SE-ResNet |
| History length | 3 | Détection répétitions + contexte |

### MCTS
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Simulations | 800 | Bon compromis force/vitesse |
| c_puct | 1.5 | Exploration standard |
| Dirichlet alpha | 0.3 | Échecs (0.03 pour Go) |
| Dirichlet epsilon | 0.25 | 25% bruit racine |
| Temperature | 1.0 → 0.1 | Annealing après 30 coups |

### Training
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Batch size | 256 | RTX 3060 (12 GB) |
| Learning rate | 0.01 → 0.0001 | Cosine annealing |
| Weight decay | 1e-4 | Régularisation L2 |
| Epochs per iter | 3 | Évite l'overfitting |
| Games per iter | 100 | Données fraîches |

### Pretraining
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Chunk size | 10000 parties | Mémoire maîtrisée |
| Min ELO | 2200 | Qualité des parties |
| Epochs | 5-10 | Évite l'overfitting |
| LR | 0.001 | Plus doux que self-play |

---

## Pipeline d'Entraînement

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1 : PRETRAINING                        │
├─────────────────────────────────────────────────────────────────┤
│  lichess_elite.pgn                                              │
│        ↓                                                        │
│  PGN Processor (chunks de 10k parties)                          │
│        ↓                                                        │
│  ChessPositionDataset                                           │
│        ↓                                                        │
│  DualHeadNetwork.train() sur (position, policy, value)          │
│        ↓                                                        │
│  Checkpoint: pretrained_model.pt                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2 : SELF-PLAY                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Iteration N                                             │   │
│  │  ┌────────────┐    ┌──────────────┐    ┌─────────────┐   │   │
│  │  │ Self-Play  │ →  │ Replay Buffer│ →  │  Training   │   │   │
│  │  │ (100 games)│    │ (500k pos)   │    │ (3 epochs)  │   │   │
│  │  └────────────┘    └──────────────┘    └─────────────┘   │   │
│  │        ↓                                      ↓          │   │
│  │  MCTS + Network                       Updated Network    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Evaluation vs Previous Best                             │   │
│  │  Win rate > 55% → Nouveau champion                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Commandes Principales

```bash
# Pretraining sur parties de maîtres
python -m src.pretraining.pretrain --pgn data/lichess_elite_2020-08.pgn --epochs 5

# Self-play training (depuis zéro)
python -m src.alphazero.trainer --iterations 100 --games 100

# Self-play training (depuis pretrained)
python -m src.alphazero.trainer --checkpoint pretrained.pt --iterations 100

# Lancer l'interface graphique
make && ./neural_mate_play

# Diagnostics réseau
python diagnose_network.py models/latest.pt

# Match entre deux réseaux
python -m src.alphazero.arena --model1 best.pt --model2 challenger.pt --games 50
```

---

## Dépendances

```
torch>=2.0.0
numpy>=1.24.0
python-chess>=1.9.0
h5py>=3.8.0        # Cache des positions
```

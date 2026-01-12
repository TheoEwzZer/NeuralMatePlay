# NeuralMate2 - Architecture Technique

## Vue d'Ensemble

NeuralMate2 est un moteur d'échecs par reinforcement learning qui combine :

1. **Pretraining supervisé** sur 334 MB de parties de maîtres
2. **Self-play** pour dépasser le niveau humain

---

## Innovations par rapport à AlphaZero Classique

### 1. Architecture Réseau : SE-ResNet + Attention + WDL

**Problème AlphaZero classique** : Les convolutions standard ont une réceptivité locale limitée.

**Solution** : Architecture hybride avec :

- **Squeeze-and-Excitation (SE)** : Recalibration adaptative des canaux
- **Spatial Attention** dans les blocs finaux : Vision globale du plateau
- **Group Normalization** au lieu de Batch Norm : Stable avec petits batches
- **GELU activation** : Gradients plus lisses que ReLU
- **WDL Head** : Sortie Win/Draw/Loss au lieu d'un scalaire

```
Input (68 planes) → Conv 3x3 → [SE-ResBlock × 10] → [SE-ResBlock + Attention × 2]
    ↑                                                           ↓
    │                                             ┌─────────────┴─────────────┐
    │                                             ↓                           ↓
 4 positions                               Policy Head                  WDL Head
 (T, T-1, T-2, T-3)                        (4672 moves)            (Win/Draw/Loss)
 + 8 semantic planes
```

**Pourquoi 68 planes ?**

- 48 planes d'historique (4 positions × 12 pièces)
- 12 planes de métadonnées (roque, en passant, répétitions, etc.)
- 8 planes sémantiques NNUE-style (attaquants roi, mobilité, pions passés, etc.)
- Détection des répétitions (règle des 3 coups)
- Gain estimé : +100-150 ELO vs architecture classique

### 2. MCTS Amélioré : PUCT + Transpositions

**Innovations** :

- **Virtual losses** : Parallélisation efficace sur GPU
- **Transposition table** : Cache des positions déjà évaluées
- **Progressive widening** : Focus sur les coups prometteurs
- **First Play Urgency (FPU)** : Réduction pour coups non explorés
- **Gumbel noise** : Alternative au Dirichlet pour l'exploration

### 3. Pretraining Efficace

**Chunked PGN Processing** :

- Lecture par blocs de 20000 positions (chunks HDF5)
- Filtrage ELO ≥ 2200
- Skip des 12 premiers coups (évite théorie d'ouverture)
- Extraction position → (coup joué, WDL)
- Streaming trainer pour gros datasets

### 4. Training Optimisé RTX 3060

- **Mixed Precision (FP16)** : 2× vitesse, 50% VRAM
- **Gradient Checkpointing** : Modèles plus grands possibles
- **Efficient DataLoader** : Prefetch async avec pinned memory (2 workers)
- **ReduceLROnPlateau** : LR réduit de 50% après 3 epochs sans amélioration
- **Gradient Accumulation** : 2 steps pour batch effectif plus grand

---

## Structure des Modules

```
src/
├── alphazero/
│   ├── __init__.py              # Exports publics
│   ├── network.py               # DualHeadNetwork (SE-ResNet + WDL)
│   ├── mcts.py                  # MCTS avec virtual losses et transpositions
│   ├── trainer.py               # AlphaZeroTrainer + TrainingConfig
│   ├── train.py                 # Point d'entrée CLI pour self-play training
│   ├── arena.py                 # Évaluation : NetworkPlayer, RandomPlayer
│   ├── move_encoding.py         # Encodage/décodage des coups (4672 classes)
│   ├── spatial_encoding.py      # Encodage du plateau (68 planes)
│   ├── replay_buffer.py         # Buffer circulaire avec sampling prioritaire
│   ├── device.py                # Gestion GPU/CPU et device selection
│   └── checkpoint_manager.py    # Gestion des checkpoints et versioning
│
├── pretraining/
│   ├── __init__.py
│   ├── pgn_processor.py         # Lecture chunked du PGN
│   ├── dataset.py               # ChessPositionDataset pour PyTorch
│   ├── pretrain.py              # Script de pretraining
│   ├── streaming_trainer.py     # Trainer streaming pour gros datasets
│   └── chunk_manager.py         # Gestion des chunks HDF5
│
├── chess_encoding/
│   ├── __init__.py
│   └── board_utils.py           # Utilitaires (material diff, etc.)
│
├── ui/
│   ├── app.py                   # Application principale
│   ├── board_widget.py          # Widget échiquier
│   ├── styles.py                # Styles Qt
│   ├── training_app.py          # Application training avec UI
│   ├── training_panel.py        # Panel de contrôle training
│   └── match_app.py             # Application pour matchs entre modèles
│
├── config.py                    # Configuration centralisée (JSON)
└── play.py                      # Point d'entrée principal
```

---

## Encodage du Plateau (68 planes)

L'encodage utilise **68 planes** : 4 positions d'historique + métadonnées + features sémantiques NNUE-style.

**Formule** : `(history_length + 1) × 12 + 20 = (3 + 1) × 12 + 20 = 68 planes`

### Structure des 68 Planes

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
│ POSITION T-3 (3 coups avant) - 12 planes                        │
├─────────────────────────────────────────────────────────────────┤
│ Planes 36-41 : Pièces blanches                                  │
│ Planes 42-47 : Pièces noires                                    │
├─────────────────────────────────────────────────────────────────┤
│ MÉTADONNÉES - 12 planes                                         │
├─────────────────────────────────────────────────────────────────┤
│ Plane 48     : Couleur au trait (1=blanc, 0=noir)               │
│ Plane 49     : Compteur de coups totaux (normalisé)             │
│ Plane 50     : Roque roi blanc disponible                       │
│ Plane 51     : Roque dame blanc disponible                      │
│ Plane 52     : Roque roi noir disponible                        │
│ Plane 53     : Roque dame noir disponible                       │
│ Plane 54     : Compteur 50 coups (normalisé)                    │
│ Plane 55     : Case en passant (si applicable)                  │
│ Planes 56-57 : Répétition (1 si position vue 1×, 2×)            │
│ Planes 58-59 : Cartes d'attaque (cases attaquées)               │
├─────────────────────────────────────────────────────────────────┤
│ FEATURES SÉMANTIQUES NNUE-STYLE - 8 planes                      │
├─────────────────────────────────────────────────────────────────┤
│ Plane 60     : Attaquants du roi adverse                        │
│ Plane 61     : Défenseurs de mon roi                            │
│ Plane 62     : Mobilité des cavaliers                           │
│ Plane 63     : Mobilité des fous                                │
│ Plane 64     : Pions passés                                     │
│ Plane 65     : Pions isolés                                     │
│ Plane 66     : Cases faibles (non défendues)                    │
│ Plane 67     : Contrôle du centre (d4, d5, e4, e5)              │
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

## Encodage des Coups (4672 classes)

Système "Queen-like moves" d'AlphaZero :

- **64 cases de départ × 73 types de coups = 4672 indices**

### Types de coups (73 par case)

| Type            | Nombre | Description                |
| --------------- | ------ | -------------------------- |
| Queen-like      | 56     | 8 directions × 7 distances |
| Knight          | 8      | 8 sauts en L               |
| Underpromotions | 9      | 3 pièces × 3 directions    |

### Encodage

```python
# Index = case_départ × 73 + type_coup
base_idx = from_square * 73

# Queen-like moves: direction × 7 + (distance - 1)
idx = base_idx + direction_idx * 7 + (distance - 1)

# Knight moves: offset 56 + knight_idx
idx = base_idx + 56 + knight_idx

# Underpromotions: offset 64 + piece_idx × 3 + direction_idx
idx = base_idx + 64 + piece_idx * 3 + direction_idx
```

Les promotions en dame sont encodées comme mouvements queen-like normaux.

---

## WDL Head (Win/Draw/Loss)

Le réseau utilise une **WDL Head** au lieu d'une sortie scalaire :

### Architecture

```
Backbone features → Conv(192→8) → Flatten → FC(512→512) → FC(512→3) → Softmax
                                                                         ↓
                                                            [P(Win), P(Draw), P(Loss)]
```

### Avantages

- **Meilleure calibration** des positions de nulle (+100-150 Elo)
- **Information plus riche** que scalaire unique
- **Utilisé par Leela Chess Zero** avec succès prouvé

### Calcul de la Value

```python
# Value reconstruite pour MCTS
value = P(Win) - P(Loss)  # dans [-1, 1]
```

### Loss Function

```python
# CrossEntropy au lieu de MSE
wdl_loss = CrossEntropy(wdl_logits, wdl_target)
# wdl_target: [1,0,0]=win, [0,1,0]=draw, [0,0,1]=loss
```

---

## Hyperparamètres Recommandés

### Réseau

| Paramètre        | Valeur  | Justification                                |
| ---------------- | ------- | -------------------------------------------- |
| **Input planes** | **68**  | 48 history + 12 metadata + 8 semantic        |
| Residual blocks  | 12      | Bon ratio performance/vitesse                |
| Filters          | 192     | Optimal pour RTX 3060                        |
| SE reduction     | 8       | Standard pour SE-ResNet                      |
| Attention heads  | 4       | Multi-head attention spatiale                |
| History length   | 3       | Détection répétitions + contexte (fixe)      |
| WDL Head         | enabled | Win/Draw/Loss (obligatoire)                  |

### MCTS

| Paramètre         | Valeur    | Justification            |
| ----------------- | --------- | ------------------------ |
| Simulations       | 100       | Rapide pour self-play    |
| c_puct            | 1.5       | Exploration standard     |
| Dirichlet alpha   | 0.3       | Échecs (0.03 pour Go)    |
| Dirichlet epsilon | 0.25      | 25% bruit racine         |
| Temperature       | 1.0 → 0.1 | Annealing après 30 coups |
| FPU reduction     | 0.25      | First Play Urgency       |

### Training (Self-play)

| Paramètre       | Valeur | Justification                 |
| --------------- | ------ | ----------------------------- |
| Batch size      | 640    | RTX 3060 Laptop (6 GB)        |
| Learning rate   | 0.01   | Initial LR                    |
| LR decay        | 0.95   | Decay factor                  |
| Min LR          | 1e-5   | Floor LR                      |
| Weight decay    | 1e-4   | Régularisation L2             |
| Epochs per iter | 3      | Évite l'overfitting           |
| Games per iter  | 100    | Données fraîches              |
| Buffer size     | 500k   | Replay buffer capacity        |
| Recent weight   | 0.8    | Priorité aux données récentes |

### Pretraining

| Paramètre         | Valeur          | Justification              |
| ----------------- | --------------- | -------------------------- |
| Chunk size        | 20000 positions | Chunks HDF5                |
| Min ELO           | 2200            | Qualité des parties        |
| Skip first N      | 12 coups        | Évite théorie d'ouverture  |
| Epochs            | 5               | Évite l'overfitting        |
| Batch size        | 640             | RTX 3060 optimisé          |
| LR                | 0.0001          | Plus doux que self-play    |
| WDL loss weight   | 5.0             | Poids WDL vs policy        |
| Entropy coef      | 0.01            | Encourage diversité policy |
| Gradient accum    | 2 steps         | Batch effectif = 1280      |
| Patience          | 5 epochs        | Early stopping             |

---

## Pipeline d'Entraînement

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1 : PRETRAINING                        │
├─────────────────────────────────────────────────────────────────┤
│  lichess_elite.pgn                                              │
│        ↓                                                        │
│  PGN Processor → Chunks HDF5 (20k positions/chunk)              │
│        ↓                                                        │
│  Streaming Trainer (prefetch async)                             │
│        ↓                                                        │
│  DualHeadNetwork.train() sur (position, policy, WDL)            │
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
# Pretraining sur parties de maîtres (utilise config/config.json)
python -m src.pretraining.pretrain --config config/config.json

# Pretraining avec options CLI
python -m src.pretraining.pretrain --pgn data/lichess_elite_2020-08.pgn --epochs 5 --batch-size 640

# Self-play training (depuis zéro)
python -m src.alphazero.train --iterations 100 --games 100

# Self-play training (depuis pretrained)
python -m src.alphazero.train --checkpoint pretrained.pt --iterations 100

# Self-play training avec config JSON
python -m src.alphazero.train --config config/config.json

# Lancer l'interface de jeu
python -m src.play

# Lancer l'interface d'entraînement
python -m src.ui.training_app

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

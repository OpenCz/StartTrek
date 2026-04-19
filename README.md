# StartTrek — Reinforcement Learning

DQN-based reinforcement learning agent trained to autonomously land a lunar module using OpenAI Gymnasium's LunarLander-v3 environment. Built with PyTorch — achieves mean score ≥ 200 over 100 consecutive episodes.

---

## Arborescence du projet

```
StartTrek/
│
├── train.py                  # Script principal d'entraînement
├── eval.py                   # Script d'évaluation du modèle
├── agent.py                  # Architecture DQN + replay buffer
├── README.md                 # Documentation du projet
├── requirements.txt          # Dépendances Python
├── repro.sh                  # Script one-click de reproduction
│
├── configs/
│   ├── default.yaml          # Hyperparamètres principaux
│   └── ablation_buffer.yaml  # Config pour ablation (buffer size)
│
├── plots/
│   ├── returns.png           # Courbe des récompenses par épisode
│   ├── epsilon.png           # Courbe du decay d'epsilon
│   └── loss.png              # Courbe de la loss du réseau
│
├── videos/
│   ├── random_policy.mp4     # Baseline : agent aléatoire
│   └── trained_agent.mp4     # Agent entraîné qui atterrit
│
├── results/
│   └── metrics.csv           # Métriques brutes par épisode et seed
│
└── checkpoints/
    └── best_model.pt         # Meilleur modèle sauvegardé (PyTorch)
```

---

## Description des fichiers

### `train.py`
Script principal à lancer pour entraîner l'agent.
- Charge la configuration depuis un fichier YAML (`configs/default.yaml`)
- Crée l'environnement Gymnasium (`LunarLander-v3`)
- Lance la boucle d'entraînement : `reset → action → reward → apprentissage`
- Log les métriques (score, episode length, cause de fin) dans `results/metrics.csv`
- Génère les plots dans `plots/`
- Sauvegarde le meilleur modèle dans `checkpoints/best_model.pt`

```bash
python train.py --config configs/default.yaml
```

---

### `eval.py`
Script d'évaluation du modèle entraîné.
- Charge le modèle sauvegardé (`checkpoints/best_model.pt`)
- Fait jouer l'agent sans apprentissage (epsilon = 0)
- Mesure le score moyen sur 100 épisodes consécutifs
- Génère les vidéos dans `videos/`
- Affiche la distribution des scores et la variance

```bash
python eval.py --model checkpoints/best_model.pt
```

---

### `agent.py`
Le cerveau de l'agent — contient toute la logique DQN.
- **Réseau MLP** : 2–3 couches cachées avec activation ReLU (entrée : 8 dims, sortie : 4 actions)
- **Replay Buffer** : stocke les transitions `(state, action, reward, next_state, done)` pour briser les corrélations temporelles
- **Target Network** : copie du réseau principal, mise à jour périodiquement pour stabiliser l'apprentissage
- **Epsilon-greedy** : décroissance d'epsilon de 1.0 → 0.01 pour équilibrer exploration et exploitation
- **Méthode `act(state)`** : choisit une action selon la politique courante
- **Méthode `learn()`** : calcule la loss Bellman et met à jour les poids via backprop

---

### `README.md`
Ce fichier. Documentation complète du projet.

---

### `requirements.txt`
Liste des dépendances Python nécessaires pour faire tourner le projet.

```bash
pip install -r requirements.txt
```

Généré avec :
```bash
pip freeze > requirements.txt
```

---

### `repro.sh`
Script de reproduction one-click — exigé par le sujet.
- Entraîne l'agent sur 5 seeds (`0, 1, 2, 3, 4`)
- Lance l'évaluation sur chaque seed
- Régénère tous les plots et CSV
- Les résultats doivent être reproductibles à ±5% avec les mêmes seeds et configs

```bash
bash repro.sh
```

---

### `configs/default.yaml`
Fichier de configuration principal. Contient tous les hyperparamètres séparés du code.

| Paramètre | Valeur | Description |
|---|---|---|
| `lr` | `0.001` | Learning rate de l'optimiseur Adam |
| `gamma` | `0.99` | Facteur de discount (importance des récompenses futures) |
| `batch_size` | `64` | Nombre d'expériences tirées du replay buffer par update |
| `buffer_size` | `50000` | Capacité maximale du replay buffer |
| `epsilon_start` | `1.0` | Epsilon initial (100% exploration) |
| `epsilon_end` | `0.01` | Epsilon final (1% exploration, 99% exploitation) |
| `epsilon_decay` | `0.995` | Multiplicateur appliqué à epsilon après chaque épisode |
| `target_update` | `10` | Fréquence (en épisodes) de mise à jour de la target network |
| `seeds` | `[0,1,2,3,4]` | Seeds pour la reproductibilité |
| `max_episodes` | `2000` | Nombre maximum d'épisodes d'entraînement |

---

### `configs/ablation_buffer.yaml`
Config identique à `default.yaml` mais avec `buffer_size` modifié.
Utilisée pour l'ablation : comparer les performances avec un buffer petit (1 000) vs grand (200 000) vs défaut (50 000).

---

### `plots/returns.png`
Courbe des récompenses moyennes par épisode (moving average sur 100 épisodes).
- Axe X : numéro d'épisode
- Axe Y : score moyen
- Objectif : atteindre ≥ 200 de manière stable
- Généré automatiquement par `train.py`

---

### `plots/epsilon.png`
Courbe du decay d'epsilon au fil des épisodes.
- Montre comment l'agent passe progressivement de l'exploration pure (epsilon = 1.0) à l'exploitation de sa politique apprise (epsilon = 0.01)
- Généré automatiquement par `train.py`

---

### `plots/loss.png`
Courbe de la loss du réseau de neurones (MSE entre Q-values prédites et Q-values cibles).
- Doit diminuer et se stabiliser au fil du temps
- Une loss qui explose indique un problème dans le DQN (learning rate trop élevé, pas de target network, etc.)
- Généré automatiquement par `train.py`

---

### `videos/random_policy.mp4`
Vidéo de la baseline aléatoire : la fusée effectue des actions aléatoires et crashe systématiquement.
Sert de référence visuelle pour montrer le progrès de l'agent entraîné.

---

### `videos/trained_agent.mp4`
Vidéo de l'agent entraîné en action.
Montre la fusée corriger sa trajectoire et atterrir proprement sur la plateforme.
Générée par `eval.py` avec `render_mode="rgb_array"`.

---

### `results/metrics.csv`
Fichier CSV contenant les métriques brutes de chaque épisode d'entraînement.

| Colonne | Description |
|---|---|
| `episode` | Numéro de l'épisode |
| `seed` | Seed utilisée |
| `score` | Récompense totale de l'épisode |
| `episode_length` | Nombre de steps avant fin |
| `termination_cause` | Raison de fin : `crash`, `out-of-view`, ou `sleep` |
| `epsilon` | Valeur d'epsilon à cet épisode |

---

### `checkpoints/best_model.pt`
Fichier PyTorch contenant les poids du meilleur modèle entraîné.
Sauvegardé automatiquement par `train.py` quand le score moyen sur 100 épisodes dépasse le record.
Chargé par `eval.py` pour l'évaluation finale.

---

## Entraînement

```bash
python train.py --config configs/default.yaml
```

---

## Évaluation

```bash
python eval.py --model checkpoints/best_model.pt
```

---

## Reproduction complète

```bash
bash repro.sh
```

---

## Critères de succès

- Score moyen ≥ 200 sur 100 épisodes consécutifs
- Reproductibilité à ±5% avec les mêmes seeds
- Logging de la cause de fin d'épisode (`crash`, `out-of-view`, `sleep`)
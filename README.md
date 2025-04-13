# NaviLearn

NaviLearn est un système d'apprentissage par renforcement pour la navigation collective adaptative. Il simule des agents autonomes qui apprennent à naviguer dans des environnements complexes, à collecter des ressources et à collaborer entre eux.

![NaviLearn Demo](docs/images/navilearn_demo.png)

## 🌟 Caractéristiques

- **Apprentissage par renforcement avancé**: Implémentation des algorithmes DQN et PPO avec diverses améliorations
- **Navigation collective**: Les agents peuvent communiquer et collaborer pour atteindre des objectifs communs
- **Environnement dynamique**: Obstacles, ressources, conditions météorologiques variables
- **Visualisation en temps réel**: Représentation graphique 2D/3D avec contrôles interactifs
- **Architecture modulaire**: Facilement extensible pour ajouter de nouvelles fonctionnalités

## 📋 Prérequis

- Python 3.8+
- PyTorch 1.8+
- Pygame 2.0+
- NumPy
- OpenGL (optionnel, pour le rendu 3D)

## 🔧 Installation

1. Clonez le dépôt:
   ```bash
   git clone https://github.com/votre-username/navilearn.git
   cd navilearn
   ```

2. Installez les dépendances:
   ```bash
   pip install -r requirements.txt
   ```

3. Vérifiez l'installation:
   ```bash
   python main.py --config config.yaml --render
   ```

## 🚀 Utilisation

### Commandes de base

```bash
# Lancer une simulation avec visualisation
python main.py --config config.yaml --render

# Lancer une simulation sans visualisation (plus rapide pour l'entraînement)
python main.py --config config.yaml

# Spécifier un algorithme d'apprentissage particulier
python main.py --config config.yaml --algorithm ppo

# Charger un modèle pré-entraîné
python main.py --config config.yaml --load models/model_episode_50.pt

# Définir le niveau de journalisation
python main.py --config config.yaml --log-level DEBUG
```

### Configuration

NaviLearn utilise un fichier de configuration YAML pour définir tous les paramètres de la simulation. Vous pouvez personnaliser:

- La taille et la complexité de l'environnement
- Le nombre et les propriétés des agents
- Les paramètres de l'algorithme d'apprentissage
- Les options de visualisation et de rendu

Exemple de configuration (dans `config.yaml`):

```yaml
environment:
  width: 1000
  height: 1000
  obstacleCount: 30
  resourceCount: 15
  weatherEnabled: true

agents:
  count: 10
  sensorRange: 150
  communicationRange: 200

reinforcement_learning:
  algorithm: "dqn"
  stateDimension: 44
  actionDimension: 9
  learningRate: 0.001
  gamma: 0.99
  prioritizedExperience: true
```

### Contrôles de l'interface

Pendant la simulation avec visualisation, vous pouvez utiliser les contrôles suivants:

- **Espace**: Pause/Reprise de la simulation
- **+/-**: Augmenter/Diminuer la vitesse de simulation
- **Flèches Gauche/Droite**: Changer d'agent sélectionné
- **F**: Activer/Désactiver le suivi de l'agent sélectionné
- **G**: Activer/Désactiver la grille
- **P**: Activer/Désactiver l'affichage des trajectoires
- **S**: Activer/Désactiver l'affichage des capteurs
- **C**: Activer/Désactiver l'affichage des communications
- **R**: Réinitialiser la caméra
- **Souris**:
  - **Clic gauche**: Sélection d'agent
  - **Clic droit + Déplacement**: Rotation de la caméra
  - **Clic milieu + Déplacement**: Panoramique
  - **Molette**: Zoom avant/arrière

## 📊 Architecture du projet

```
navilearn/
├── agents/                  # Implémentation des agents
│   ├── baseAgent.py         # Classe de base pour tous les agents
│   ├── collectiveAgent.py   # Agent avec capacités collaboratives
│   ├── knowledgeMemory.py   # Mémoire de connaissances
│   └── smartNavigator.py    # Agent avec navigation intelligente
│
├── environment/             # Implémentation de l'environnement
│   ├── dynamicObstacle.py   # Obstacles statiques et dynamiques
│   ├── resourceNode.py      # Ressources à collecter
│   ├── terrainGenerator.py  # Génération procédurale de terrain
│   ├── weatherSystem.py     # Système météorologique dynamique
│   └── world.py             # Environnement de simulation
│
├── rl/                      # Algorithmes d'apprentissage
│   ├── dqnLearner.py        # Implémentation de DQN
│   ├── policyNetwork.py     # Réseau de politique
│   ├── ppoLearner.py        # Implémentation de PPO
│   ├── replayBuffer.py      # Tampons d'expérience
│   ├── rewardSystem.py      # Système de récompense
│   └── valueNetwork.py      # Réseau de valeur
│
├── utils/                   # Utilitaires divers
│   ├── configManager.py     # Gestion de la configuration
│   ├── logger.py            # Journalisation
│   ├── mathHelper.py        # Fonctions mathématiques
│   └── profiler.py          # Analyse de performances
│
├── visualization/           # Visualisation et interface
│   ├── animations.py        # Système d'animations
│   ├── camera.py            # Caméra pour la visualisation 3D
│   ├── guiManager.py        # Interface utilisateur
│   ├── renderer.py          # Moteur de rendu
│   └── shaders.py           # Shaders OpenGL
│
├── config.yaml              # Configuration par défaut
├── main.py                  # Point d'entrée principal
├── requirements.txt         # Dépendances du projet
└── README.md                # Documentation
```

## 🔬 Expérimentation

Pour conduire vos propres expériences, vous pouvez:

1. **Modifier les paramètres d'apprentissage**:
   Ajustez les hyperparamètres dans `config.yaml` pour explorer leur impact sur les performances.

2. **Créer de nouveaux environnements**:
   Modifiez les paramètres de génération d'environnement pour créer des défis plus complexes.

3. **Implémenter de nouveaux comportements d'agent**:
   Étendez la classe `CollectiveAgent` pour ajouter des comportements personnalisés.

4. **Comparer les algorithmes d'apprentissage**:
   Utilisez l'argument `--algorithm` pour comparer les performances de DQN et PPO.

## 🔁 Cycle d'entraînement typique

1. **Entraînement initial**:
   ```bash
   python main.py --config config.yaml --episodes 100
   ```

2. **Évaluation intermédiaire**:
   ```bash
   python main.py --config config.yaml --load models/model_episode_100.pt --render
   ```

3. **Entraînement supplémentaire**:
   ```bash
   python main.py --config config.yaml --load models/model_episode_100.pt --episodes 100
   ```

4. **Visualisation finale**:
   ```bash
   python main.py --config config.yaml --load models/model_episode_200.pt --render
   ```

## 📝 Analyse des résultats

NaviLearn génère des statistiques détaillées pendant l'entraînement:
- Taux de collecte de ressources
- Efficacité de navigation
- Niveaux de collaboration entre agents
- Convergence de l'apprentissage

Ces statistiques sont affichées dans la visualisation et enregistrées dans les fichiers de journalisation pour une analyse ultérieure.

## 🤝 Contribuer

Les contributions sont les bienvenues! Voici comment vous pouvez contribuer:

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/amazing-feature`)
3. Committez vos changements (`git commit -m 'Add some amazing feature'`)
4. Poussez vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 📮 Contact

Pour toute question, suggestion ou commentaire, n'hésitez pas à ouvrir une issue sur GitHub ou à contacter les mainteneurs directement.

---

Développé avec ❤️ par l'équipe NaviLearn
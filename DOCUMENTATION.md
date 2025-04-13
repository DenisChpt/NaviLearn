# NaviLearn: Documentation Technique

Ce document présente une description technique détaillée du fonctionnement interne du projet NaviLearn, un système d'apprentissage par renforcement pour la navigation collective adaptative.

## Table des matières

1. [Architecture globale](#architecture-globale)
2. [Système d'agents](#système-dagents)
3. [Environnement de simulation](#environnement-de-simulation)
4. [Algorithmes d'apprentissage](#algorithmes-dapprentissage)
5. [Système de récompense](#système-de-récompense)
6. [Tampons d'expérience](#tampons-dexpérience)
7. [Visualisation](#visualisation)
8. [Flux d'exécution](#flux-dexécution)
9. [Mécanismes de communication](#mécanismes-de-communication)
10. [Optimisation des performances](#optimisation-des-performances)
11. [Bonnes pratiques de développement](#bonnes-pratiques-de-développement)

## Architecture globale

NaviLearn adopte une architecture modulaire orientée objet avec séparation claire des responsabilités. Le système se compose de quatre composants principaux:

1. **Agents**: Entités autonomes qui perçoivent l'environnement, prennent des décisions et agissent.
2. **Environnement**: Monde simulé contenant des obstacles, des ressources et des conditions variables.
3. **Algorithmes d'apprentissage**: Mécanismes qui permettent aux agents d'apprendre des comportements optimaux.
4. **Visualisation**: Représentation graphique de la simulation avec contrôles interactifs.

Le flux de données principal suit le cycle classique de l'apprentissage par renforcement:
1. Les agents observent l'état actuel de l'environnement.
2. Ils sélectionnent des actions basées sur leur politique actuelle.
3. Ils exécutent ces actions dans l'environnement.
4. Ils reçoivent des récompenses et observent les nouveaux états.
5. Ils améliorent leur politique en fonction des expériences accumulées.

### Diagramme de classes simplifié

```
+----------------+     +----------------+     +----------------+
|     World      |<----+  BaseAgent     +---->|   DQNLearner   |
+----------------+     +----------------+     +----------------+
        ^                      ^                      ^
        |                      |                      |
+----------------+     +----------------+     +----------------+
| TerrainGenerator|    | SmartNavigator |    |  ReplayBuffer  |
+----------------+     +----------------+     +----------------+
        ^                      ^                      ^
        |                      |                      |
+----------------+     +----------------+     +----------------+
| WeatherSystem  |    |CollectiveAgent |    |PrioritizedReplay|
+----------------+     +----------------+     +----------------+
```

## Système d'agents

### Hiérarchie des agents

NaviLearn implémente une hiérarchie d'agents à trois niveaux:

1. **BaseAgent** (`agents/baseAgent.py`): Classe abstraite qui définit l'interface commune et les fonctionnalités de base pour tous les agents. Elle gère:
   - Position et orientation dans l'environnement
   - Mouvement de base et détection de collision
   - Capteurs simples pour détecter les objets environnants
   - Statistiques de performance (collisions, objectifs complétés)

2. **SmartNavigator** (`agents/smartNavigator.py`): Étend BaseAgent avec des capacités de navigation intelligente:
   - Capteurs plus sophistiqués avec champ de vision configurable
   - Stratégies de sélection de cibles (plus proche, plus précieuse, plus efficace)
   - Capacités d'évitement d'obstacles
   - Adaptation de la vitesse en fonction de l'environnement
   - Fonction de récompense adaptée à la navigation

3. **CollectiveAgent** (`agents/collectiveAgent.py`): Étend SmartNavigator avec des capacités collaboratives:
   - Communication avec d'autres agents dans un rayon configurable
   - Partage d'informations sur l'environnement (ressources, obstacles)
   - Coordination pour éviter la redondance d'exploration
   - Mémoire partagée et apprentissage social
   - Rôles dynamiques (explorateur, collecteur, éclaireur)

### Système de capteurs

Les agents perçoivent leur environnement à travers un système de capteurs implémenté dans `BaseAgent.scan()` et amélioré dans `SmartNavigator._updateSensors()`:

- Les capteurs sont distribués uniformément dans le champ de vision de l'agent (FOV)
- Chaque capteur détecte le type d'objet le plus proche dans sa direction (obstacle, ressource, bord)
- La portée de détection est configurable (sensorRange)
- Les informations des capteurs sont normalisées pour l'entrée des réseaux neuronaux

Code d'implémentation clé des capteurs:

```python
def _updateSensors(self) -> None:
    # Réinitialiser tous les capteurs à la portée maximale
    self.obstacleDetectors = np.ones(self.sensorCount) * self.sensorRange
    self.resourceDetectors = np.ones(self.sensorCount) * self.sensorRange
    self.edgeDetectors = np.ones(self.sensorCount) * self.sensorRange
    
    # Récupérer les objets détectés
    detected = self.scan()
    
    # Analyser les objets détectés
    for obj in detected:
        # Calculer dans quel capteur tombe l'objet
        objAngle = obj["angle"]
        
        # Normaliser l'angle dans la plage FOV
        halfFOV = self.sensorFOV / 2
        if objAngle < -halfFOV or objAngle > halfFOV:
            continue  # Objet en dehors du champ de vision
        
        # Convertir l'angle normalisé en indice de capteur
        normalizedAngle = (objAngle + halfFOV) / self.sensorFOV
        sensorIndex = min(self.sensorCount - 1, 
                        int(normalizedAngle * self.sensorCount))
        
        # Mettre à jour le capteur approprié
        if obj["type"] == "obstacle" and obj["distance"] < self.obstacleDetectors[sensorIndex]:
            self.obstacleDetectors[sensorIndex] = obj["distance"]
        elif obj["type"] == "resource" and obj["distance"] < self.resourceDetectors[sensorIndex]:
            self.resourceDetectors[sensorIndex] = obj["distance"]
        elif obj["type"] == "edge" and obj["distance"] < self.edgeDetectors[sensorIndex]:
            self.edgeDetectors[sensorIndex] = obj["distance"]
```

### Mémoire de connaissances

Les agents collectifs utilisent une structure de mémoire de connaissances (`agents/knowledgeMemory.py`) pour:

- Stocker et organiser les informations recueillies
- Suivre la fiabilité des informations avec un système de certitude
- Gérer la dégradation temporelle des connaissances
- Fusionner des informations provenant de différentes sources
- Organiser spatialement les connaissances pour un accès efficace

La mémoire utilise une structure de données avec les champs suivants pour chaque entrée:
- `type`: Type d'information (ressource, obstacle, etc.)
- `position`: Coordonnées (x, y) si applicable
- `timestamp`: Horodatage de l'observation
- `certainty`: Niveau de certitude (0-1)
- `source`: ID de l'agent ayant fourni l'information

## Environnement de simulation

### Structure du monde

L'environnement (`environment/world.py`) est un monde 2D contenant:

- Obstacles statiques et dynamiques
- Ressources à collecter avec différentes valeurs
- Effets météorologiques variables
- Terrain avec différentes caractéristiques

Le monde utilise une grille d'occupation pour optimiser la détection de collision:

```python
def _updateOccupancyGrid(self) -> None:
    self.occupancyGrid = {}
    
    # Ajouter les obstacles à la grille
    for obstacle in self.obstacles:
        cells = self._getGridCells(obstacle.position, obstacle.radius)
        
        for cell in cells:
            if cell not in self.occupancyGrid:
                self.occupancyGrid[cell] = []
            self.occupancyGrid[cell].append(("obstacle", obstacle))
    
    # Ajouter les ressources à la grille
    for resource in self.resources:
        if not resource.isCollected:
            cells = self._getGridCells(resource.position, resource.radius)
            
            for cell in cells:
                if cell not in self.occupancyGrid:
                    self.occupancyGrid[cell] = []
                self.occupancyGrid[cell].append(("resource", resource))
```

### Génération procédurale de terrain

Le terrain est généré de façon procédurale par `environment/terrainGenerator.py` en utilisant:

1. **Cartes de hauteur et d'humidité**: Générées avec du bruit de Perlin à plusieurs octaves
2. **Classification de terrain**: Définition des types de terrain (plaine, forêt, montagne, eau, marais) en fonction de la hauteur et de l'humidité
3. **Génération de caractéristiques**: Placement de montagnes, lacs, forêts et autres éléments
4. **Système de routes**: Création d'un réseau de routes connectant les points d'intérêt

### Système météorologique

Le système météorologique (`environment/weatherSystem.py`) simule différentes conditions:

- Différents types de météo: clair, nuageux, pluvieux, orageux, brumeux, neigeux, blizzard
- Transitions graduelles entre les conditions
- Effets sur la vitesse et la visibilité des agents
- Variations aléatoires pour plus de réalisme

Le système utilise une matrice de transition pour des changements météorologiques réalistes:

```python
transitionMatrix = {
    "clear": {"cloudy": 0.6, "foggy": 0.3, "rainy": 0.1, "stormy": 0.0, "snowy": 0.0, "blizzard": 0.0},
    "cloudy": {"clear": 0.4, "foggy": 0.2, "rainy": 0.3, "stormy": 0.1, "snowy": 0.0, "blizzard": 0.0},
    "rainy": {"clear": 0.1, "cloudy": 0.4, "foggy": 0.2, "stormy": 0.3, "snowy": 0.0, "blizzard": 0.0},
    # ...autres transitions...
}
```

## Algorithmes d'apprentissage

NaviLearn implémente deux algorithmes d'apprentissage par renforcement majeurs:

### DQN (Deep Q-Network)

Implémenté dans `rl/dqnLearner.py`, l'algorithme DQN comprend plusieurs améliorations:

1. **Double DQN**: Utilise un réseau cible distinct pour réduire la surestimation des valeurs Q
2. **Dueling Network Architecture**: Sépare l'estimation de la valeur d'état et de l'avantage des actions
3. **Prioritized Experience Replay**: Échantillonne les expériences en fonction de leur importance

Le cœur de l'algorithme est dans la méthode `learn()`, qui met à jour le réseau Q à partir des expériences:

```python
def learn(self) -> Optional[float]:
    # ... [code d'échantillonnage omis] ...
    
    # Calculer les Q-values pour l'état actuel
    qValues = self.qNetwork(states).gather(1, actions)
    
    # Calculer les Q-values cibles
    with torch.no_grad():
        if self.doubleDQN:
            # Double DQN: utiliser le réseau principal pour choisir l'action
            # et le réseau cible pour évaluer sa valeur
            nextActions = self.qNetwork(nextStates).argmax(1, keepdim=True)
            nextQValues = self.targetQNetwork(nextStates).gather(1, nextActions)
        else:
            # DQN standard: utiliser le réseau cible pour choisir et évaluer
            nextQValues = self.targetQNetwork(nextStates).max(1, keepdim=True)[0]
        
        # Calculer les valeurs cibles
        targets = rewards + (self.gamma * nextQValues * (1 - dones))
    
    # Calculer la perte et mettre à jour les priorités pour PER
    td_errors = targets - qValues
    loss = (weights * td_errors.pow(2)).mean()
    
    # ... [code d'optimisation omis] ...
    
    return loss.item()
```

### PPO (Proximal Policy Optimization)

Implémenté dans `rl/ppoLearner.py`, PPO est un algorithme de politique qui utilise:

1. **Clipping objective**: Limite les mises à jour de politique pour éviter les changements trop importants
2. **Generalized Advantage Estimation (GAE)**: Réduit la variance des estimations d'avantage
3. **Multiple epochs**: Effectue plusieurs passes d'optimisation sur les mêmes données

Le calcul des avantages avec GAE est implémenté ainsi:

```python
def _computeAdvantagesAndReturns(self) -> Tuple[torch.Tensor, torch.Tensor]:
    # ... [code de préparation omis] ...
    
    # Calculer les retours et avantages avec GAE
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    lastGae = 0
    
    # Parcourir les expériences en ordre inverse
    for t in reversed(range(len(rewards))):
        # Pour le dernier état, ou si l'épisode est terminé
        if t == len(rewards) - 1 or dones[t]:
            nextValue = 0
        else:
            nextValue = nextValues[t]
        
        # Delta: r + gamma * V(s') - V(s)
        delta = rewards[t] + self.gamma * nextValue * (1 - dones[t]) - values[t]
        
        # GAE: δt + (γλ)δt+1 + (γλ)²δt+2 + ...
        lastGae = delta + self.gamma * self.gaeParameter * (1 - dones[t]) * lastGae
        advantages[t] = lastGae
        
        # Retours: avantage + valeur
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns
```

### Réseaux neuronaux

NaviLearn utilise deux types de réseaux neuronaux:

1. **QNetwork** (`rl/valueNetwork.py`): Réseau pour estimer les valeurs Q des paires état-action
   - Peut utiliser une architecture duel pour séparer l'estimation de la valeur d'état et de l'avantage des actions

2. **PolicyNetwork** (`rl/policyNetwork.py`): Réseau pour estimer la distribution de probabilité des actions
   - Version discrète pour les espaces d'action finis
   - Version continue (ContinuousPolicyNetwork) pour les espaces d'action continus

## Système de récompense

Le système de récompense (`rl/rewardSystem.py`) est conçu pour encourager des comportements spécifiques:

### Récompenses de base

- **Collecte de ressources**: Récompense principale (+10.0)
- **Progression vers l'objectif**: Récompense incrémentale pour se rapprocher de l'objectif (+0.1)
- **Complétion d'objectif**: Bonus important (+50.0)

### Pénalités

- **Collisions**: Pénalité pour collision avec obstacle (-2.0)
- **Inactivité**: Pénalité pour absence de mouvement (-0.5)
- **Temps**: Légère pénalité par pas de temps pour encourager l'efficacité (-0.01)

### Récompenses adaptatives

- **Évitement d'obstacles**: Récompense pour naviguer près d'obstacles sans collision
- **Adaptation météorologique**: Récompense pour adapter sa vitesse aux conditions météo
- **Efficacité énergétique**: Récompense basée sur le ratio ressources/distance

### Récompenses collaboratives

- **Partage d'information**: Récompense pour le partage de connaissances (+0.3)
- **Actions coordonnées**: Récompense pour la coordination réussie (+0.5)
- **Performance de rôle**: Récompense spécifique au rôle (explorateur, collecteur, éclaireur)

## Tampons d'expérience

NaviLearn utilise quatre types de tampons d'expérience, définis dans `rl/replayBuffer.py`:

### ReplayBuffer (standard)

Tampon simple de taille fixe avec échantillonnage uniforme:
- Stockage d'expériences dans une file circulaire (deque)
- Échantillonnage aléatoire uniforme
- Méthode `sample()` qui renvoie 5 valeurs (états, actions, récompenses, états suivants, terminaisons)

### PrioritizedReplayBuffer

Tampon avec échantillonnage prioritaire:
- Stockage d'expériences avec priorités basées sur les erreurs TD
- Utilisation des priorités proportionnelles à l'erreur TD élevées à la puissance alpha
- Correction du biais d'échantillonnage via des poids d'importance-sampling
- Méthode `sample()` qui renvoie 7 valeurs (les 5 valeurs standard + poids + indices)
- Annealing du paramètre beta pour converger vers un échantillonnage sans biais

L'échantillonnage prioritaire est implémenté comme suit:

```python
def sample(self, batchSize: Optional[int] = None, device: torch.device = torch.device("cpu")):
    # Calculer les probabilités d'échantillonnage
    priorities = self.priorities[:memorySize]
    probabilities = priorities ** self.alpha
    probabilities = probabilities / np.sum(probabilities)
    
    # Échantillonner les indices selon les priorités
    indices = np.random.choice(memorySize, batchSize, replace=False, p=probabilities)
    
    # Calculer les poids d'importance-sampling
    weights = (memorySize * probabilities[indices]) ** (-self.beta)
    weights = weights / np.max(weights)  # Normaliser
    
    # Augmenter beta pour la convergence vers 1
    self.beta = min(1.0, self.beta + self.betaAnnealing)
    
    # ... [extraction des expériences omise] ...
    
    return states, actions, rewards, nextStates, dones, weights, indices
```

### MultiAgentReplayBuffer

Tampon pour environnements multi-agents:
- Maintient des buffers séparés pour chaque agent
- Échantillonnage équilibré entre les agents pour éviter le déséquilibre d'apprentissage
- Méthode `sample()` qui renvoie 5 valeurs comme le ReplayBuffer standard

L'échantillonnage équilibré est implémenté comme suit:

```python
def sample(self, batchSize: Optional[int] = None, agentId: Optional[Any] = None, device: torch.device = torch.device("cpu")):
    # ... [code pour le cas d'agent spécifique omis] ...
    
    if self.balanceAgents and len(self.agentMemories) > 1:
        # Échantillonnage équilibré entre agents
        experiences = []
        agentIds = list(self.agentMemories.keys())
        
        # Déterminer le nombre d'échantillons par agent
        samplesPerAgent = max(1, batchSize // len(agentIds))
        remainingSamples = batchSize - samplesPerAgent * len(agentIds)
        
        # Échantillonner pour chaque agent
        for aid in agentIds:
            agentMemory = self.agentMemories[aid]
            numSamples = min(samplesPerAgent, len(agentMemory))
            
            if numSamples > 0:
                agentSamples = random.sample(list(agentMemory), numSamples)
                experiences.extend(agentSamples)
        
        # ... [code pour échantillons supplémentaires omis] ...
```

### MultiAgentPrioritizedReplayBuffer

Tampon combinant les fonctionnalités multi-agents et prioritaires:
- Maintient des buffers prioritaires séparés pour chaque agent
- Échantillonnage équilibré entre agents tout en respectant les priorités
- Méthode `sample()` qui renvoie 7 valeurs comme PrioritizedReplayBuffer
- Maintient une correspondance entre indices d'agent et indices globaux

Ce tampon combine les avantages des deux approches précédentes, permettant:
1. Un apprentissage équilibré entre agents (même si certains agents ont plus d'expériences)
2. Une focalisation sur les expériences les plus instructives pour chaque agent

## Visualisation

Le système de visualisation utilise Pygame et optionnellement OpenGL pour le rendu 3D:

### Renderer

La classe `Renderer` (`visualization/renderer.py`) est responsable du rendu de la simulation:
- Initialise la fenêtre et le contexte OpenGL si disponible
- Gère le rendu 2D ou 3D selon la configuration
- Traite les entrées utilisateur
- Gère les captures d'écran et les animations

Le rendu s'adapte au niveau de détail graphique configuré, de minimal (0) à avancé (3):

```python
def _drawAgent(self, agent: Any, selected: bool) -> None:
    # ... [code de positionnement omis] ...
    
    # Dessiner le corps de l'agent
    if self.graphicalLevel <= 1:
        # Rendu simple: sphère
        self._drawSphere(agent.size, 12, 12)
    else:
        # Rendu détaillé: corps + direction
        
        # Corps
        self._drawSphere(agent.size, 16, 16)
        
        # Direction (cône)
        glPushMatrix()
        glTranslatef(0, 0, agent.size)
        glRotatef(90, 1, 0, 0)  # Orienter le cône vers l'avant
        self._drawCone(agent.size * 0.6, agent.size * 1.5, 12)
        glPopMatrix()
```

### Camera

La classe `Camera` (`visualization/camera.py`) gère la caméra 3D:
- Gère la position et l'orientation de la caméra
- Permet la rotation, le panoramique et le zoom
- Calcule les matrices de vue et de projection pour OpenGL

### GUIManager

La classe `GUIManager` (`visualization/guiManager.py`) gère l'interface utilisateur:
- Affiche les statistiques, contrôles et informations
- Gère les boutons, sliders et checkboxes
- Affiche les graphiques de performance

### ShaderManager et Animations

- `ShaderManager` (`visualization/shaders.py`) gère les shaders OpenGL pour les effets visuels avancés
- `AnimationManager` (`visualization/animations.py`) gère les animations pour une visualisation fluide

## Flux d'exécution

Le flux d'exécution principal est orchestré par la fonction `runSimulation` dans `main.py`:

```python
def runSimulation(world, agents, learner, renderer, episodes, logger, profiler):
    for episode in range(1, episodes + 1):
        logger.info(f"Début de l'épisode {episode}/{episodes}")
        world.reset()
        
        for agent in agents:
            agent.reset()
        
        stepCount = 0
        episodeComplete = False
        totalReward = 0.0
        
        # Mesure le temps d'exécution de l'épisode
        startTime = time.time()
        
        while not episodeComplete and stepCount < world.maxSteps:
            profiler.startSection("step_processing")
            
            # Mise à jour de l'environnement
            world.update()
            
            # Itération sur chaque agent
            actionsDict = {}
            
            for agent in agents:
                # Observation de l'environnement
                state = agent.observeEnvironment()
                
                # Décision d'action
                action = learner.getAction(agent, state)
                actionsDict[agent.agentId] = action
                
                # Exécution de l'action
                agent.executeAction(action)
                
                # Calcul de la récompense
                reward = agent.calculateReward()
                totalReward += reward
                
                # Nouvel état après action
                nextState = agent.observeEnvironment()
                
                # Vérifier si l'agent a terminé son objectif
                done = agent.hasCompletedObjective()
                
                # Mémorisation de l'expérience
                learner.storeExperience(agent, state, action, reward, nextState, done)
                
                # Communication entre agents
                agent.shareFeedback(reward)
            
            # Apprentissage à partir des expériences
            learner.learn()
            
            # Vérifier si tous les agents ont terminé
            episodeComplete = all(agent.hasCompletedObjective() for agent in agents)
            
            # Affichage si le renderer est actif
            if renderer is not None:
                continueSimulation = renderer.render(world, agents, learner)
                
                if not continueSimulation:
                    logger.info("Simulation interrompue par l'utilisateur")
                    return
            
            stepCount += 1
        
        # Fin de l'épisode
        episodeDuration = time.time() - startTime
        logger.info(f"Fin de l'épisode {episode}: {stepCount} étapes, récompense totale: {totalReward:.2f}, durée: {episodeDuration:.2f}s")
        
        # Affiche les statistiques de performance
        profiler.printStats()
        
        # Sauvegarde périodique du modèle
        if episode % 10 == 0:
            learner.saveModel(f"models/model_episode_{episode}.pt")
```

Ce flux implémente le cycle standard de l'apprentissage par renforcement:
1. Réinitialisation de l'environnement et des agents pour un nouvel épisode
2. Boucle principale:
   - Mise à jour de l'environnement
   - Pour chaque agent: observation → action → exécution → récompense → mémorisation
   - Apprentissage à partir des expériences collectées
   - Vérification de la complétion de l'épisode
   - Rendu visuel si activé
3. Finalisation de l'épisode et sauvegarde périodique du modèle

## Mécanismes de communication

Les agents collectifs peuvent communiquer entre eux via plusieurs mécanismes:

### Partage de connaissances

Les agents partagent des informations sur l'environnement quand ils sont à portée de communication:

```python
def shareKnowledge(self, targetAgent: 'CollectiveAgent') -> bool:
    # Vérifier si l'agent est à portée
    distance = self.getDistanceTo(targetAgent.position)
    if distance > self.communicationRange:
        return False
    
    # Déterminer quelles connaissances partager
    sharedKnowledge = self.memory.getRecentEntries(10)
    
    # Filtrer les connaissances pertinentes
    relevantKnowledge = [
        entry for entry in sharedKnowledge 
        if entry["certainty"] > 0.7 and 
           entry["timestamp"] > targetAgent.memory.getLastUpdateTime()
    ]
    
    # Partager les connaissances
    for entry in relevantKnowledge:
        # Ajuster la certitude en fonction de la compétence de communication
        adjustedEntry = entry.copy()
        adjustedEntry["certainty"] *= self.communicationSkill
        adjustedEntry["source"] = self.agentId
        targetAgent.memory.addEntry(adjustedEntry)
    
    # ... [mise à jour des métriques omise] ...
    
    return True
```

### Coordination explicite

Les agents peuvent se coordonner pour des tâches spécifiques:

```python
def requestCoordination(self, targetAgent: 'CollectiveAgent', coordinationType: str) -> bool:
    # ... [vérification de portée omise] ...
    
    # Probabilité d'acceptation basée sur le niveau de coopération
    acceptanceProbability = targetAgent.cooperationLevel
    
    # Facteurs influençant l'acceptation
    if coordinationType == "explore" and targetAgent.assignedRole == "explorer":
        acceptanceProbability *= 1.5
    
    # ... [décision d'acceptation omise] ...
    
    # Coordination acceptée
    if coordinationType == "explore":
        # Coordonner l'exploration (éviter la redondance)
        if self.assignedRegion is not None and targetAgent.assignedRegion is None:
            # Diviser la région
            newRegion1, newRegion2 = self._splitRegion(self.assignedRegion)
            self.assignedRegion = newRegion1
            targetAgent.assignedRegion = newRegion2
    
    # ... [autres types de coordination omis] ...
```

### Partage de feedback

Les agents partagent leurs récompenses et apprentissages:

```python
def shareFeedback(self, reward: float) -> None:
    # Créer une entrée dans la mémoire pour cette récompense
    feedbackEntry = {
        "type": "feedback",
        "position": self.position,
        "reward": reward,
        "timestamp": self.world.currentTime,
        "certainty": 1.0,
        "source": self.agentId
    }
    
    # Ajouter à sa propre mémoire
    self.memory.addEntry(feedbackEntry)
```

### Attribution de rôles dynamiques

Les agents s'attribuent des rôles en fonction de leurs performances et des besoins du groupe:

```python
def _updateRole(self) -> None:
    # Évaluer les performances dans différents rôles
    explorationPerformance = self.explorationSkill * (1.0 + self.memory.getCoverageRatio(self.world.width, self.world.height))
    collectionPerformance = self.collectionSkill * (1.0 + self.resourcesCollected / max(1, self.objectiveTarget))
    communicationPerformance = self.communicationSkill * (1.0 + self.collaborationScore / 50.0)
    
    # Identifier la compétence dominante
    performances = {
        "explorer": explorationPerformance,
        "collector": collectionPerformance,
        "scout": communicationPerformance
    }
    
    bestRole = max(performances, key=performances.get)
    
    # Ne pas changer de rôle trop fréquemment (stabilité)
    if bestRole != self.assignedRole and np.random.random() < 0.1:
        self.assignedRole = bestRole
```

## Optimisation des performances

NaviLearn intègre plusieurs mécanismes d'optimisation des performances:

### Profiler

La classe `Profiler` (`utils/profiler.py`) permet de mesurer le temps d'exécution des différentes parties du code:
- Mesure du temps par section
- Statistiques d'utilisation du temps
- Identification des goulots d'étranglement

```python
def startSection(self, sectionName: str) -> None:
    if not self.enabled:
        return
        
    # Enregistrer le temps de début
    self.sectionTimers[sectionName] = time.time()

def endSection(self, sectionName: str) -> None:
    if not self.enabled:
        return
        
    # Vérifier si la section a été démarrée
    if sectionName not in self.sectionTimers:
        return
        
    # Calculer le temps écoulé
    startTime = self.sectionTimers[sectionName]
    elapsedTime = time.time() - startTime
    
    # Mettre à jour les statistiques pour cette section
    if sectionName not in self.sectionStats:
        self.sectionStats[sectionName] = {
            "totalTime": 0.0,
            "callCount": 0,
            "minTime": float('inf'),
            "maxTime": 0.0
        }
        
    stats = self.sectionStats[sectionName]
    stats["totalTime"] += elapsedTime
    stats["callCount"] += 1
    stats["minTime"] = min(stats["minTime"], elapsedTime)
    stats["maxTime"] = max(stats["maxTime"], elapsedTime)
```

### Grille d'occupation spatiale

Pour optimiser la détection de collision et les recherches spatiales, une grille d'occupation est utilisée:

```python
def _getGridCells(self, position: Tuple[float, float], radius: float) -> List[Tuple[int, int]]:
    x, y = position
    cells = []
    
    # Calculer les limites de l'objet
    minX = max(0, int((x - radius) / self.gridCellSize))
    maxX = min(int(self.width / self.gridCellSize), int((x + radius) / self.gridCellSize) + 1)
    minY = max(0, int((y - radius) / self.gridCellSize))
    maxY = min(int(self.height / self.gridCellSize), int((y + radius) / self.gridCellSize) + 1)
    
    # Ajouter toutes les cellules dans ces limites
    for cellX in range(minX, maxX):
        for cellY in range(minY, maxY):
            cells.append((cellX, cellY))
    
    return cells
```

### Optimisation de l'apprentissage

Plusieurs techniques sont utilisées pour optimiser l'apprentissage:
- Batch processing pour exploiter le parallélisme GPU
- Mise à jour asynchrone des réseaux cibles pour la stabilité
- Fréquence d'apprentissage configurable pour équilibrer performance et qualité

## Bonnes pratiques de développement

NaviLearn suit plusieurs bonnes pratiques de développement:

### Typage statique

Le code utilise les annotations de type Python pour améliorer la lisibilité et permettre la vérification statique:

```python
def getAction(self, agent: CollectiveAgent, state: np.ndarray) -> Tuple[int, float]:
    # ...
```

### Séparation des responsabilités

L'architecture suit le principe de responsabilité unique:
- Les agents sont responsables de la prise de décision
- L'environnement gère la physique et les interactions
- Les algorithmes d'apprentissage gèrent l'optimisation des politiques
- La visualisation gère uniquement le rendu

### Documentation complète

Le code est entièrement documenté avec des docstrings au format Google:

```python
def add(
    self, 
    state: np.ndarray, 
    action: Any, 
    reward: float, 
    nextState: np.ndarray, 
    done: bool
) -> None:
    """
    Ajoute une expérience au tampon.
    
    Args:
        state (np.ndarray): État actuel
        action (Any): Action effectuée
        reward (float): Récompense reçue
        nextState (np.ndarray): État suivant
        done (bool): Indique si l'épisode est terminé
    """
    # ...
```

### Gestion des erreurs

Le code implémente une gestion robuste des erreurs avec des messages clairs:

```python
try:
    # Initialisation de l'environnement
    world = setupEnvironment(config)
    logger.info(f"Environnement créé: {world.width}x{world.height}, {world.obstacleCount} obstacles, {world.resourceCount} ressources")
    
    # ... [code omis] ...
    
except Exception as e:
    logger.error(f"Erreur lors de l'exécution: {e}")
    import traceback
    logger.error(traceback.format_exc())
    return 1
```

### Configuration externe

Toute la configuration est externalisée dans un fichier YAML, permettant des modifications sans changer le code:

```yaml
environment:
  width: 1000
  height: 1000
  obstacleCount: 30
  resourceCount: 15
  weatherEnabled: true
  terrainComplexity: 0.7

agents:
  count: 10
  sensorRange: 150
  communicationRange: 200
  maxSpeed: 5.0
  maxTurnRate: 0.4
  memorySize: 1000
```

La gestion de configuration est implémentée dans `utils/configManager.py`.

---

Ce document technique détaille le fonctionnement interne du système NaviLearn. Pour plus d'informations sur l'utilisation pratique, consultez le README.md.
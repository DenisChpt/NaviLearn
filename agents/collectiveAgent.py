#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module implémentant un agent collectif capable de collaborer avec d'autres agents.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Set

from agents.smartNavigator import SmartNavigator
from agents.knowledgeMemory import KnowledgeMemory
from environment.world import World


class CollectiveAgent(SmartNavigator):
	"""
	Agent collectif capable de communiquer et de collaborer avec d'autres agents.
	
	Cette classe étend SmartNavigator pour ajouter:
	- Des capacités de communication avec d'autres agents
	- Un partage de connaissances sur l'environnement
	- Des stratégies de coordination pour l'exploration et la collecte
	- Des mécanismes d'apprentissage social
	"""
	
	def __init__(
		self, 
		agentId: int, 
		world: World, 
		initialPosition: Optional[Tuple[float, float]] = None,
		sensorRange: float = 150.0,
		communicationRange: float = 200.0,
		maxSpeed: float = 5.0,
		maxTurnRate: float = 0.4,
		targetSelectionStrategy: str = "collaborative",
		memorySize: int = 1000,
		cooperationLevel: float = 0.7
	) -> None:
		"""
		Initialise un agent collectif.
		
		Args:
			agentId (int): Identifiant unique de l'agent
			world (World): Référence au monde dans lequel l'agent évolue
			initialPosition (Optional[Tuple[float, float]]): Position initiale (x, y)
			sensorRange (float): Portée des capteurs de l'agent
			communicationRange (float): Portée de communication avec d'autres agents
			maxSpeed (float): Vitesse maximale de déplacement
			maxTurnRate (float): Taux de rotation maximal en radians par pas de temps
			targetSelectionStrategy (str): Stratégie de sélection des cibles
			memorySize (int): Capacité de la mémoire de connaissances
			cooperationLevel (float): Niveau de coopération (0-1)
		"""
		super().__init__(
			agentId=agentId, 
			world=world, 
			initialPosition=initialPosition,
			sensorRange=sensorRange,
			maxSpeed=maxSpeed,
			maxTurnRate=maxTurnRate,
			targetSelectionStrategy=targetSelectionStrategy
		)
		
		# Paramètres de communication
		self.communicationRange = communicationRange
		self.knownAgents = set()  # Ensemble des IDs des agents connus
		self.cooperationLevel = cooperationLevel
		
		# Mémoire de connaissances
		self.memory = KnowledgeMemory(capacity=memorySize)
		
		# Métriques de collaboration
		self.informationShared = 0
		self.informationReceived = 0
		self.collaborativeActions = 0
		self.lastCommunicationTime = 0
		
		# État de coordination
		self.assignedRole = "explorer"  # explorer, collector, or scout
		self.assignedRegion = None
		self.collaborationScore = 0.0
		
		# Compétences de l'agent (spécialisation)
		self.explorationSkill = np.random.uniform(0.5, 1.0)
		self.collectionSkill = np.random.uniform(0.5, 1.0)
		self.communicationSkill = np.random.uniform(0.5, 1.0)
		
		# Paramètres de récompense pour la collaboration
		self.rewardForSharing = 0.5
		self.rewardForCollaboration = 2.0
	
	def reset(self) -> None:
		"""
		Réinitialise l'agent pour un nouvel épisode.
		"""
		super().reset()
		
		# Réinitialiser les métriques de collaboration
		self.informationShared = 0
		self.informationReceived = 0
		self.collaborativeActions = 0
		self.lastCommunicationTime = 0
		self.collaborationScore = 0.0
		
		# Réinitialiser la mémoire tout en conservant les connaissances sur le monde
		self.memory.clearTemporaryData()
		
		# Conserver les compétences et le rôle assigné
	
	def discoverPeers(self, allAgents: List['CollectiveAgent']) -> None:
		"""
		Découvre les autres agents présents dans l'environnement.
		
		Args:
			allAgents (List[CollectiveAgent]): Liste de tous les agents
		"""
		for agent in allAgents:
			if agent.agentId != self.agentId:
				self.knownAgents.add(agent.agentId)
	
	def getNearbyAgents(self, allAgents: List['CollectiveAgent']) -> List['CollectiveAgent']:
		"""
		Identifie les agents à portée de communication.
		
		Args:
			allAgents (List[CollectiveAgent]): Liste de tous les agents
			
		Returns:
			List[CollectiveAgent]: Agents à portée de communication
		"""
		nearbyAgents = []
		
		for agent in allAgents:
			if agent.agentId != self.agentId and agent.isActive:
				distance = self.getDistanceTo(agent.position)
				if distance <= self.communicationRange:
					nearbyAgents.append(agent)
		
		return nearbyAgents
	
	def shareKnowledge(self, targetAgent: 'CollectiveAgent') -> bool:
		"""
		Partage des connaissances avec un agent cible.
		
		Args:
			targetAgent (CollectiveAgent): Agent avec qui partager
			
		Returns:
			bool: True si des connaissances ont été partagées, False sinon
		"""
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
		
		if not relevantKnowledge:
			return False
		
		# Partager les connaissances
		for entry in relevantKnowledge:
			# Ajuster la certitude en fonction de la compétence de communication
			adjustedEntry = entry.copy()
			adjustedEntry["certainty"] *= self.communicationSkill
			adjustedEntry["source"] = self.agentId
			targetAgent.memory.addEntry(adjustedEntry)
		
		# Mettre à jour les métriques
		self.informationShared += len(relevantKnowledge)
		targetAgent.informationReceived += len(relevantKnowledge)
		self.collaborativeActions += 1
		self.lastCommunicationTime = self.world.currentTime
		
		return True
	
	def requestCoordination(self, targetAgent: 'CollectiveAgent', coordinationType: str) -> bool:
		"""
		Demande une coordination spécifique à un autre agent.
		
		Args:
			targetAgent (CollectiveAgent): Agent avec qui se coordonner
			coordinationType (str): Type de coordination ("explore", "collect", "avoid")
			
		Returns:
			bool: True si la coordination a été acceptée, False sinon
		"""
		# Vérifier si l'agent est à portée
		distance = self.getDistanceTo(targetAgent.position)
		if distance > self.communicationRange:
			return False
		
		# Probabilité d'acceptation basée sur le niveau de coopération de l'agent cible
		acceptanceProbability = targetAgent.cooperationLevel
		
		# Facteurs influençant l'acceptation
		if coordinationType == "explore" and targetAgent.assignedRole == "explorer":
			acceptanceProbability *= 1.5
		elif coordinationType == "collect" and targetAgent.assignedRole == "collector":
			acceptanceProbability *= 1.5
		
		# Décision d'acceptation
		if np.random.random() < acceptanceProbability:
			# Coordination acceptée
			if coordinationType == "explore":
				# Coordonner l'exploration (éviter la redondance)
				if self.assignedRegion is not None and targetAgent.assignedRegion is None:
					# Diviser la région
					newRegion1, newRegion2 = self._splitRegion(self.assignedRegion)
					self.assignedRegion = newRegion1
					targetAgent.assignedRegion = newRegion2
					
			elif coordinationType == "collect":
				# Coordonner la collecte (optimiser l'efficacité)
				if self.currentTarget is not None and targetAgent.currentTarget is not None:
					# Échanger les cibles si cela réduit la distance totale
					distanceSelf = self.getDistanceTo(self.currentTarget)
					distanceTarget = targetAgent.getDistanceTo(targetAgent.currentTarget)
					
					distanceSelfToTargetTarget = self.getDistanceTo(targetAgent.currentTarget)
					distanceTargetToSelfTarget = targetAgent.getDistanceTo(self.currentTarget)
					
					if (distanceSelfToTargetTarget + distanceTargetToSelfTarget) < (distanceSelf + distanceTarget):
						# Échanger les cibles
						tempTarget = self.currentTarget
						self.currentTarget = targetAgent.currentTarget
						targetAgent.currentTarget = tempTarget
			
			elif coordinationType == "avoid":
				# Éviter les zones déjà explorées par l'autre agent
				avoidRegion = targetAgent.memory.getExploredRegions(self.world.currentTime - 100)
				self.memory.addAvoidanceRegion(avoidRegion)
			
			# Mettre à jour les métriques
			self.collaborativeActions += 1
			targetAgent.collaborativeActions += 1
			return True
			
		return False
	
	def _splitRegion(self, region: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
		"""
		Divise une région en deux sous-régions pour l'exploration coopérative.
		
		Args:
			region (Tuple[float, float, float, float]): Région à diviser (x1, y1, x2, y2)
			
		Returns:
			Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]: 
				Deux sous-régions résultantes
		"""
		x1, y1, x2, y2 = region
		
		# Déterminer l'axe de division (côté le plus long)
		width = x2 - x1
		height = y2 - y1
		
		if width > height:
			# Diviser horizontalement
			midX = (x1 + x2) / 2
			region1 = (x1, y1, midX, y2)
			region2 = (midX, y1, x2, y2)
		else:
			# Diviser verticalement
			midY = (y1 + y2) / 2
			region1 = (x1, y1, x2, midY)
			region2 = (x1, midY, x2, y2)
		
		return region1, region2
	
	def shareFeedback(self, reward: float) -> None:
		"""
		Partage le feedback reçu avec d'autres agents proches.
		
		Args:
			reward (float): Récompense reçue à partager
		"""
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
		
		# Le partage effectif avec d'autres agents se fait lors d'interactions de communication
	
	def observeEnvironment(self) -> np.ndarray:
		"""
		Observe l'état actuel de l'environnement et des autres agents.
		
		Returns:
			np.ndarray: Vecteur d'état pour l'algorithme d'apprentissage
		"""
		# Obtenir l'observation de base du SmartNavigator
		baseObservation = super().observeEnvironment()
		
		# Ajouter les informations sur les autres agents à proximité
		nearbyAgentsCount = len(self.memory.getRecentAgentPositions(self.world.currentTime - 50))
		normalizedNearbyCount = min(1.0, nearbyAgentsCount / 10.0)  # Normaliser entre 0 et 1
		
		# Obtenir les informations de coordination et de collaboration
		roleEncoding = self._encodeRole()
		collaborationScore = min(1.0, self.collaborationScore / 100.0)
		
		# Récupérer des statistiques de la mémoire
		memoryFreshness = self.memory.getFreshness(self.world.currentTime)
		knowledgeCoverage = self.memory.getCoverageRatio(self.world.width, self.world.height)
		
		# Construire le vecteur d'état étendu
		collectiveFeatures = np.array([
			normalizedNearbyCount,         # Nombre d'agents proches
			roleEncoding[0], roleEncoding[1], roleEncoding[2],  # Encodage du rôle
			collaborationScore,            # Score de collaboration
			self.cooperationLevel,         # Niveau de coopération
			memoryFreshness,               # Fraîcheur des connaissances
			knowledgeCoverage,             # Couverture des connaissances
			self.explorationSkill,         # Compétence d'exploration
			self.collectionSkill,          # Compétence de collecte
			self.communicationSkill        # Compétence de communication
		])
		
		# Combiner les observations
		return np.concatenate([baseObservation, collectiveFeatures])
	
	def _encodeRole(self) -> np.ndarray:
		"""
		Encode le rôle actuel de l'agent en vecteur one-hot.
		
		Returns:
			np.ndarray: Encodage one-hot du rôle
		"""
		roleVector = np.zeros(3)
		
		if self.assignedRole == "explorer":
			roleVector[0] = 1.0
		elif self.assignedRole == "collector":
			roleVector[1] = 1.0
		elif self.assignedRole == "scout":
			roleVector[2] = 1.0
			
		return roleVector
	
	def update(self) -> None:
		"""
		Met à jour l'état de l'agent, y compris les interactions sociales.
		"""
		if not self.isActive:
			return
			
		# Mise à jour de base du SmartNavigator
		super().update()
		
		# Ajouter sa position actuelle à la mémoire
		self.memory.addPositionEntry(self.position, self.world.currentTime)
		
		# Ajouter les observations à la mémoire
		observations = self.scan()
		for observation in observations:
			if observation["type"] in ["resource", "obstacle"]:
				memoryEntry = {
					"type": observation["type"],
					"position": observation["object"].position if "object" in observation else None,
					"value": observation.get("value", 0.0),
					"timestamp": self.world.currentTime,
					"certainty": 1.0 - (observation["distance"] / self.sensorRange) * 0.5,
					"source": self.agentId
				}
				self.memory.addEntry(memoryEntry)
		
		# Mettre à jour le score de collaboration
		self.collaborationScore = (
			self.informationShared * 0.3 +
			self.informationReceived * 0.2 +
			self.collaborativeActions * 2.0
		)
		
		# Ajuster le rôle en fonction des performances
		self._updateRole()
	
	def _updateRole(self) -> None:
		"""
		Met à jour dynamiquement le rôle de l'agent en fonction des performances.
		"""
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
	
	def selectTarget(self) -> bool:
		"""
		Sélectionne une cible pour l'agent selon une stratégie collaborative.
		
		Returns:
			bool: True si une cible a été trouvée, False sinon
		"""
		# Stratégie spécifique au rôle
		if self.assignedRole == "explorer":
			return self._selectExplorationTarget()
		elif self.assignedRole == "collector":
			return self._selectCollectionTarget()
		elif self.assignedRole == "scout":
			return self._selectScoutingTarget()
		
		# Stratégie par défaut
		return super().selectTarget()
	
	def _selectExplorationTarget(self) -> bool:
		"""
		Sélectionne une cible d'exploration pour découvrir de nouvelles zones.
		
		Returns:
			bool: True si une cible a été trouvée, False sinon
		"""
		# Identifier les régions les moins explorées
		exploredRegions = self.memory.getExploredRegions(0)
		
		# Diviser le monde en grille
		gridSize = 4
		cellWidth = self.world.width / gridSize
		cellHeight = self.world.height / gridSize
		
		# Calculer le niveau d'exploration pour chaque cellule
		explorationLevels = np.zeros((gridSize, gridSize))
		
		for i in range(gridSize):
			for j in range(gridSize):
				cellX = i * cellWidth
				cellY = j * cellHeight
				cellCenter = (cellX + cellWidth/2, cellY + cellHeight/2)
				
				# Vérifier combien de points explorés sont dans cette cellule
				cellExplorationCount = 0
				for region in exploredRegions:
					if (cellX <= region[0] <= cellX + cellWidth and
						cellY <= region[1] <= cellY + cellHeight):
						cellExplorationCount += 1
				
				# Normaliser
				explorationLevels[i, j] = cellExplorationCount
		
		# Trouver la cellule la moins explorée qui n'est pas attribuée à un autre agent
		minExplorationVal = float('inf')
		targetCell = None
		
		for i in range(gridSize):
			for j in range(gridSize):
				if explorationLevels[i, j] < minExplorationVal:
					cellX = i * cellWidth
					cellY = j * cellHeight
					cellCenter = (cellX + cellWidth/2, cellY + cellHeight/2)
					
					# Vérifier si une cellule est déjà attribuée à un autre agent
					isAssigned = False
					for entry in self.memory.getEntries():
						if (entry.get("type") == "assigned_region" and
							entry.get("timestamp") > self.world.currentTime - 200):
							
							assignedX, assignedY = entry.get("position", (0, 0))
							if (cellX <= assignedX <= cellX + cellWidth and
								cellY <= assignedY <= cellY + cellHeight):
								isAssigned = True
								break
					
					if not isAssigned:
						minExplorationVal = explorationLevels[i, j]
						targetCell = cellCenter
		
		if targetCell is not None:
			self.currentTarget = targetCell
			self.targetType = "exploration"
			self.targetValue = 5.0  # Valeur arbitraire pour l'exploration
			
			# Marquer cette région comme attribuée
			self.memory.addEntry({
				"type": "assigned_region",
				"position": targetCell,
				"timestamp": self.world.currentTime,
				"certainty": 1.0,
				"source": self.agentId
			})
			
			return True
		
		# Si aucune cellule n'est appropriée, utiliser la stratégie par défaut
		return super().selectTarget()
	
	def _selectCollectionTarget(self) -> bool:
		"""
		Sélectionne une cible pour la collecte efficace de ressources.
		
		Returns:
			bool: True si une cible a été trouvée, False sinon
		"""
		# Rechercher des ressources connues dans la mémoire
		knownResources = []
		
		for entry in self.memory.getEntries():
			if (entry["type"] == "resource" and 
				entry["timestamp"] > self.world.currentTime - 500 and
				entry["certainty"] > 0.6):
				
				# Vérifier si cette ressource est probablement encore disponible
				isCollected = False
				for otherEntry in self.memory.getEntries():
					if (otherEntry["type"] == "resource_collected" and
						self._isSamePosition(otherEntry.get("position"), entry.get("position"), 20.0)):
						isCollected = True
						break
				
				if not isCollected:
					knownResources.append({
						"position": entry["position"],
						"value": entry.get("value", 1.0),
						"certainty": entry["certainty"],
						"timestamp": entry["timestamp"]
					})
		
		if knownResources:
			# Sélectionner la ressource la plus efficace (valeur/distance pondérée par certitude)
			bestResource = max(knownResources, 
							  key=lambda r: (r["value"] * r["certainty"]) / 
										 max(1.0, self.getDistanceTo(r["position"])))
			
			self.currentTarget = bestResource["position"]
			self.targetType = "resource"
			self.targetValue = bestResource["value"]
			return True
		
		# Si aucune ressource connue n'est disponible, utiliser la stratégie par défaut
		return super().selectTarget()
	
	def _selectScoutingTarget(self) -> bool:
		"""
		Sélectionne une cible pour la fonction de reconnaissance (scout).
		
		Returns:
			bool: True si une cible a été trouvée, False sinon
		"""
		# Les scouts se déplacent stratégiquement pour maximiser la communication
		
		# Obtenir la position moyenne des autres agents connus
		agentPositions = self.memory.getRecentAgentPositions(self.world.currentTime - 200)
		
		if not agentPositions:
			# Aucun agent connu, explorer normalement
			return self._selectExplorationTarget()
		
		# Calculer le centre du groupe
		centerX = sum(pos[0] for pos in agentPositions) / len(agentPositions)
		centerY = sum(pos[1] for pos in agentPositions) / len(agentPositions)
		
		# Calculer la position optimale pour la communication
		# (à une distance intermédiaire du centre du groupe)
		angle = np.random.uniform(0, 2 * math.pi)
		optimalDistance = self.communicationRange * 0.7
		
		targetX = centerX + math.cos(angle) * optimalDistance
		targetY = centerY + math.sin(angle) * optimalDistance
		
		# S'assurer que la cible reste dans les limites du monde
		targetX = max(0, min(self.world.width, targetX))
		targetY = max(0, min(self.world.height, targetY))
		
		self.currentTarget = (targetX, targetY)
		self.targetType = "scouting"
		self.targetValue = 3.0
		return True
	
	def _isSamePosition(self, pos1: Optional[Tuple[float, float]], 
					  pos2: Optional[Tuple[float, float]], 
					  threshold: float = 10.0) -> bool:
		"""
		Vérifie si deux positions sont approximativement les mêmes.
		
		Args:
			pos1 (Optional[Tuple[float, float]]): Première position
			pos2 (Optional[Tuple[float, float]]): Deuxième position
			threshold (float): Seuil de distance pour considérer les positions identiques
			
		Returns:
			bool: True si les positions sont considérées identiques, False sinon
		"""
		if pos1 is None or pos2 is None:
			return False
			
		dx = pos1[0] - pos2[0]
		dy = pos1[1] - pos2[1]
		distance = math.sqrt(dx*dx + dy*dy)
		
		return distance <= threshold
	
	def calculateReward(self) -> float:
		"""
		Calcule la récompense avec des composants collaboratifs.
		
		Returns:
			float: Récompense calculée
		"""
		# Récompense de base du SmartNavigator
		baseReward = super().calculateReward()
		
		# Récompenses spécifiques à la collaboration
		collaborationReward = 0.0
		
		# Récompense pour le partage d'informations
		collaborationReward += self.informationShared * self.rewardForSharing
		
		# Récompense pour les actions collaboratives
		collaborationReward += self.collaborativeActions * self.rewardForCollaboration
		
		# Récompenses spécifiques au rôle
		roleReward = 0.0
		
		if self.assignedRole == "explorer":
			# Récompenser l'exploration de nouvelles zones
			coverageRatio = self.memory.getCoverageRatio(self.world.width, self.world.height)
			roleReward += coverageRatio * 2.0 * self.explorationSkill
			
		elif self.assignedRole == "collector":
			# Récompenser l'efficacité de collecte
			if self.distanceTraveled > 0:
				efficiency = self.resourcesCollected / self.distanceTraveled * 100
				roleReward += efficiency * self.collectionSkill
				
		elif self.assignedRole == "scout":
			# Récompenser la maintenance d'un réseau de communication
			connectedAgents = len(self.memory.getRecentAgentPositions(self.world.currentTime - 100))
			roleReward += connectedAgents * 0.5 * self.communicationSkill
		
		# Combiner les récompenses
		totalReward = baseReward + collaborationReward + roleReward
		
		return totalReward
	
	def hasCompletedObjective(self) -> bool:
		"""
		Vérifie si l'agent a complété son objectif collectif.
		
		Returns:
			bool: True si l'objectif est complété, False sinon
		"""
		# L'objectif collectif peut être différent de l'objectif individuel
		if self.currentObjective == "collect_resources":
			# Calculer le nombre total de ressources collectées par tous les agents connus
			totalCollected = self.resourcesCollected
			
			# Ajouter les ressources collectées par d'autres agents (si connues)
			for entry in self.memory.getEntries():
				if entry["type"] == "resource_collected" and entry["source"] != self.agentId:
					totalCollected += 1
			
			return totalCollected >= self.objectiveTarget
		
		# Par défaut, utiliser la vérification d'objectif standard
		return super().hasCompletedObjective()
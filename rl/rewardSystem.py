#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module implémentant le système de récompense pour l'apprentissage par renforcement.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from agents.collectiveAgent import CollectiveAgent
from environment.world import World


class RewardSystem:
	"""
	Système de récompense configurable et adaptable.
	
	Cette classe définit différentes récompenses et pénalités pour guider
	l'apprentissage des agents dans l'environnement.
	"""
	
	def __init__(
		self,
		world: World,
		agents: List[CollectiveAgent],
		configParams: Optional[Dict[str, float]] = None
	) -> None:
		"""
		Initialise le système de récompense.
		
		Args:
			world (World): Référence à l'environnement
			agents (List[CollectiveAgent]): Liste des agents
			configParams (Optional[Dict[str, float]]): Paramètres de configuration
		"""
		self.world = world
		self.agents = agents
		
		# Paramètres de récompense par défaut
		self.defaultParams = {
			# Récompenses de base
			"resourceCollectionReward": 10.0,       # Récompense pour collecter une ressource
			"resourceValueMultiplier": 1.0,         # Multiplicateur basé sur la valeur de la ressource
			"completionBonus": 50.0,                # Bonus pour compléter l'objectif
			
			# Pénalités
			"collisionPenalty": -2.0,               # Pénalité pour collision
			"timeStepPenalty": -0.01,               # Pénalité par pas de temps (encourage la rapidité)
			"inactivePenalty": -0.5,                # Pénalité pour inactivité (pas de mouvement)
			
			# Récompenses pour l'exploration
			"explorationReward": 0.5,               # Récompense pour explorer de nouvelles zones
			"noveltyReward": 1.0,                   # Récompense pour découvrir un nouvel élément
			
			# Récompenses pour les comportements adaptatifs
			"efficiencyReward": 0.2,                # Récompense pour efficacité (distance/ressources)
			"obstacleAvoidanceReward": 0.1,         # Récompense pour éviter les obstacles
			"weatherAdaptationReward": 0.3,         # Récompense pour adaptation aux conditions météo
			
			# Récompenses collaboratives
			"informationSharingReward": 0.3,        # Récompense pour partage d'informations
			"coordinatedActionReward": 0.5,         # Récompense pour actions coordonnées
			"collectiveProgressReward": 0.2,        # Récompense basée sur le progrès collectif
			"rolePerformanceReward": 0.4,           # Récompense pour performance selon le rôle
			
			# Facteurs de décroissance des récompenses
			"explorationDecay": 0.99,               # Décroissance de la récompense d'exploration
			"noveltyDecay": 0.95,                   # Décroissance de la récompense de nouveauté
			
			# Seuils
			"inactivityThreshold": 0.1,             # Seuil de vitesse pour considérer inactivité
			"efficiencyThreshold": 0.05,            # Seuil minimal d'efficacité
			"explorationResolution": 20.0           # Résolution spatiale pour l'exploration
		}
		
		# Mettre à jour avec les paramètres spécifiés
		self.params = self.defaultParams.copy()
		if configParams:
			self.params.update(configParams)
		
		# Espaces déjà explorés par agent (pour le suivi de l'exploration)
		self.exploredSpaces = {agent.agentId: set() for agent in agents}
		
		# Historique des positions par agent (pour détecter l'inactivité)
		self.positionHistory = {agent.agentId: [] for agent in agents}
		self.maxHistoryLength = 10
		
		# Suivi des ressources collectées
		self.collectedResources = {agent.agentId: [] for agent in agents}
		
		# Statistiques pour normalisation
		self.rewardStats = {
			"min": float('inf'),
			"max": float('-inf'),
			"sum": 0.0,
			"count": 0,
			"recent": []
		}
		self.maxRecentRewards = 1000
	
	def calculateReward(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la récompense totale pour un agent.
		
		Args:
			agent (CollectiveAgent): Agent pour lequel calculer la récompense
			
		Returns:
			float: Récompense calculée
		"""
		# Récompense totale
		totalReward = 0.0
		
		# Récompense/pénalité de base par pas de temps
		totalReward += self.params["timeStepPenalty"]
		
		# Vérifier collecte de ressource
		resourceReward = self._calculateResourceReward(agent)
		totalReward += resourceReward
		
		# Pénalité pour collision
		collisionPenalty = self._calculateCollisionPenalty(agent)
		totalReward += collisionPenalty
		
		# Récompense pour exploration
		explorationReward = self._calculateExplorationReward(agent)
		totalReward += explorationReward
		
		# Pénalité pour inactivité
		inactivityPenalty = self._calculateInactivityPenalty(agent)
		totalReward += inactivityPenalty
		
		# Récompense pour efficacité
		efficiencyReward = self._calculateEfficiencyReward(agent)
		totalReward += efficiencyReward
		
		# Récompense pour évitement d'obstacles
		avoidanceReward = self._calculateObstacleAvoidanceReward(agent)
		totalReward += avoidanceReward
		
		# Récompense pour adaptation aux conditions météo
		weatherReward = self._calculateWeatherAdaptationReward(agent)
		totalReward += weatherReward
		
		# Récompenses collaboratives
		if hasattr(agent, "collaborationScore"):
			collaborativeReward = self._calculateCollaborativeReward(agent)
			totalReward += collaborativeReward
		
		# Bonus de progression vers l'objectif
		progressReward = self._calculateProgressReward(agent)
		totalReward += progressReward
		
		# Bonus pour complétion de l'objectif
		if agent.objectiveComplete:
			totalReward += self.params["completionBonus"]
		
		# Mettre à jour les statistiques
		self._updateRewardStats(totalReward)
		
		# Mettre à jour l'historique des positions
		self._updatePositionHistory(agent)
		
		return totalReward
	
	def _calculateResourceReward(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la récompense pour la collecte de ressources.
		
		Args:
			agent (CollectiveAgent): Agent concerné
			
		Returns:
			float: Récompense calculée
		"""
		resourceReward = 0.0
		
		# Vérifier s'il y a eu une collecte de ressource
		# En utilisant le delta du nombre de ressources collectées
		prevResourceCount = getattr(agent, "_prevResourceCount", 0)
		currentResourceCount = agent.resourcesCollected
		
		if currentResourceCount > prevResourceCount:
			# Une ressource a été collectée
			resourcesCollected = currentResourceCount - prevResourceCount
			baseReward = self.params["resourceCollectionReward"] * resourcesCollected
			
			# Multiplicateur basé sur la valeur si disponible
			lastValue = getattr(agent, "lastResourceValue", 1.0)
			valueMultiplier = lastValue * self.params["resourceValueMultiplier"]
			
			resourceReward = baseReward * max(1.0, valueMultiplier)
			
			# Stocker pour les statistiques
			self.collectedResources[agent.agentId].append({
				"time": self.world.currentTime,
				"value": lastValue,
				"reward": resourceReward
			})
		
		# Mettre à jour le compteur pour la prochaine fois
		agent._prevResourceCount = currentResourceCount
		
		return resourceReward
	
	def _calculateCollisionPenalty(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la pénalité pour les collisions.
		
		Args:
			agent (CollectiveAgent): Agent concerné
			
		Returns:
			float: Pénalité calculée (négative)
		"""
		penalty = 0.0
		
		# Vérifier s'il y a eu une collision
		prevCollisionCount = getattr(agent, "_prevCollisionCount", 0)
		currentCollisionCount = agent.collisionCount
		
		if currentCollisionCount > prevCollisionCount:
			# Une collision s'est produite
			collisions = currentCollisionCount - prevCollisionCount
			penalty = self.params["collisionPenalty"] * collisions
		
		# Mettre à jour le compteur pour la prochaine fois
		agent._prevCollisionCount = currentCollisionCount
		
		return penalty
	
	def _calculateExplorationReward(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la récompense pour l'exploration.
		
		Args:
			agent (CollectiveAgent): Agent concerné
			
		Returns:
			float: Récompense calculée
		"""
		explorationReward = 0.0
		
		# Discrétiser la position actuelle
		resolution = self.params["explorationResolution"]
		gridX = int(agent.position[0] / resolution)
		gridY = int(agent.position[1] / resolution)
		gridPos = (gridX, gridY)
		
		# Vérifier si cette cellule a déjà été explorée
		if gridPos not in self.exploredSpaces[agent.agentId]:
			# Nouvelle zone explorée
			self.exploredSpaces[agent.agentId].add(gridPos)
			
			# Récompense qui diminue avec le nombre de cellules explorées
			explorationProgress = len(self.exploredSpaces[agent.agentId]) / 1000.0  # Normaliser
			decayFactor = self.params["explorationDecay"] ** explorationProgress
			explorationReward = self.params["explorationReward"] * decayFactor
			
			# Bonus si c'est une découverte collective (premier agent à explorer)
			isFirstExplorer = True
			for otherAgentId, otherExplored in self.exploredSpaces.items():
				if otherAgentId != agent.agentId and gridPos in otherExplored:
					isFirstExplorer = False
					break
			
			if isFirstExplorer:
				explorationReward *= 2.0  # Bonus pour la première découverte
		
		return explorationReward
	
	def _calculateInactivityPenalty(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la pénalité pour l'inactivité.
		
		Args:
			agent (CollectiveAgent): Agent concerné
			
		Returns:
			float: Pénalité calculée (négative)
		"""
		penalty = 0.0
		
		# Vérifier l'historique des positions
		history = self.positionHistory[agent.agentId]
		
		if len(history) >= 2:
			# Calculer la distance parcourue récemment
			recentPositions = history[-2:]
			dx = recentPositions[-1][0] - recentPositions[-2][0]
			dy = recentPositions[-1][1] - recentPositions[-2][1]
			distance = math.sqrt(dx*dx + dy*dy)
			
			# Pénalité si la distance est inférieure au seuil
			if distance < self.params["inactivityThreshold"]:
				penalty = self.params["inactivePenalty"]
		
		return penalty
	
	def _calculateEfficiencyReward(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la récompense pour l'efficacité.
		
		Args:
			agent (CollectiveAgent): Agent concerné
			
		Returns:
			float: Récompense calculée
		"""
		reward = 0.0
		
		# Calculer l'efficacité (ressources collectées / distance parcourue)
		if agent.distanceTraveled > 0:
			efficiency = agent.resourcesCollected / agent.distanceTraveled * 100.0
			
			# Récompense si l'efficacité dépasse un seuil
			if efficiency > self.params["efficiencyThreshold"]:
				reward = efficiency * self.params["efficiencyReward"]
		
		return reward
	
	def _calculateObstacleAvoidanceReward(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la récompense pour l'évitement d'obstacles.
		
		Args:
			agent (CollectiveAgent): Agent concerné
			
		Returns:
			float: Récompense calculée
		"""
		reward = 0.0
		
		# Analyser les capteurs d'obstacle
		if hasattr(agent, "obstacleDetectors"):
			# Détecter les obstacles proches
			nearObstacles = [d for d in agent.obstacleDetectors if d < agent.sensorRange * 0.3]
			
			# Récompense pour évitement réussi si en mouvement et obstacles proches
			if nearObstacles and agent.velocity > 0:
				reward = len(nearObstacles) * self.params["obstacleAvoidanceReward"]
		
		return reward
	
	def _calculateWeatherAdaptationReward(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la récompense pour l'adaptation aux conditions météorologiques.
		
		Args:
			agent (CollectiveAgent): Agent concerné
			
		Returns:
			float: Récompense calculée
		"""
		reward = 0.0
		
		# Obtenir les conditions météorologiques actuelles
		weatherConditions = self.world.getWeatherConditions()
		
		# Vérifier les adaptations pertinentes
		if weatherConditions:
			# Adaptation à la visibilité réduite
			visibilityModifier = weatherConditions.get("visibilityModifier", 1.0)
			if visibilityModifier < 0.8 and agent.velocity > 0:
				# Réduire la vitesse dans le brouillard/pluie est adaptatif
				speedRatio = agent.velocity / agent.maxSpeed
				if speedRatio < 0.7:  # Vitesse réduite adaptative
					reward += (0.7 - speedRatio) * self.params["weatherAdaptationReward"]
			
			# Adaptation aux vents forts
			windStrength = weatherConditions.get("windStrength", 0.0)
			if windStrength > 0.7:
				# Se déplacer perpendiculairement au vent est plus efficace
				# (simplification - la direction du vent n'est pas modélisée explicitement)
				reward += 0.5 * self.params["weatherAdaptationReward"]
		
		return reward
	
	def _calculateCollaborativeReward(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la récompense pour les comportements collaboratifs.
		
		Args:
			agent (CollectiveAgent): Agent concerné
			
		Returns:
			float: Récompense calculée
		"""
		reward = 0.0
		
		# Récompense pour partage d'informations
		infoShared = getattr(agent, "informationShared", 0)
		prevInfoShared = getattr(agent, "_prevInfoShared", 0)
		
		if infoShared > prevInfoShared:
			reward += (infoShared - prevInfoShared) * self.params["informationSharingReward"]
			agent._prevInfoShared = infoShared
		
		# Récompense pour actions coordonnées
		coordActions = getattr(agent, "collaborativeActions", 0)
		prevCoordActions = getattr(agent, "_prevCoordActions", 0)
		
		if coordActions > prevCoordActions:
			reward += (coordActions - prevCoordActions) * self.params["coordinatedActionReward"]
			agent._prevCoordActions = coordActions
		
		# Récompense de performance selon le rôle
		assignedRole = getattr(agent, "assignedRole", None)
		if assignedRole:
			if assignedRole == "explorer":
				# Récompenser l'exploration efficace
				exploredCells = len(self.exploredSpaces[agent.agentId])
				exploredRatio = min(1.0, exploredCells / 500.0)  # Normaliser
				roleReward = exploredRatio * self.params["rolePerformanceReward"]
				reward += roleReward
				
			elif assignedRole == "collector":
				# Récompenser l'efficacité de collecte
				if agent.distanceTraveled > 0:
					efficiency = agent.resourcesCollected / agent.distanceTraveled * 100.0
					roleReward = efficiency * self.params["rolePerformanceReward"] * 0.01
					reward += roleReward
					
			elif assignedRole == "scout":
				# Récompenser la maintenance d'un réseau de communication
				if hasattr(agent, "memory") and hasattr(agent.memory, "getRecentAgentPositions"):
					recentContacts = len(agent.memory.getRecentAgentPositions(self.world.currentTime - 100))
					contactRatio = min(1.0, recentContacts / (len(self.agents) - 1))
					roleReward = contactRatio * self.params["rolePerformanceReward"]
					reward += roleReward
		
		# Récompense pour le progrès collectif
		collectiveResourceCount = sum(a.resourcesCollected for a in self.agents)
		prevCollectiveCount = getattr(self, "_prevCollectiveCount", 0)
		
		if collectiveResourceCount > prevCollectiveCount:
			reward += (collectiveResourceCount - prevCollectiveCount) * self.params["collectiveProgressReward"]
			self._prevCollectiveCount = collectiveResourceCount
		
		return reward
	
	def _calculateProgressReward(self, agent: CollectiveAgent) -> float:
		"""
		Calcule la récompense pour la progression vers l'objectif.
		
		Args:
			agent (CollectiveAgent): Agent concerné
			
		Returns:
			float: Récompense calculée
		"""
		reward = 0.0
		
		# Récompense basée sur le progrès relatif vers l'objectif
		if agent.currentObjective == "collect_resources" and agent.objectiveTarget > 0:
			progress = agent.resourcesCollected / agent.objectiveTarget
			prevProgress = getattr(agent, "_prevProgress", 0.0)
			
			# Récompense pour l'amélioration du progrès
			if progress > prevProgress:
				reward = (progress - prevProgress) * 5.0  # Facteur arbitraire
				agent._prevProgress = progress
		
		return reward
	
	def _updatePositionHistory(self, agent: CollectiveAgent) -> None:
		"""
		Met à jour l'historique des positions pour un agent.
		
		Args:
			agent (CollectiveAgent): Agent concerné
		"""
		# Ajouter la position actuelle
		history = self.positionHistory[agent.agentId]
		history.append(agent.position)
		
		# Limiter la taille de l'historique
		if len(history) > self.maxHistoryLength:
			history.pop(0)
	
	def _updateRewardStats(self, reward: float) -> None:
		"""
		Met à jour les statistiques des récompenses.
		
		Args:
			reward (float): Récompense à ajouter aux statistiques
		"""
		# Mettre à jour les statistiques globales
		self.rewardStats["min"] = min(self.rewardStats["min"], reward)
		self.rewardStats["max"] = max(self.rewardStats["max"], reward)
		self.rewardStats["sum"] += reward
		self.rewardStats["count"] += 1
		
		# Ajouter aux récompenses récentes
		self.rewardStats["recent"].append(reward)
		
		# Limiter la taille
		if len(self.rewardStats["recent"]) > self.maxRecentRewards:
			self.rewardStats["recent"].pop(0)
	
	def getRewardStats(self) -> Dict[str, Any]:
		"""
		Retourne les statistiques des récompenses.
		
		Returns:
			Dict[str, Any]: Statistiques calculées
		"""
		count = self.rewardStats["count"]
		
		if count == 0:
			return {
				"min": 0.0,
				"max": 0.0,
				"mean": 0.0,
				"recent_mean": 0.0,
				"recent_std": 0.0
			}
		
		recent = self.rewardStats["recent"]
		
		return {
			"min": self.rewardStats["min"],
			"max": self.rewardStats["max"],
			"mean": self.rewardStats["sum"] / count,
			"recent_mean": np.mean(recent) if recent else 0.0,
			"recent_std": np.std(recent) if recent else 0.0
		}
	
	def normalizeReward(self, reward: float) -> float:
		"""
		Normalise une récompense basée sur les statistiques.
		
		Args:
			reward (float): Récompense brute
			
		Returns:
			float: Récompense normalisée
		"""
		if self.rewardStats["count"] < 100:
			# Pas assez de données pour une normalisation fiable
			return reward
			
		recent = self.rewardStats["recent"]
		if not recent:
			return reward
			
		mean = np.mean(recent)
		std = np.std(recent)
		
		if std < 1e-8:
			return 0.0  # Éviter la division par zéro
			
		# Normaliser à moyenne 0 et écart-type 1
		normalizedReward = (reward - mean) / std
		
		# Limiter à un intervalle raisonnable
		normalizedReward = max(-5.0, min(5.0, normalizedReward))
		
		return normalizedReward
	
	def resetExplorationMap(self) -> None:
		"""
		Réinitialise les cartes d'exploration pour tous les agents.
		"""
		self.exploredSpaces = {agent.agentId: set() for agent in self.agents}
	
	def resetStats(self) -> None:
		"""
		Réinitialise les statistiques de récompense.
		"""
		self.rewardStats = {
			"min": float('inf'),
			"max": float('-inf'),
			"sum": 0.0,
			"count": 0,
			"recent": []
		}
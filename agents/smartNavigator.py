#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module implémentant un agent de navigation intelligent avec capacités d'apprentissage.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

from agents.baseAgent import BaseAgent
from environment.world import World
from environment.resourceNode import ResourceNode


class SmartNavigator(BaseAgent):
	"""
	Agent navigator intelligent capable d'apprendre par renforcement.
	
	Cette classe étend BaseAgent pour fournir:
	- Des capteurs plus sophistiqués
	- Des capacités de prise de décision améliorées
	- Une fonction de récompense adaptée à la navigation
	- Une représentation d'état adaptée aux algorithmes d'apprentissage
	"""
	
	ACTION_SPACE = {
		0: {"speed": 0.0, "turn": 0.0},                  # Ne rien faire
		1: {"speed": 1.0, "turn": 0.0},                  # Avancer
		2: {"speed": 0.8, "turn": -0.15},                # Avancer + Tourner légèrement à gauche
		3: {"speed": 0.8, "turn": 0.15},                 # Avancer + Tourner légèrement à droite
		4: {"speed": 0.6, "turn": -0.3},                 # Avancer + Tourner moyennement à gauche
		5: {"speed": 0.6, "turn": 0.3},                  # Avancer + Tourner moyennement à droite
		6: {"speed": 0.0, "turn": -0.5},                 # Tourner à gauche sur place
		7: {"speed": 0.0, "turn": 0.5},                  # Tourner à droite sur place
		8: {"speed": -0.5, "turn": 0.0}                  # Reculer
	}
	
	def __init__(
		self, 
		agentId: int, 
		world: World, 
		initialPosition: Optional[Tuple[float, float]] = None,
		sensorRange: float = 150.0,
		sensorCount: int = 8,
		sensorFOV: float = 2 * math.pi,  # Champ de vision en radians
		maxSpeed: float = 5.0,
		maxTurnRate: float = 0.4,
		targetSelectionStrategy: str = "nearest"
	) -> None:
		"""
		Initialise un agent navigateur intelligent.
		
		Args:
			agentId (int): Identifiant unique de l'agent
			world (World): Référence au monde dans lequel l'agent évolue
			initialPosition (Optional[Tuple[float, float]]): Position initiale (x, y)
			sensorRange (float): Portée des capteurs de l'agent
			sensorCount (int): Nombre de capteurs distribués uniformément dans le FOV
			sensorFOV (float): Champ de vision en radians (2π = 360°)
			maxSpeed (float): Vitesse maximale de déplacement
			maxTurnRate (float): Taux de rotation maximal en radians par pas de temps
			targetSelectionStrategy (str): Stratégie de sélection des cibles 
				("nearest", "valuable", "efficient")
		"""
		super().__init__(
			agentId=agentId, 
			world=world, 
			initialPosition=initialPosition,
			sensorRange=sensorRange,
			maxSpeed=maxSpeed,
			maxTurnRate=maxTurnRate
		)
		
		# Configuration des capteurs
		self.sensorCount = sensorCount
		self.sensorFOV = sensorFOV
		
		# Capteurs de distance pour différents types d'objets
		self.obstacleDetectors = np.ones(sensorCount) * sensorRange
		self.resourceDetectors = np.ones(sensorCount) * sensorRange
		self.edgeDetectors = np.ones(sensorCount) * sensorRange
		
		# Stratégie de sélection des cibles
		self.targetSelectionStrategy = targetSelectionStrategy
		
		# Cible actuelle
		self.currentTarget = None
		self.targetType = None
		self.targetValue = 0.0
		self.timeWithoutTarget = 0
		
		# Métriques de performance
		self.resourcesCollected = 0
		self.distanceTraveled = 0.0
		self.efficiencyScore = 0.0
		
		# Paramètres d'apprentissage et de récompense
		self.rewardForResource = 10.0
		self.penaltyForCollision = -5.0
		self.penaltyForInaction = -0.01
		self.rewardForProgress = 0.1
		
		# Objectifs
		self.currentObjective = "collect_resources"
		self.objectiveTarget = 5  # Nombre de ressources à collecter
	
	def reset(self) -> None:
		"""
		Réinitialise l'agent pour un nouvel épisode.
		"""
		super().reset()
		
		# Réinitialiser les capteurs
		self.obstacleDetectors = np.ones(self.sensorCount) * self.sensorRange
		self.resourceDetectors = np.ones(self.sensorCount) * self.sensorRange
		self.edgeDetectors = np.ones(self.sensorCount) * self.sensorRange
		
		# Réinitialiser les statistiques
		self.resourcesCollected = 0
		self.distanceTraveled = 0.0
		self.efficiencyScore = 0.0
		
		# Réinitialiser la cible
		self.currentTarget = None
		self.targetType = None
		self.targetValue = 0.0
		self.timeWithoutTarget = 0
		
		# Réinitialiser l'objectif
		self.objectiveComplete = False
	
	def _updateSensors(self) -> None:
		"""
		Met à jour l'état des capteurs en fonction de l'environnement.
		"""
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
			if obj["type"] == "obstacle":
				# Mettre à jour seulement si c'est plus proche que la valeur actuelle
				if obj["distance"] < self.obstacleDetectors[sensorIndex]:
					self.obstacleDetectors[sensorIndex] = obj["distance"]
			
			elif obj["type"] == "resource":
				if obj["distance"] < self.resourceDetectors[sensorIndex]:
					self.resourceDetectors[sensorIndex] = obj["distance"]
			
			elif obj["type"] == "edge":
				if obj["distance"] < self.edgeDetectors[sensorIndex]:
					self.edgeDetectors[sensorIndex] = obj["distance"]
	
	def selectTarget(self) -> bool:
		"""
		Sélectionne une cible pour l'agent selon la stratégie définie.
		
		Returns:
			bool: True si une cible a été trouvée, False sinon
		"""
		# Obtenir toutes les ressources non collectées
		availableResources = [r for r in self.world.resources if not r.isCollected]
		
		if not availableResources:
			self.currentTarget = None
			self.targetType = None
			self.targetValue = 0.0
			return False
		
		# Stratégies de sélection de cible
		if self.targetSelectionStrategy == "nearest":
			# Sélectionner la ressource la plus proche
			nearestResource = min(availableResources, 
								 key=lambda r: self.getDistanceTo(r.position))
			self.currentTarget = nearestResource.position
			self.targetType = "resource"
			self.targetValue = nearestResource.value
			
		elif self.targetSelectionStrategy == "valuable":
			# Sélectionner la ressource de plus haute valeur
			valueResource = max(availableResources, key=lambda r: r.value)
			self.currentTarget = valueResource.position
			self.targetType = "resource"
			self.targetValue = valueResource.value
			
		elif self.targetSelectionStrategy == "efficient":
			# Calculer un score d'efficacité (valeur / distance)
			bestResource = max(availableResources, 
							  key=lambda r: r.value / max(1.0, self.getDistanceTo(r.position)))
			self.currentTarget = bestResource.position
			self.targetType = "resource"
			self.targetValue = bestResource.value
		
		else:
			# Stratégie par défaut: la plus proche
			nearestResource = min(availableResources, 
								 key=lambda r: self.getDistanceTo(r.position))
			self.currentTarget = nearestResource.position
			self.targetType = "resource"
			self.targetValue = nearestResource.value
		
		self.timeWithoutTarget = 0
		return True
	
	def navigateToTarget(self) -> Tuple[float, float]:
		"""
		Calcule la vitesse et la rotation nécessaires pour atteindre la cible actuelle.
		
		Returns:
			Tuple[float, float]: (vitesse, taux de rotation)
		"""
		if self.currentTarget is None:
			return 0.0, 0.0
		
		# Calculer la distance et l'angle vers la cible
		distance = self.getDistanceTo(self.currentTarget)
		angle = self.getAngleTo(self.currentTarget)
		
		# Proportionnel à la distance et à l'angle
		speed = min(self.maxSpeed, distance / 10.0)
		
		# Tourner plus rapidement si l'angle est grand
		turnRate = angle * 0.5
		
		# Limiter la vitesse en cas de virage serré
		if abs(angle) > math.pi/4:
			speed *= 0.5
		
		# Éviter les obstacles (comportement réactif de base)
		minObstacleDistance = min(self.obstacleDetectors)
		frontObstacleDetector = self.sensorCount // 2
		
		# Réduire la vitesse près des obstacles
		if minObstacleDistance < self.sensorRange * 0.3:
			speed *= 0.7
		
		# Éviter les collisions imminentes
		if self.obstacleDetectors[frontObstacleDetector] < self.sensorRange * 0.2:
			# Trouver le côté avec le plus d'espace
			leftSpace = np.mean(self.obstacleDetectors[:frontObstacleDetector])
			rightSpace = np.mean(self.obstacleDetectors[frontObstacleDetector:])
			
			if leftSpace > rightSpace:
				turnRate = -self.maxTurnRate * 0.8  # Tourner à gauche
			else:
				turnRate = self.maxTurnRate * 0.8   # Tourner à droite
				
			speed *= 0.5  # Ralentir fortement
		
		# S'arrêter si on est très proche de la cible
		if distance < 5.0:
			speed = 0.0
			turnRate = 0.0
		
		return speed, turnRate
	
	def update(self) -> None:
		"""
		Met à jour l'état de l'agent à chaque pas de temps.
		"""
		if not self.isActive:
			return
			
		# Mettre à jour les capteurs
		self._updateSensors()
		
		# Vérifier si on a besoin de sélectionner une nouvelle cible
		if self.currentTarget is None or self.timeWithoutTarget > 100:
			self.selectTarget()
		else:
			self.timeWithoutTarget += 1
			
		# Si on a atteint la cible, en sélectionner une nouvelle
		if self.currentTarget is not None and self.getDistanceTo(self.currentTarget) < 5.0:
			self.currentTarget = None
			
		# Interaction avec l'environnement
		resourceCollected = self._checkResourceCollection()
		if resourceCollected:
			self.resourcesCollected += 1
			# Réinitialiser la cible car on vient de la collecter
			self.currentTarget = None
			
			# Vérifier si l'objectif est atteint
			if self.resourcesCollected >= self.objectiveTarget:
				self.objectiveComplete = True
				
		# Calculer la distance parcourue
		if len(self.positionHistory) >= 2:
			lastPosition = self.positionHistory[-2]
			dx = self.position[0] - lastPosition[0]
			dy = self.position[1] - lastPosition[1]
			stepDistance = math.sqrt(dx*dx + dy*dy)
			self.distanceTraveled += stepDistance
			
			# Mettre à jour le score d'efficacité
			if self.distanceTraveled > 0:
				self.efficiencyScore = self.resourcesCollected / self.distanceTraveled * 1000
		
		super().update()
	
	def handleResourceCollection(self, resource: ResourceNode) -> None:
		"""
		Gère la collecte d'une ressource.
		
		Args:
			resource (ResourceNode): La ressource collectée
		"""
		self.totalReward += resource.value * self.rewardForResource
	
	def observeEnvironment(self) -> np.ndarray:
		"""
		Observe l'état actuel de l'environnement et retourne un vecteur d'état.
		
		Returns:
			np.ndarray: Vecteur d'état pour l'algorithme d'apprentissage
		"""
		# Mettre à jour les capteurs
		self._updateSensors()
		
		# Normaliser les valeurs des capteurs (0-1)
		normalizedObstacles = self.obstacleDetectors / self.sensorRange
		normalizedResources = self.resourceDetectors / self.sensorRange
		normalizedEdges = self.edgeDetectors / self.sensorRange
		
		# Obtenir les informations sur la cible
		targetDistance = 1.0
		targetAngle = 0.0
		hasTarget = 0.0
		
		if self.currentTarget is not None:
			# Normaliser la distance et l'angle à la cible
			targetDistance = min(1.0, self.getDistanceTo(self.currentTarget) / self.sensorRange)
			targetAngle = self.getAngleTo(self.currentTarget) / math.pi  # Normaliser entre -1 et 1
			hasTarget = 1.0
		
		# Position relative dans le monde (normalisée)
		normalizedPosX = self.position[0] / self.world.width
		normalizedPosY = self.position[1] / self.world.height
		
		# Direction normalisée (sin et cos pour continuité)
		headingSin = math.sin(self.heading)
		headingCos = math.cos(self.heading)
		
		# Vitesse normalisée
		normalizedVelocity = self.velocity / self.maxSpeed
		
		# Construire le vecteur d'état
		stateVector = np.concatenate([
			normalizedObstacles,                    # Capteurs d'obstacles
			normalizedResources,                    # Capteurs de ressources
			normalizedEdges,                        # Capteurs de bords
			[normalizedPosX, normalizedPosY],       # Position normalisée
			[headingSin, headingCos],               # Orientation
			[normalizedVelocity],                   # Vitesse normalisée
			[targetDistance, targetAngle, hasTarget], # Informations sur la cible
			[self.resourcesCollected / self.objectiveTarget] # Progression vers l'objectif
		])
		
		return stateVector
	
	def executeAction(self, action: Union[int, Dict[str, float]]) -> None:
		"""
		Exécute une action choisie par l'algorithme d'apprentissage.
		
		Args:
			action (Union[int, Dict[str, float]]): Action à exécuter.
				Si int: indice dans l'espace d'action discrétisé
				Si dict: contient directement les valeurs de vitesse et rotation
		"""
		if isinstance(action, int):
			# Action discrète: récupérer les valeurs dans l'espace d'action
			if action in self.ACTION_SPACE:
				actionValues = self.ACTION_SPACE[action]
				speed = actionValues["speed"] * self.maxSpeed
				turnRate = actionValues["turn"] * self.maxTurnRate
			else:
				# Action invalide
				speed, turnRate = 0.0, 0.0
		else:
			# Action continue directe
			speed = action.get("speed", 0.0) * self.maxSpeed
			turnRate = action.get("turn", 0.0) * self.maxTurnRate
		
		# Appliquer l'action
		self.move(speed, turnRate)
	
	def calculateReward(self) -> float:
		"""
		Calcule la récompense actuelle de l'agent en fonction de son état.
		
		Returns:
			float: Récompense calculée
		"""
		reward = 0.0
		
		# Récompense de base: pénalité pour le temps qui passe
		reward += self.penaltyForInaction
		
		# Récompense pour l'approche de la cible
		if self.currentTarget is not None:
			# Récupérer la dernière position pour calculer le progrès
			if len(self.positionHistory) >= 2:
				lastPosition = self.positionHistory[-2]
				currentPosition = self.position
				
				# Calculer la distance actuelle et précédente vers la cible
				previousDistance = math.sqrt(
					(lastPosition[0] - self.currentTarget[0])**2 +
					(lastPosition[1] - self.currentTarget[1])**2
				)
				
				currentDistance = math.sqrt(
					(currentPosition[0] - self.currentTarget[0])**2 +
					(currentPosition[1] - self.currentTarget[1])**2
				)
				
				# Récompense proportionnelle au progrès fait
				progress = previousDistance - currentDistance
				reward += progress * self.rewardForProgress
		
		# Pénalité pour collision (si la vitesse est nulle après avoir essayé de bouger)
		if self.velocity == 0 and len(self.positionHistory) >= 2 and self.positionHistory[-1] == self.positionHistory[-2]:
			reward += self.penaltyForCollision
		
		return reward
	
	def hasCompletedObjective(self) -> bool:
		"""
		Vérifie si l'agent a complété son objectif actuel.
		
		Returns:
			bool: True si l'objectif est complété, False sinon
		"""
		if self.currentObjective == "collect_resources":
			return self.resourcesCollected >= self.objectiveTarget
		return False
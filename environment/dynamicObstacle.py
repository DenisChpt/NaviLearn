#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module définissant les obstacles présents dans l'environnement, statiques ou dynamiques.
"""

from typing import Tuple, Dict, Any, Optional, List
import random
import math
import numpy as np


class DynamicObstacle:
	"""
	Représente un obstacle dans l'environnement, potentiellement mobile.
	
	Les obstacles peuvent être statiques ou se déplacer selon différents
	modèles de mouvement, et peuvent avoir différentes propriétés affectant
	les agents.
	"""
	
	def __init__(
		self, 
		position: Tuple[float, float],
		radius: float = 20.0,
		isDynamic: bool = False,
		speed: float = 1.0,
		movementPattern: Optional[str] = None,
		traversable: bool = False,
		properties: Optional[Dict[str, Any]] = None
	) -> None:
		"""
		Initialise un nouvel obstacle.
		
		Args:
			position (Tuple[float, float]): Position (x, y) de l'obstacle
			radius (float): Rayon de l'obstacle
			isDynamic (bool): Si True, l'obstacle est mobile
			speed (float): Vitesse de déplacement si dynamique
			movementPattern (Optional[str]): Modèle de mouvement 
				("random", "patrol", "chase", "flowField")
			traversable (bool): Si True, l'obstacle peut être traversé mais avec pénalité
			properties (Optional[Dict[str, Any]]): Propriétés spéciales de l'obstacle
		"""
		self.position = position
		self.radius = radius
		self.isDynamic = isDynamic
		self.speed = speed
		self.movementPattern = movementPattern
		self.traversable = traversable
		self.properties = properties if properties is not None else {}
		
		# Paramètres de mouvement pour les obstacles dynamiques
		self.direction = random.uniform(0, 2 * math.pi)
		self.targetPosition = None
		self.patrolPoints = []
		self.lastUpdateTime = 0
		
		# Apparence visuelle
		self.color = self._getColorForType()
		self.opacity = 1.0 if not traversable else 0.6
		
		# Initialiser les paramètres spécifiques au modèle de mouvement
		self._initializeMovementPattern()
	
	def _getColorForType(self) -> Tuple[int, int, int]:
		"""
		Définit la couleur de l'obstacle en fonction de ses propriétés.
		
		Returns:
			Tuple[int, int, int]: Couleur RGB
		"""
		# Base color: gray
		baseColor = (100, 100, 100)
		
		# Modifier en fonction des propriétés
		if self.isDynamic:
			if self.movementPattern == "random":
				return (150, 75, 0)  # Brown
			elif self.movementPattern == "patrol":
				return (70, 130, 180)  # Steel blue
			elif self.movementPattern == "chase":
				return (178, 34, 34)  # Firebrick
			elif self.movementPattern == "flowField":
				return (147, 112, 219)  # Medium purple
			return (200, 100, 0)  # Orange (default dynamic)
		elif self.traversable:
			return (100, 150, 100)  # Light green
		
		return baseColor
	
	def _initializeMovementPattern(self) -> None:
		"""
		Initialise les paramètres spécifiques au modèle de mouvement.
		"""
		if not self.isDynamic:
			return
			
		if self.movementPattern == "patrol":
			# Créer un chemin de patrouille
			numPoints = random.randint(2, 5)
			center = self.position
			radius = random.uniform(50.0, 150.0)
			
			for i in range(numPoints):
				angle = (i / numPoints) * 2 * math.pi
				x = center[0] + math.cos(angle) * radius
				y = center[1] + math.sin(angle) * radius
				self.patrolPoints.append((x, y))
				
			self.targetPosition = self.patrolPoints[0]
			
		elif self.movementPattern == "random":
			# Initialiser avec une direction aléatoire
			self.direction = random.uniform(0, 2 * math.pi)
			self.directionChangeInterval = random.uniform(50, 200)
			self.timeUntilDirectionChange = self.directionChangeInterval
			
		elif self.movementPattern == "chase":
			# Les paramètres de poursuite seront définis pendant l'exécution
			self.detectionRange = 150.0
			self.targetAgent = None
			
		elif self.movementPattern == "flowField":
			# Configuration spécifique au champ de vecteurs
			self.fieldFrequency = random.uniform(0.005, 0.02)
			self.fieldAmplitude = random.uniform(0.5, 1.0)
			self.fieldOffset = (random.uniform(0, 1000), random.uniform(0, 1000))
	
	def update(self, currentTime: int) -> None:
		"""
		Met à jour la position de l'obstacle dynamique.
		
		Args:
			currentTime (int): Temps actuel de la simulation
		"""
		if not self.isDynamic:
			return
			
		# Calculer le temps écoulé depuis la dernière mise à jour
		timeDelta = 1 if self.lastUpdateTime == 0 else min(5, currentTime - self.lastUpdateTime)
		self.lastUpdateTime = currentTime
		
		# Appliquer le modèle de mouvement approprié
		if self.movementPattern == "random":
			self._updateRandomMovement(timeDelta)
		elif self.movementPattern == "patrol":
			self._updatePatrolMovement(timeDelta)
		elif self.movementPattern == "chase":
			self._updateChaseMovement(timeDelta)
		elif self.movementPattern == "flowField":
			self._updateFlowFieldMovement(currentTime, timeDelta)
	
	def _updateRandomMovement(self, timeDelta: int) -> None:
		"""
		Met à jour le mouvement aléatoire.
		
		Args:
			timeDelta (int): Temps écoulé depuis la dernière mise à jour
		"""
		# Décider si on change de direction
		self.timeUntilDirectionChange -= timeDelta
		
		if self.timeUntilDirectionChange <= 0:
			# Changer de direction
			self.direction += random.uniform(-math.pi/2, math.pi/2)
			self.direction = self.direction % (2 * math.pi)
			
			# Réinitialiser le compteur
			self.timeUntilDirectionChange = self.directionChangeInterval
		
		# Calculer le nouveau déplacement
		distance = self.speed * timeDelta
		dx = math.cos(self.direction) * distance
		dy = math.sin(self.direction) * distance
		
		# Mettre à jour la position
		self.position = (self.position[0] + dx, self.position[1] + dy)
	
	def _updatePatrolMovement(self, timeDelta: int) -> None:
		"""
		Met à jour le mouvement de patrouille.
		
		Args:
			timeDelta (int): Temps écoulé depuis la dernière mise à jour
		"""
		if not self.targetPosition or not self.patrolPoints:
			return
			
		# Calculer la direction vers la cible
		dx = self.targetPosition[0] - self.position[0]
		dy = self.targetPosition[1] - self.position[1]
		distance = math.sqrt(dx*dx + dy*dy)
		
		if distance < 5.0:
			# Atteint la cible, passer au point suivant
			currentIndex = self.patrolPoints.index(self.targetPosition)
			nextIndex = (currentIndex + 1) % len(self.patrolPoints)
			self.targetPosition = self.patrolPoints[nextIndex]
			
			# Recalculer la direction
			dx = self.targetPosition[0] - self.position[0]
			dy = self.targetPosition[1] - self.position[1]
			distance = math.sqrt(dx*dx + dy*dy)
		
		# Normaliser le vecteur de direction
		if distance > 0:
			dx /= distance
			dy /= distance
		
		# Calculer le déplacement
		moveDistance = self.speed * timeDelta
		newX = self.position[0] + dx * moveDistance
		newY = self.position[1] + dy * moveDistance
		
		# Mettre à jour la position
		self.position = (newX, newY)
	
	def _updateChaseMovement(self, timeDelta: int) -> None:
		"""
		Met à jour le mouvement de poursuite.
		
		Args:
			timeDelta (int): Temps écoulé depuis la dernière mise à jour
		"""
		# Note: Le code pour définir targetAgent est implémenté dans la classe World
		if not self.targetAgent:
			# Aucune cible, comportement aléatoire par défaut
			self._updateRandomMovement(timeDelta)
			return
			
		# Calculer la direction vers l'agent cible
		agentPosition = self.targetAgent.position
		dx = agentPosition[0] - self.position[0]
		dy = agentPosition[1] - self.position[1]
		distance = math.sqrt(dx*dx + dy*dy)
		
		# Vérifier si l'agent est toujours à portée
		if distance > self.detectionRange:
			self.targetAgent = None
			self._updateRandomMovement(timeDelta)
			return
		
		# Normaliser le vecteur de direction
		if distance > 0:
			dx /= distance
			dy /= distance
		
		# Calculer le déplacement
		moveDistance = self.speed * timeDelta
		newX = self.position[0] + dx * moveDistance
		newY = self.position[1] + dy * moveDistance
		
		# Mettre à jour la position
		self.position = (newX, newY)
	
	def _updateFlowFieldMovement(self, currentTime: int, timeDelta: int) -> None:
		"""
		Met à jour le mouvement basé sur un champ de vecteurs.
		
		Args:
			currentTime (int): Temps actuel
			timeDelta (int): Temps écoulé depuis la dernière mise à jour
		"""
		# Utiliser un champ de vecteurs de Perlin Noise pour le mouvement
		x, y = self.position
		
		# Calculer le vecteur de direction à partir du bruit de Perlin
		noiseX = np.sin(x * self.fieldFrequency + self.fieldOffset[0] + currentTime * 0.01) * self.fieldAmplitude
		noiseY = np.cos(y * self.fieldFrequency + self.fieldOffset[1] + currentTime * 0.01) * self.fieldAmplitude
		
		# Normaliser
		length = math.sqrt(noiseX*noiseX + noiseY*noiseY)
		if length > 0:
			noiseX /= length
			noiseY /= length
		
		# Calculer le déplacement
		moveDistance = self.speed * timeDelta
		newX = x + noiseX * moveDistance
		newY = y + noiseY * moveDistance
		
		# Mettre à jour la position
		self.position = (newX, newY)
	
	def reverseDirection(self) -> None:
		"""
		Inverse la direction de mouvement (utilisé quand l'obstacle heurte une limite).
		"""
		if self.isDynamic:
			self.direction = (self.direction + math.pi) % (2 * math.pi)
	
	def setTarget(self, targetAgent: Any) -> None:
		"""
		Définit un agent comme cible pour le mode poursuite.
		
		Args:
			targetAgent (Any): Agent à poursuivre
		"""
		if self.movementPattern == "chase":
			self.targetAgent = targetAgent
	
	def getInfo(self) -> Dict[str, Any]:
		"""
		Retourne les informations détaillées sur l'obstacle.
		
		Returns:
			Dict[str, Any]: Informations sur l'obstacle
		"""
		return {
			"position": self.position,
			"radius": self.radius,
			"isDynamic": self.isDynamic,
			"speed": self.speed,
			"movementPattern": self.movementPattern,
			"traversable": self.traversable,
			"properties": self.properties,
			"color": self.color,
			"opacity": self.opacity
		}
	
	def getDistanceTo(self, position: Tuple[float, float]) -> float:
		"""
		Calcule la distance entre l'obstacle et une position donnée.
		
		Args:
			position (Tuple[float, float]): Position à comparer
			
		Returns:
			float: Distance euclidienne
		"""
		dx = self.position[0] - position[0]
		dy = self.position[1] - position[1]
		return math.sqrt(dx*dx + dy*dy)
	
	def isCollidingWith(self, position: Tuple[float, float], radius: float = 0.0) -> bool:
		"""
		Vérifie si l'obstacle est en collision avec un cercle.
		
		Args:
			position (Tuple[float, float]): Centre du cercle
			radius (float): Rayon du cercle
			
		Returns:
			bool: True en cas de collision, False sinon
		"""
		distance = self.getDistanceTo(position)
		return distance < (self.radius + radius)
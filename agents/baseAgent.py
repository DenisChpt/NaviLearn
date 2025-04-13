#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module contenant la classe abstraite de base pour tous les agents.
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional

from environment.world import World


class BaseAgent(ABC):
	"""
	Classe abstraite définissant l'interface commune à tous les agents.
	
	Cette classe fournit les fonctionnalités de base pour les agents navigant
	dans l'environnement, y compris la perception, le mouvement et l'interaction
	avec les éléments du monde.
	"""
	
	def __init__(
		self, 
		agentId: int, 
		world: World, 
		initialPosition: Optional[Tuple[float, float]] = None,
		sensorRange: float = 100.0,
		maxSpeed: float = 3.0,
		maxTurnRate: float = 0.3
	) -> None:
		"""
		Initialise un nouvel agent.
		
		Args:
			agentId (int): Identifiant unique de l'agent
			world (World): Référence au monde dans lequel l'agent évolue
			initialPosition (Optional[Tuple[float, float]]): Position initiale (x, y)
				Si None, une position aléatoire valide est attribuée
			sensorRange (float): Portée des capteurs de l'agent (rayon de perception)
			maxSpeed (float): Vitesse maximale de déplacement
			maxTurnRate (float): Taux de rotation maximal en radians par pas de temps
		"""
		self.agentId = agentId
		self.world = world
		
		# Position et orientation
		if initialPosition is None:
			self.position = self.world.getRandomValidPosition()
		else:
			self.position = initialPosition
			
		# Direction initiale aléatoire (angle en radians)
		self.heading = np.random.uniform(0, 2 * math.pi)
		
		# Paramètres de mouvement
		self.velocity = 0.0
		self.maxSpeed = maxSpeed
		self.maxTurnRate = maxTurnRate
		
		# Paramètres des capteurs
		self.sensorRange = sensorRange
		
		# Historique des positions (pour tracer des chemins)
		self.positionHistory = []
		self.maxHistoryLength = 100  # Nombre de positions à conserver
		
		# Statistiques de performances
		self.collisionCount = 0
		self.objectivesCompleted = 0
		self.totalReward = 0.0
		self.isActive = True
		
		# État actuel d'achèvement de l'objectif
		self.objectiveComplete = False
		
		# Pour le rendu graphique
		self.color = self._generateColor()
		self.size = 10.0
	
	def _generateColor(self) -> Tuple[int, int, int]:
		"""
		Génère une couleur unique basée sur l'ID de l'agent.
		
		Returns:
			Tuple[int, int, int]: Couleur RGB (0-255)
		"""
		# Utiliser une fonction de hachage pour générer des couleurs distinctes
		hue = (self.agentId * 0.618033988749895) % 1.0
		
		# Conversion HSV vers RGB
		h = hue * 6.0
		sector = math.floor(h)
		f = h - sector
		p = 0.8
		q = 0.8 * (1.0 - f)
		t = 0.8 * f
		
		if sector == 0 or sector == 6:
			r, g, b = 0.8, t, p
		elif sector == 1:
			r, g, b = q, 0.8, p
		elif sector == 2:
			r, g, b = p, 0.8, t
		elif sector == 3:
			r, g, b = p, q, 0.8
		elif sector == 4:
			r, g, b = t, p, 0.8
		else:  # sector == 5
			r, g, b = 0.8, p, q
		
		return (int(r * 255), int(g * 255), int(b * 255))
	
	def reset(self) -> None:
		"""
		Réinitialise l'agent à un état initial pour un nouvel épisode.
		"""
		self.position = self.world.getRandomValidPosition()
		self.heading = np.random.uniform(0, 2 * math.pi)
		self.velocity = 0.0
		self.positionHistory = []
		self.objectiveComplete = False
		self.isActive = True
	
	def move(self, speed: float, turnRate: float) -> None:
		"""
		Déplace l'agent selon les paramètres spécifiés.
		
		Args:
			speed (float): Vitesse de déplacement, entre -maxSpeed et maxSpeed
			turnRate (float): Taux de rotation, entre -maxTurnRate et maxTurnRate
		"""
		# Limiter la vitesse et le taux de rotation aux valeurs maximales
		speed = np.clip(speed, -self.maxSpeed, self.maxSpeed)
		turnRate = np.clip(turnRate, -self.maxTurnRate, self.maxTurnRate)
		
		# Mettre à jour l'orientation
		self.heading += turnRate
		# Normaliser l'orientation entre 0 et 2π
		self.heading = self.heading % (2 * math.pi)
		
		# Calculer le nouveau vecteur de vitesse
		self.velocity = speed
		
		# Calculer la nouvelle position
		newX = self.position[0] + math.cos(self.heading) * self.velocity
		newY = self.position[1] + math.sin(self.heading) * self.velocity
		newPosition = (newX, newY)
		
		# Vérifier si la nouvelle position est valide (pas de collision)
		if self.world.isPositionValid(newPosition):
			self.position = newPosition
			# Ajouter la position à l'historique
			self.positionHistory.append(self.position)
			# Limiter la taille de l'historique
			if len(self.positionHistory) > self.maxHistoryLength:
				self.positionHistory.pop(0)
		else:
			# Collision détectée
			self.handleCollision()
	
	def handleCollision(self) -> None:
		"""
		Gère le comportement de l'agent en cas de collision.
		"""
		self.collisionCount += 1
		# Réduire la vitesse
		self.velocity = 0
		
		# Changer légèrement l'orientation pour éviter de rester bloqué
		self.heading += np.random.uniform(-math.pi/4, math.pi/4)
	
	def getDistanceTo(self, point: Tuple[float, float]) -> float:
		"""
		Calcule la distance euclidienne entre l'agent et un point donné.
		
		Args:
			point (Tuple[float, float]): Coordonnées du point cible
			
		Returns:
			float: Distance entre l'agent et le point
		"""
		dx = self.position[0] - point[0]
		dy = self.position[1] - point[1]
		return math.sqrt(dx*dx + dy*dy)
	
	def getAngleTo(self, point: Tuple[float, float]) -> float:
		"""
		Calcule l'angle entre l'orientation actuelle et un point cible.
		
		Args:
			point (Tuple[float, float]): Coordonnées du point cible
			
		Returns:
			float: Angle en radians
		"""
		dx = point[0] - self.position[0]
		dy = point[1] - self.position[1]
		targetAngle = math.atan2(dy, dx)
		
		# Calculer la différence d'angle
		angleDiff = targetAngle - self.heading
		
		# Normaliser entre -π et π
		while angleDiff > math.pi:
			angleDiff -= 2 * math.pi
		while angleDiff < -math.pi:
			angleDiff += 2 * math.pi
			
		return angleDiff
	
	def scan(self, range: Optional[float] = None) -> List[Dict[str, Any]]:
		"""
		Analyse l'environnement proche pour détecter les objets.
		
		Args:
			range (Optional[float]): Portée de l'analyse, utilise sensorRange si None
			
		Returns:
			List[Dict[str, Any]]: Liste des objets détectés avec leurs propriétés
		"""
		if range is None:
			range = self.sensorRange
			
		detected = []
		
		# Détecter les obstacles
		for obstacle in self.world.obstacles:
			distance = self.getDistanceTo(obstacle.position)
			if distance <= range:
				angle = self.getAngleTo(obstacle.position)
				detected.append({
					"type": "obstacle",
					"distance": distance,
					"angle": angle,
					"object": obstacle
				})
		
		# Détecter les ressources
		for resource in self.world.resources:
			if not resource.isCollected:
				distance = self.getDistanceTo(resource.position)
				if distance <= range:
					angle = self.getAngleTo(resource.position)
					detected.append({
						"type": "resource",
						"distance": distance,
						"angle": angle,
						"value": resource.value,
						"object": resource
					})
		
		# Ajouter la détection des limites du monde
		edgeDetections = self._detectWorldEdges(range)
		detected.extend(edgeDetections)
		
		return detected
	
	def _detectWorldEdges(self, range: float) -> List[Dict[str, Any]]:
		"""
		Détecte les bords du monde dans la portée des capteurs.
		
		Args:
			range (float): Portée de détection
			
		Returns:
			List[Dict[str, Any]]: Liste des bords détectés
		"""
		edges = []
		
		# Distance aux bords
		distanceToLeft = self.position[0]
		distanceToRight = self.world.width - self.position[0]
		distanceToTop = self.position[1]
		distanceToBottom = self.world.height - self.position[1]
		
		# Ajouter les bords qui sont dans la portée
		if distanceToLeft <= range:
			angle = self.getAngleTo((0, self.position[1]))
			edges.append({
				"type": "edge",
				"edge": "left",
				"distance": distanceToLeft,
				"angle": angle
			})
			
		if distanceToRight <= range:
			angle = self.getAngleTo((self.world.width, self.position[1]))
			edges.append({
				"type": "edge",
				"edge": "right",
				"distance": distanceToRight,
				"angle": angle
			})
			
		if distanceToTop <= range:
			angle = self.getAngleTo((self.position[0], 0))
			edges.append({
				"type": "edge",
				"edge": "top",
				"distance": distanceToTop,
				"angle": angle
			})
			
		if distanceToBottom <= range:
			angle = self.getAngleTo((self.position[0], self.world.height))
			edges.append({
				"type": "edge",
				"edge": "bottom",
				"distance": distanceToBottom,
				"angle": angle
			})
			
		return edges
	
	def update(self) -> None:
		"""
		Met à jour l'état de l'agent à chaque pas de temps.
		"""
		if not self.isActive:
			return
			
		# Interaction avec l'environnement
		self._checkResourceCollection()
	
	def _checkResourceCollection(self) -> bool:
		"""
		Vérifie si l'agent a collecté une ressource et la traite.
		
		Returns:
			bool: True si une ressource a été collectée, False sinon
		"""
		for resource in self.world.resources:
			if not resource.isCollected and self.getDistanceTo(resource.position) < resource.radius + self.size/2:
				resource.collect(self.agentId)
				self.handleResourceCollection(resource)
				return True
		return False
	
	def handleResourceCollection(self, resource: Any) -> None:
		"""
		Gère la collecte d'une ressource par l'agent.
		
		Args:
			resource (Any): La ressource collectée
		"""
		# À implémenter dans les classes dérivées
		pass
	
	@abstractmethod
	def observeEnvironment(self) -> np.ndarray:
		"""
		Observe l'état actuel de l'environnement et retourne un vecteur d'état.
		
		Returns:
			np.ndarray: Vecteur d'état pour l'algorithme d'apprentissage
		"""
		pass
	
	@abstractmethod
	def executeAction(self, action: Any) -> None:
		"""
		Exécute une action choisie par l'algorithme d'apprentissage.
		
		Args:
			action (Any): Action à exécuter (format dépendant de l'implémentation)
		"""
		pass
	
	@abstractmethod
	def calculateReward(self) -> float:
		"""
		Calcule la récompense actuelle de l'agent en fonction de son état.
		
		Returns:
			float: Récompense calculée
		"""
		pass
	
	@abstractmethod
	def hasCompletedObjective(self) -> bool:
		"""
		Vérifie si l'agent a complété son objectif actuel.
		
		Returns:
			bool: True si l'objectif est complété, False sinon
		"""
		pass
	
	def getState(self) -> Dict[str, Any]:
		"""
		Retourne l'état actuel de l'agent.
		
		Returns:
			Dict[str, Any]: État de l'agent
		"""
		return {
			"id": self.agentId,
			"position": self.position,
			"heading": self.heading,
			"velocity": self.velocity,
			"collisions": self.collisionCount,
			"objectivesCompleted": self.objectivesCompleted,
			"totalReward": self.totalReward,
			"active": self.isActive,
			"objectiveComplete": self.objectiveComplete
		}
	
	def getRelativePosition(self, point: Tuple[float, float]) -> Tuple[float, float]:
		"""
		Calcule la position relative d'un point par rapport à l'agent.
		
		Args:
			point (Tuple[float, float]): Coordonnées du point
			
		Returns:
			Tuple[float, float]: Position relative (distance, angle)
		"""
		distance = self.getDistanceTo(point)
		angle = self.getAngleTo(point)
		return (distance, angle)
		
	def __str__(self) -> str:
		"""
		Représentation textuelle de l'agent.
		
		Returns:
			str: Description de l'agent
		"""
		return f"Agent#{self.agentId} at ({self.position[0]:.1f}, {self.position[1]:.1f}), heading: {math.degrees(self.heading):.1f}°"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module définissant les ressources présentes dans l'environnement.
"""

from typing import Tuple, Dict, Any, Optional, List
import random
import math


class ResourceNode:
	"""
	Représente une ressource pouvant être collectée par les agents.
	
	Les ressources ont différentes caractéristiques comme leur type,
	leur valeur, et peuvent avoir des effets spéciaux lorsqu'elles
	sont collectées.
	"""
	
	def __init__(
		self, 
		position: Tuple[float, float],
		value: float = 1.0,
		type: str = "standard",
		radius: float = 10.0,
		respawnTime: Optional[int] = None,
		properties: Optional[Dict[str, Any]] = None
	) -> None:
		"""
		Initialise une nouvelle ressource.
		
		Args:
			position (Tuple[float, float]): Position (x, y) de la ressource
			value (float): Valeur de base de la ressource
			type (str): Type de ressource ("standard", "bonus", "rare", etc.)
			radius (float): Rayon de la ressource (pour la collision)
			respawnTime (Optional[int]): Temps de réapparition, None = pas de réapparition
			properties (Optional[Dict[str, Any]]): Propriétés spéciales de la ressource
		"""
		self.position = position
		self.value = value
		self.type = type
		self.radius = radius
		self.respawnTime = respawnTime
		self.properties = properties if properties is not None else {}
		
		# État de la ressource
		self.isCollected = False
		self.collectedBy = None
		self.collectionTime = None
		self.respawnCounter = 0
		
		# Apparence visuelle
		self.color = self._getColorForType()
		self.pulsating = type in ["rare", "bonus"]
		self.pulsePhase = random.uniform(0, 2 * math.pi)
		self.pulseSpeed = 0.05
	
	def _getColorForType(self) -> Tuple[int, int, int]:
		"""
		Définit la couleur de la ressource en fonction de son type.
		
		Returns:
			Tuple[int, int, int]: Couleur RGB
		"""
		colorMap = {
			"standard": (0, 255, 0),     # Vert
			"bonus": (255, 215, 0),      # Or
			"rare": (138, 43, 226),      # Violet
			"penalty": (255, 0, 0),      # Rouge
			"special": (0, 191, 255)     # Bleu clair
		}
		
		return colorMap.get(self.type, (0, 255, 0))
	
	def collect(self, agentId: int) -> float:
		"""
		Marque la ressource comme collectée et retourne sa valeur.
		
		Args:
			agentId (int): ID de l'agent qui collecte la ressource
			
		Returns:
			float: Valeur effective de la ressource
		"""
		if self.isCollected:
			return 0.0
			
		self.isCollected = True
		self.collectedBy = agentId
		self.collectionTime = 0  # Sera défini par le monde lors de la mise à jour
		
		# Appliquer des effets spéciaux basés sur le type
		effectiveValue = self.value
		
		if self.type == "bonus":
			effectiveValue *= 2.0
		elif self.type == "rare":
			effectiveValue *= 3.0
		elif self.type == "penalty":
			effectiveValue = -self.value  # Valeur négative
		
		# Appliquer des modificateurs spéciaux
		for key, value in self.properties.items():
			if key == "valueMultiplier":
				effectiveValue *= value
			elif key == "bonusValue":
				effectiveValue += value
		
		return effectiveValue
	
	def update(self, currentTime: int) -> None:
		"""
		Met à jour l'état de la ressource.
		
		Args:
			currentTime (int): Temps actuel de la simulation
		"""
		# Mettre à jour le temps de collection si nécessaire
		if self.isCollected and self.collectionTime is None:
			self.collectionTime = currentTime
		
		# Gérer la réapparition si activée
		if self.isCollected and self.respawnTime is not None:
			timeSinceCollection = currentTime - self.collectionTime
			
			if timeSinceCollection >= self.respawnTime:
				self.respawn()
		
		# Mettre à jour l'animation de pulsation
		if self.pulsating:
			self.pulsePhase += self.pulseSpeed
			if self.pulsePhase > 2 * math.pi:
				self.pulsePhase -= 2 * math.pi
	
	def respawn(self) -> None:
		"""
		Fait réapparaître la ressource si elle a été collectée.
		"""
		if self.isCollected:
			self.isCollected = False
			self.collectedBy = None
			self.collectionTime = None
			self.respawnCounter += 1
			
			# Éventuellement modifier les propriétés après réapparition
			if self.respawnCounter > 3:
				# Diminuer progressivement la valeur après plusieurs réapparitions
				self.value *= 0.8
	
	def getVisualRadius(self) -> float:
		"""
		Calcule le rayon visuel actuel, incluant les effets d'animation.
		
		Returns:
			float: Rayon visuel
		"""
		if not self.pulsating:
			return self.radius
			
		# Effet de pulsation
		pulseFactor = 0.3 * math.sin(self.pulsePhase) + 1.0
		return self.radius * pulseFactor
	
	def getInfo(self) -> Dict[str, Any]:
		"""
		Retourne les informations détaillées sur la ressource.
		
		Returns:
			Dict[str, Any]: Informations sur la ressource
		"""
		return {
			"position": self.position,
			"value": self.value,
			"type": self.type,
			"isCollected": self.isCollected,
			"collectedBy": self.collectedBy,
			"respawnTime": self.respawnTime,
			"respawnCounter": self.respawnCounter,
			"properties": self.properties,
			"color": self.color
		}
	
	def getDistanceTo(self, position: Tuple[float, float]) -> float:
		"""
		Calcule la distance entre la ressource et une position donnée.
		
		Args:
			position (Tuple[float, float]): Position à comparer
			
		Returns:
			float: Distance euclidienne
		"""
		dx = self.position[0] - position[0]
		dy = self.position[1] - position[1]
		return math.sqrt(dx*dx + dy*dy)
	
	def isAccessibleFrom(self, position: Tuple[float, float], sensorRange: float) -> bool:
		"""
		Vérifie si la ressource est accessible depuis une position donnée.
		
		Args:
			position (Tuple[float, float]): Position de départ
			sensorRange (float): Portée du capteur
			
		Returns:
			bool: True si la ressource est accessible, False sinon
		"""
		if self.isCollected:
			return False
			
		distance = self.getDistanceTo(position)
		return distance <= sensorRange
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module définissant l'environnement de simulation pour les agents.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
import time
import random

from environment.dynamicObstacle import DynamicObstacle
from environment.resourceNode import ResourceNode
from environment.weatherSystem import WeatherSystem
from environment.terrainGenerator import TerrainGenerator


class World:
	"""
	Environnement de simulation dans lequel les agents évoluent.
	
	Cette classe est responsable de:
	- Générer et maintenir l'environnement physique (obstacles, ressources)
	- Gérer les conditions dynamiques (météo, événements)
	- Fournir des méthodes d'interaction avec l'environnement
	- Gérer la progression du temps dans la simulation
	"""
	
	def __init__(
		self, 
		width: float = 1000.0, 
		height: float = 1000.0,
		obstacleCount: int = 30,
		resourceCount: int = 15,
		maxSteps: int = 10000,
		weatherEnabled: bool = True,
		terrainComplexity: float = 0.5
	) -> None:
		"""
		Initialise un nouvel environnement de simulation.
		
		Args:
			width (float): Largeur du monde
			height (float): Hauteur du monde
			obstacleCount (int): Nombre d'obstacles à générer
			resourceCount (int): Nombre de ressources à générer
			maxSteps (int): Nombre maximum d'étapes par épisode
			weatherEnabled (bool): Activer les effets météorologiques
			terrainComplexity (float): Complexité du terrain (0-1)
		"""
		self.width = width
		self.height = height
		self.obstacleCount = obstacleCount
		self.resourceCount = resourceCount
		self.maxSteps = maxSteps
		
		# Temps et état
		self.currentTime = 0
		self.currentStep = 0
		self.episodeComplete = False
		
		# Éléments de l'environnement
		self.obstacles = []
		self.resources = []
		self.events = []
		
		# Systèmes dynamiques
		self.weatherSystem = WeatherSystem(enabled=weatherEnabled)
		
		# Générateur de terrain
		self.terrainGenerator = TerrainGenerator(
			width=width,
			height=height,
			complexity=terrainComplexity
		)
		
		# Grille d'occupation pour les collisions efficaces
		self.gridCellSize = 50.0  # Taille des cellules de la grille
		self.occupancyGrid = {}
		
		# Générer l'environnement initial
		self._generateEnvironment()
	
	def reset(self) -> None:
		"""
		Réinitialise l'environnement pour un nouvel épisode.
		"""
		self.currentTime = 0
		self.currentStep = 0
		self.episodeComplete = False
		
		# Vider les listes d'éléments
		self.obstacles = []
		self.resources = []
		self.events = []
		self.occupancyGrid = {}
		
		# Réinitialiser les systèmes dynamiques
		self.weatherSystem.reset()
		
		# Régénérer l'environnement
		self._generateEnvironment()
	
	def _generateEnvironment(self) -> None:
		"""
		Génère les éléments de l'environnement (obstacles, ressources).
		"""
		# Générer le terrain de base
		terrainFeatures = self.terrainGenerator.generate()
		
		# Générer les obstacles
		self._generateObstacles(self.obstacleCount, terrainFeatures)
		
		# Générer les ressources
		self._generateResources(self.resourceCount, terrainFeatures)
		
		# Mettre à jour la grille d'occupation
		self._updateOccupancyGrid()
	
	def _generateObstacles(self, count: int, terrainFeatures: Dict[str, Any]) -> None:
		"""
		Génère des obstacles dans l'environnement.
		
		Args:
			count (int): Nombre d'obstacles à générer
			terrainFeatures (Dict[str, Any]): Caractéristiques du terrain
		"""
		# Utiliser les caractéristiques du terrain pour placer des obstacles
		# de manière plus naturelle
		
		# 1. Obstacles fixes basés sur le terrain (ex: montagnes, lacs)
		terrainObstacles = terrainFeatures.get("obstacles", [])
		for obstacle in terrainObstacles:
			self.obstacles.append(DynamicObstacle(
				position=obstacle.get("position"),
				radius=obstacle.get("radius", 20.0),
				isDynamic=False,
				traversable=obstacle.get("traversable", False)
			))
		
		# 2. Obstacles regroupés en clusters (ex: forêts)
		clusters = terrainFeatures.get("clusters", [])
		for cluster in clusters:
			center = cluster.get("center")
			radius = cluster.get("radius", 100.0)
			density = cluster.get("density", 0.5)
			
			# Nombre d'obstacles dans ce cluster
			clusterCount = int(density * 10)
			
			for _ in range(clusterCount):
				# Position aléatoire dans le cluster
				angle = random.uniform(0, 2 * math.pi)
				distance = random.uniform(0, radius)
				posX = center[0] + math.cos(angle) * distance
				posY = center[1] + math.sin(angle) * distance
				
				# S'assurer que la position est dans les limites
				posX = max(20.0, min(self.width - 20.0, posX))
				posY = max(20.0, min(self.height - 20.0, posY))
				
				# Créer l'obstacle
				self.obstacles.append(DynamicObstacle(
					position=(posX, posY),
					radius=random.uniform(10.0, 25.0),
					isDynamic=False
				))
		
		# 3. Obstacles restants placés de manière aléatoire
		remainingCount = max(0, count - len(self.obstacles))
		
		for _ in range(remainingCount):
			# Position aléatoire
			posX = random.uniform(20.0, self.width - 20.0)
			posY = random.uniform(20.0, self.height - 20.0)
			
			# Si on est trop près d'un obstacle existant, réessayer
			tooClose = any(self._getDistance((posX, posY), obs.position) < 40.0 
						 for obs in self.obstacles)
			
			if not tooClose:
				# Ajouter l'obstacle
				isDynamic = random.random() < 0.3  # 30% d'obstacles dynamiques
				
				self.obstacles.append(DynamicObstacle(
					position=(posX, posY),
					radius=random.uniform(10.0, 30.0),
					isDynamic=isDynamic,
					speed=random.uniform(0.5, 2.0) if isDynamic else 0.0,
					movementPattern="random" if isDynamic else None
				))
	
	def _generateResources(self, count: int, terrainFeatures: Dict[str, Any]) -> None:
		"""
		Génère des ressources dans l'environnement.
		
		Args:
			count (int): Nombre de ressources à générer
			terrainFeatures (Dict[str, Any]): Caractéristiques du terrain
		"""
		# 1. Ressources basées sur les caractéristiques du terrain
		resourceHotspots = terrainFeatures.get("resourceHotspots", [])
		
		for hotspot in resourceHotspots:
			center = hotspot.get("center")
			radius = hotspot.get("radius", 150.0)
			density = hotspot.get("density", 0.7)
			quality = hotspot.get("quality", 1.0)
			
			# Nombre de ressources dans ce hotspot
			hotspotCount = int(density * 5)
			
			for _ in range(hotspotCount):
				# Position aléatoire dans le hotspot
				angle = random.uniform(0, 2 * math.pi)
				distance = random.uniform(0, radius)
				posX = center[0] + math.cos(angle) * distance
				posY = center[1] + math.sin(angle) * distance
				
				# S'assurer que la position est dans les limites
				posX = max(10.0, min(self.width - 10.0, posX))
				posY = max(10.0, min(self.height - 10.0, posY))
				
				# Créer la ressource avec une valeur basée sur la qualité du hotspot
				value = random.uniform(0.8, 1.2) * quality
				
				# Vérifier qu'on n'est pas sur un obstacle
				if not self._isPositionOnObstacle((posX, posY)):
					self.resources.append(ResourceNode(
						position=(posX, posY),
						value=value,
						type=hotspot.get("type", "standard")
					))
		
		# 2. Ressources restantes placées de manière plus aléatoire
		remainingCount = max(0, count - len(self.resources))
		
		for _ in range(remainingCount):
			# Position aléatoire
			posX = random.uniform(20.0, self.width - 20.0)
			posY = random.uniform(20.0, self.height - 20.0)
			
			# Si on est sur un obstacle, réessayer
			if not self._isPositionOnObstacle((posX, posY)):
				# Ajouter la ressource
				self.resources.append(ResourceNode(
					position=(posX, posY),
					value=random.uniform(0.5, 1.5),
					type="standard"
				))
	
	def _updateOccupancyGrid(self) -> None:
		"""
		Met à jour la grille d'occupation pour les collisions.
		"""
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
	
	def _getGridCells(self, position: Tuple[float, float], radius: float) -> List[Tuple[int, int]]:
		"""
		Détermine les cellules de la grille occupées par un objet.
		
		Args:
			position (Tuple[float, float]): Position de l'objet
			radius (float): Rayon de l'objet
			
		Returns:
			List[Tuple[int, int]]: Liste des cellules occupées
		"""
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
	
	def _getDistance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
		"""
		Calcule la distance euclidienne entre deux positions.
		
		Args:
			pos1 (Tuple[float, float]): Première position
			pos2 (Tuple[float, float]): Deuxième position
			
		Returns:
			float: Distance entre les positions
		"""
		dx = pos1[0] - pos2[0]
		dy = pos1[1] - pos2[1]
		return math.sqrt(dx*dx + dy*dy)
	
	def _isPositionOnObstacle(self, position: Tuple[float, float], margin: float = 15.0) -> bool:
		"""
		Vérifie si une position est sur un obstacle.
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			margin (float): Marge supplémentaire autour des obstacles
			
		Returns:
			bool: True si la position est sur un obstacle, False sinon
		"""
		# Optimisation: utiliser la grille d'occupation
		cellX = int(position[0] / self.gridCellSize)
		cellY = int(position[1] / self.gridCellSize)
		cell = (cellX, cellY)
		
		if cell in self.occupancyGrid:
			for objType, obj in self.occupancyGrid[cell]:
				if objType == "obstacle":
					distance = self._getDistance(position, obj.position)
					if distance < obj.radius + margin:
						return True
		
		# Vérification directe (au cas où la grille n'est pas à jour)
		for obstacle in self.obstacles:
			distance = self._getDistance(position, obstacle.position)
			if distance < obstacle.radius + margin:
				return True
		
		return False
	
	def isPositionValid(self, position: Tuple[float, float]) -> bool:
		"""
		Vérifie si une position est valide (dans les limites et pas sur un obstacle).
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			
		Returns:
			bool: True si la position est valide, False sinon
		"""
		x, y = position
		
		# Vérifier les limites du monde
		if x < 0 or x >= self.width or y < 0 or y >= self.height:
			return False
			
		# Vérifier les collisions avec les obstacles
		return not self._isPositionOnObstacle(position)
	
	def getRandomValidPosition(self) -> Tuple[float, float]:
		"""
		Trouve une position aléatoire valide dans le monde.
		
		Returns:
			Tuple[float, float]: Position valide
		"""
		maxAttempts = 100
		
		for _ in range(maxAttempts):
			x = random.uniform(20.0, self.width - 20.0)
			y = random.uniform(20.0, self.height - 20.0)
			position = (x, y)
			
			if self.isPositionValid(position):
				return position
		
		# Si on ne trouve pas de position après plusieurs tentatives,
		# retourner une position par défaut
		return (self.width / 2, self.height / 2)
	
	def update(self) -> None:
		"""
		Met à jour l'état du monde pour le pas de temps suivant.
		"""
		# Incrémenter le temps
		self.currentTime += 1
		self.currentStep += 1
		
		# Mettre à jour les conditions météorologiques
		self.weatherSystem.update(self.currentTime)
		
		# Mettre à jour les obstacles dynamiques
		for obstacle in self.obstacles:
			if obstacle.isDynamic:
				# Sauvegarder l'ancienne position
				oldPosition = obstacle.position
				
				# Mettre à jour la position
				obstacle.update(self.currentTime)
				
				# Vérifier si la nouvelle position est valide
				if not self._isPositionInBounds(obstacle.position, obstacle.radius):
					obstacle.position = oldPosition
					obstacle.reverseDirection()
		
		# Mettre à jour la grille d'occupation si nécessaire
		# (pas à chaque pas de temps pour des raisons de performance)
		if self.currentStep % 5 == 0:
			self._updateOccupancyGrid()
	
	def _isPositionInBounds(self, position: Tuple[float, float], radius: float = 0.0) -> bool:
		"""
		Vérifie si une position est dans les limites du monde.
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			radius (float): Rayon de l'objet
			
		Returns:
			bool: True si la position est dans les limites, False sinon
		"""
		x, y = position
		return (radius <= x <= self.width - radius and
				radius <= y <= self.height - radius)
	
	def getWeatherConditions(self) -> Dict[str, float]:
		"""
		Récupère les conditions météorologiques actuelles.
		
		Returns:
			Dict[str, float]: Conditions météorologiques
		"""
		return self.weatherSystem.getCurrentConditions()
	
	def addEvent(self, eventType: str, position: Tuple[float, float], data: Dict[str, Any]) -> None:
		"""
		Ajoute un événement au monde.
		
		Args:
			eventType (str): Type d'événement
			position (Tuple[float, float]): Position de l'événement
			data (Dict[str, Any]): Données associées à l'événement
		"""
		self.events.append({
			"type": eventType,
			"position": position,
			"time": self.currentTime,
			"data": data
		})
	
	def getResourcesCollected(self) -> int:
		"""
		Retourne le nombre de ressources collectées.
		
		Returns:
			int: Nombre de ressources collectées
		"""
		return sum(1 for r in self.resources if r.isCollected)
	
	def getRemainingResources(self) -> int:
		"""
		Retourne le nombre de ressources restantes.
		
		Returns:
			int: Nombre de ressources non collectées
		"""
		return sum(1 for r in self.resources if not r.isCollected)
	
	def getWeatherEffect(self, position: Tuple[float, float]) -> Dict[str, float]:
		"""
		Calcule l'effet des conditions météorologiques à une position donnée.
		
		Args:
			position (Tuple[float, float]): Position pour laquelle calculer l'effet
			
		Returns:
			Dict[str, float]: Effets météorologiques (vitesse, visibilité, etc.)
		"""
		baseConditions = self.weatherSystem.getCurrentConditions()
		
		# Ajuster en fonction de la position (ex: effet réduit sous abri)
		nearbyObstacles = self._getNearbyObstacles(position, 50.0)
		shelterFactor = 0.0
		
		for obstacle in nearbyObstacles:
			distance = self._getDistance(position, obstacle.position)
			if distance < obstacle.radius * 2:
				# Plus on est proche, plus l'abri est efficace
				shelterFactor = max(shelterFactor, 1.0 - (distance / (obstacle.radius * 2)))
		
		# Appliquer l'effet d'abri
		adjustedConditions = {}
		for condition, value in baseConditions.items():
			if condition in ["windStrength", "rainIntensity", "snowIntensity"]:
				adjustedConditions[condition] = value * (1.0 - shelterFactor * 0.8)
			else:
				adjustedConditions[condition] = value
		
		return adjustedConditions
	
	def _getNearbyObstacles(self, position: Tuple[float, float], maxDistance: float) -> List[DynamicObstacle]:
		"""
		Récupère les obstacles proches d'une position.
		
		Args:
			position (Tuple[float, float]): Position centrale
			maxDistance (float): Distance maximale de recherche
			
		Returns:
			List[DynamicObstacle]: Liste des obstacles proches
		"""
		# Optimisation avec la grille d'occupation
		cellX = int(position[0] / self.gridCellSize)
		cellY = int(position[1] / self.gridCellSize)
		
		# Déterminer le rayon de recherche en cellules
		cellRadius = int(maxDistance / self.gridCellSize) + 1
		
		nearbyObstacles = []
		
		# Examiner les cellules voisines
		for i in range(max(0, cellX - cellRadius), min(int(self.width / self.gridCellSize), cellX + cellRadius + 1)):
			for j in range(max(0, cellY - cellRadius), min(int(self.height / self.gridCellSize), cellY + cellRadius + 1)):
				cell = (i, j)
				
				if cell in self.occupancyGrid:
					for objType, obj in self.occupancyGrid[cell]:
						if objType == "obstacle":
							distance = self._getDistance(position, obj.position)
							if distance <= maxDistance and obj not in nearbyObstacles:
								nearbyObstacles.append(obj)
		
		return nearbyObstacles
	
	def getTerrainType(self, position: Tuple[float, float]) -> str:
		"""
		Détermine le type de terrain à une position donnée.
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			
		Returns:
			str: Type de terrain
		"""
		return self.terrainGenerator.getTerrainType(position)
	
	def getTerrainCost(self, position: Tuple[float, float]) -> float:
		"""
		Calcule le coût de déplacement sur le terrain à une position donnée.
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			
		Returns:
			float: Coefficient de coût (1.0 = normal, >1.0 = difficile)
		"""
		terrainType = self.getTerrainType(position)
		
		# Coûts de déplacement par type de terrain
		terrainCosts = {
			"plains": 1.0,
			"forest": 1.5,
			"water": 3.0,
			"mountain": 2.0,
			"swamp": 2.5,
			"road": 0.8
		}
		
		return terrainCosts.get(terrainType, 1.0)
	
	def getVisibilityAtPosition(self, position: Tuple[float, float]) -> float:
		"""
		Calcule la visibilité à une position donnée (affectée par le terrain et la météo).
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			
		Returns:
			float: Facteur de visibilité (0-1), 1 étant une visibilité parfaite
		"""
		# Facteur de base
		visibility = 1.0
		
		# Réduction basée sur le terrain
		terrainType = self.getTerrainType(position)
		terrainVisibility = {
			"plains": 1.0,
			"forest": 0.7,
			"water": 0.9,
			"mountain": 0.8,
			"swamp": 0.6,
			"road": 1.0
		}
		visibility *= terrainVisibility.get(terrainType, 1.0)
		
		# Réduction basée sur la météo
		weatherConditions = self.getWeatherEffect(position)
		fogIntensity = weatherConditions.get("fogIntensity", 0.0)
		rainIntensity = weatherConditions.get("rainIntensity", 0.0)
		snowIntensity = weatherConditions.get("snowIntensity", 0.0)
		
		# Combiner les effets météo
		weatherReduction = max(fogIntensity * 0.8, rainIntensity * 0.5, snowIntensity * 0.4)
		visibility *= (1.0 - weatherReduction)
		
		return max(0.2, visibility)  # Garantir une visibilité minimale
	
	def getState(self) -> Dict[str, Any]:
		"""
		Retourne l'état actuel du monde pour la visualisation ou le débogage.
		
		Returns:
			Dict[str, Any]: État du monde
		"""
		return {
			"time": self.currentTime,
			"step": self.currentStep,
			"weather": self.weatherSystem.getCurrentConditions(),
			"resourcesCollected": self.getResourcesCollected(),
			"resourcesRemaining": self.getRemainingResources(),
			"episodeComplete": self.episodeComplete
		}
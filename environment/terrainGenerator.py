#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour la génération procédurale de terrains variés.
"""

import math
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class TerrainGenerator:
	"""
	Générateur de terrain procédural pour créer des environnements variés.
	
	Cette classe est responsable de:
	- Générer la topographie de base du monde
	- Placer des caractéristiques de terrain (montagnes, lacs, forêts)
	- Créer des zones d'intérêt pour les ressources
	- Fournir des informations sur le type de terrain à chaque position
	"""
	
	def __init__(
		self, 
		width: float = 1000.0,
		height: float = 1000.0,
		complexity: float = 0.5,
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise le générateur de terrain.
		
		Args:
			width (float): Largeur du monde
			height (float): Hauteur du monde
			complexity (float): Niveau de complexité du terrain (0-1)
			seed (Optional[int]): Graine pour la génération aléatoire, ou None
		"""
		self.width = width
		self.height = height
		self.complexity = complexity
		
		# Initialiser le générateur aléatoire
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
		
		# Paramètres de génération
		self.gridResolution = 50.0  # Résolution de la grille pour le bruit
		self.heightNoiseFreq = 0.01 * (0.5 + complexity)
		self.moistureNoiseFreq = 0.008 * (0.5 + complexity)
		
		# Décalages pour le bruit de Perlin
		self.heightOffset = (random.uniform(0, 1000), random.uniform(0, 1000))
		self.moistureOffset = (random.uniform(0, 1000), random.uniform(0, 1000))
		
		# Carte de hauteur et d'humidité
		self.heightMap = None
		self.moistureMap = None
		self.terrainTypeMap = None
		
		# Caractéristiques du terrain générées
		self.features = {
			"mountains": [],
			"lakes": [],
			"forests": [],
			"rivers": [],
			"roads": []
		}
		
		# Générer le terrain initial
		self._generateHeightMap()
		self._generateMoistureMap()
		self._determineTerrainTypes()
		self._placeTerrainFeatures()
	
	def generate(self) -> Dict[str, Any]:
		"""
		Génère un nouveau terrain et retourne ses caractéristiques.
		
		Returns:
			Dict[str, Any]: Caractéristiques du terrain pour l'environnement
		"""
		# Réinitialiser les caractéristiques
		self.features = {
			"mountains": [],
			"lakes": [],
			"forests": [],
			"rivers": [],
			"roads": []
		}
		
		# Nouveaux décalages pour le bruit
		self.heightOffset = (random.uniform(0, 1000), random.uniform(0, 1000))
		self.moistureOffset = (random.uniform(0, 1000), random.uniform(0, 1000))
		
		# Générer le nouveau terrain
		self._generateHeightMap()
		self._generateMoistureMap()
		self._determineTerrainTypes()
		self._placeTerrainFeatures()
		
		# Préparer les données pour l'environnement
		result = {
			# Obstacles basés sur le terrain
			"obstacles": self._createObstaclesFromFeatures(),
			
			# Clusters d'obstacles (ex: forêts)
			"clusters": self._createObstacleClusters(),
			
			# Zones riches en ressources
			"resourceHotspots": self._createResourceHotspots()
		}
		
		return result
	
	def _generateHeightMap(self) -> None:
		"""
		Génère une carte de hauteur à partir de bruit de Perlin.
		"""
		# Déterminer les dimensions de la grille
		gridWidth = int(self.width / self.gridResolution) + 1
		gridHeight = int(self.height / self.gridResolution) + 1
		
		# Initialiser la carte de hauteur
		self.heightMap = np.zeros((gridHeight, gridWidth))
		
		# Générer le bruit de Perlin
		for y in range(gridHeight):
			for x in range(gridWidth):
				# Coordonnées mondiales
				worldX = x * self.gridResolution
				worldY = y * self.gridResolution
				
				# Ajouter différentes octaves de bruit
				height = 0.0
				
				# Octave 1: grande échelle
				nx1 = worldX * self.heightNoiseFreq + self.heightOffset[0]
				ny1 = worldY * self.heightNoiseFreq + self.heightOffset[1]
				height += self._smoothNoise(nx1, ny1) * 0.6
				
				# Octave 2: moyenne échelle
				nx2 = worldX * self.heightNoiseFreq * 2 + self.heightOffset[0] + 50
				ny2 = worldY * self.heightNoiseFreq * 2 + self.heightOffset[1] + 50
				height += self._smoothNoise(nx2, ny2) * 0.3
				
				# Octave 3: petite échelle
				nx3 = worldX * self.heightNoiseFreq * 4 + self.heightOffset[0] + 100
				ny3 = worldY * self.heightNoiseFreq * 4 + self.heightOffset[1] + 100
				height += self._smoothNoise(nx3, ny3) * 0.1
				
				# Normaliser entre 0 et 1
				height = (height + 1) / 2
				
				# Appliquer une courbe de puissance pour accentuer les reliefs
				height = pow(height, 1.5)
				
				# Ajouter un gradient radial pour favoriser les plaines au centre
				centerX = self.width / 2
				centerY = self.height / 2
				distToCenter = math.sqrt((worldX - centerX)**2 + (worldY - centerY)**2)
				maxDist = math.sqrt((self.width/2)**2 + (self.height/2)**2)
				radialFactor = 0.5 * (distToCenter / maxDist)
				
				height = height * (1 - radialFactor) + radialFactor
				
				# Stocker dans la carte
				self.heightMap[y, x] = height
	
	def _generateMoistureMap(self) -> None:
		"""
		Génère une carte d'humidité à partir de bruit de Perlin.
		"""
		# Utiliser les mêmes dimensions que la carte de hauteur
		gridHeight, gridWidth = self.heightMap.shape
		
		# Initialiser la carte d'humidité
		self.moistureMap = np.zeros((gridHeight, gridWidth))
		
		# Générer le bruit de Perlin
		for y in range(gridHeight):
			for x in range(gridWidth):
				# Coordonnées mondiales
				worldX = x * self.gridResolution
				worldY = y * self.gridResolution
				
				# Ajouter différentes octaves de bruit
				moisture = 0.0
				
				# Octave 1: grande échelle
				nx1 = worldX * self.moistureNoiseFreq + self.moistureOffset[0]
				ny1 = worldY * self.moistureNoiseFreq + self.moistureOffset[1]
				moisture += self._smoothNoise(nx1, ny1) * 0.7
				
				# Octave 2: petite échelle
				nx2 = worldX * self.moistureNoiseFreq * 3 + self.moistureOffset[0] + 200
				ny2 = worldY * self.moistureNoiseFreq * 3 + self.moistureOffset[1] + 200
				moisture += self._smoothNoise(nx2, ny2) * 0.3
				
				# Normaliser entre 0 et 1
				moisture = (moisture + 1) / 2
				
				# Corrélation avec la hauteur (les zones basses sont plus humides)
				height = self.heightMap[y, x]
				moisture = moisture * 0.8 + (1 - height) * 0.2
				
				# Stocker dans la carte
				self.moistureMap[y, x] = moisture
	
	def _smoothNoise(self, x: float, y: float) -> float:
		"""
		Génère du bruit lisse à partir de fonctions sinusoïdales.
		
		Args:
			x (float): Coordonnée X
			y (float): Coordonnée Y
			
		Returns:
			float: Valeur de bruit entre -1 et 1
		"""
		# Utiliser des fonctions sinus pour générer du bruit pseudo-aléatoire
		noise = (
			math.sin(x * 12.9898 + y * 78.233) * 43758.5453 +
			math.sin(y * 19.8765 + x * 38.765) * 23421.6543
		)
		
		# Normaliser entre -1 et 1
		return math.sin(noise)
	
	def _determineTerrainTypes(self) -> None:
		"""
		Détermine les types de terrain à partir des cartes de hauteur et d'humidité.
		"""
		gridHeight, gridWidth = self.heightMap.shape
		
		# Initialiser la carte des types de terrain
		self.terrainTypeMap = np.empty((gridHeight, gridWidth), dtype=object)
		
		# Seuils de classification
		mountainThreshold = 0.75 - (self.complexity * 0.1)
		hillThreshold = 0.5
		waterThreshold = 0.3
		swampThreshold = 0.7
		forestThreshold = 0.6
		
		# Déterminer le type pour chaque cellule
		for y in range(gridHeight):
			for x in range(gridWidth):
				height = self.heightMap[y, x]
				moisture = self.moistureMap[y, x]
				
				# Classification
				if height > mountainThreshold:
					terrainType = "mountain"
				elif height > hillThreshold:
					if moisture > forestThreshold:
						terrainType = "forest"
					else:
						terrainType = "plains"
				elif height < waterThreshold:
					terrainType = "water"
				else:
					if moisture > swampThreshold:
						terrainType = "swamp"
					elif moisture > forestThreshold:
						terrainType = "forest"
					else:
						terrainType = "plains"
				
				self.terrainTypeMap[y, x] = terrainType
	
	def _placeTerrainFeatures(self) -> None:
		"""
		Place des caractéristiques spécifiques dans le terrain.
		"""
		gridHeight, gridWidth = self.heightMap.shape
		
		# 1. Identifier les régions de montagne
		mountains = []
		for y in range(gridHeight):
			for x in range(gridWidth):
				if self.terrainTypeMap[y, x] == "mountain":
					worldX = x * self.gridResolution
					worldY = y * self.gridResolution
					mountains.append((worldX, worldY, self.heightMap[y, x]))
		
		# Regrouper les montagnes en chaînes
		mountainRanges = self._clusterFeatures(mountains, clusterDistance=100.0)
		
		# Ajouter les chaînes de montagnes aux caractéristiques
		for mountainRange in mountainRanges:
			if len(mountainRange) > 3:  # Ignorer les petits groupes
				# Trouver le centre et la taille
				avgX = sum(m[0] for m in mountainRange) / len(mountainRange)
				avgY = sum(m[1] for m in mountainRange) / len(mountainRange)
				
				# Calculer le rayon approximatif
				maxDist = max(math.sqrt((m[0]-avgX)**2 + (m[1]-avgY)**2) for m in mountainRange)
				
				self.features["mountains"].append({
					"center": (avgX, avgY),
					"radius": maxDist + 50.0,
					"peaks": len(mountainRange),
					"height": max(m[2] for m in mountainRange)
				})
		
		# 2. Identifier les lacs
		lakes = []
		for y in range(gridHeight):
			for x in range(gridWidth):
				if self.terrainTypeMap[y, x] == "water":
					worldX = x * self.gridResolution
					worldY = y * self.gridResolution
					lakes.append((worldX, worldY))
		
		# Regrouper les points d'eau en lacs
		lakeClusters = self._clusterFeatures(lakes, clusterDistance=80.0)
		
		# Ajouter les lacs aux caractéristiques
		for lake in lakeClusters:
			if len(lake) > 5:  # Ignorer les très petits plans d'eau
				# Trouver le centre et la taille
				avgX = sum(l[0] for l in lake) / len(lake)
				avgY = sum(l[1] for l in lake) / len(lake)
				
				# Calculer le rayon approximatif
				maxDist = max(math.sqrt((l[0]-avgX)**2 + (l[1]-avgY)**2) for l in lake)
				
				self.features["lakes"].append({
					"center": (avgX, avgY),
					"radius": maxDist + 20.0,
					"size": len(lake)
				})
		
		# 3. Identifier les forêts
		forests = []
		for y in range(gridHeight):
			for x in range(gridWidth):
				if self.terrainTypeMap[y, x] == "forest":
					worldX = x * self.gridResolution
					worldY = y * self.gridResolution
					forests.append((worldX, worldY))
		
		# Regrouper les points forestiers
		forestClusters = self._clusterFeatures(forests, clusterDistance=120.0)
		
		# Ajouter les forêts aux caractéristiques
		for forest in forestClusters:
			if len(forest) > 8:  # Ignorer les petits bosquets
				# Trouver le centre et la taille
				avgX = sum(f[0] for f in forest) / len(forest)
				avgY = sum(f[1] for f in forest) / len(forest)
				
				# Calculer le rayon approximatif
				maxDist = max(math.sqrt((f[0]-avgX)**2 + (f[1]-avgY)**2) for f in forest)
				
				self.features["forests"].append({
					"center": (avgX, avgY),
					"radius": maxDist + 30.0,
					"density": min(1.0, len(forest) / 50.0)
				})
		
		# 4. Générer des routes
		self._generateRoads()
	
	def _clusterFeatures(self, points: List[Tuple], clusterDistance: float) -> List[List[Tuple]]:
		"""
		Regroupe des points proches en clusters.
		
		Args:
			points (List[Tuple]): Liste des points à regrouper
			clusterDistance (float): Distance maximum pour considérer deux points dans le même cluster
			
		Returns:
			List[List[Tuple]]: Liste des clusters
		"""
		if not points:
			return []
			
		# Tri initial pour la stabilité des résultats
		points = sorted(points, key=lambda p: p[0] + p[1] * 10000)
		
		clusters = []
		visited = set()
		
		for i, point in enumerate(points):
			if i in visited:
				continue
				
			# Nouveau cluster commençant par ce point
			cluster = [point]
			visited.add(i)
			
			# Recherche en largeur pour trouver les voisins
			queue = [i]
			while queue:
				current = queue.pop(0)
				currentPoint = points[current]
				
				for j, otherPoint in enumerate(points):
					if j not in visited:
						# Calculer la distance
						distance = math.sqrt((currentPoint[0] - otherPoint[0])**2 + 
										   (currentPoint[1] - otherPoint[1])**2)
						
						if distance <= clusterDistance:
							cluster.append(otherPoint)
							visited.add(j)
							queue.append(j)
			
			clusters.append(cluster)
		
		return clusters
	
	def _generateRoads(self) -> None:
		"""
		Génère un réseau de routes connectant les caractéristiques importantes.
		"""
		# Points d'intérêt à connecter
		poi = []
		
		# Ajouter les centres des caractéristiques importantes
		for feature_type in ["mountains", "lakes", "forests"]:
			for feature in self.features[feature_type]:
				poi.append({
					"position": feature["center"],
					"importance": 0.5 + random.random() * 0.5  # Importance aléatoire
				})
		
		# Ajouter des villes (points arbitraires de haute importance)
		numTowns = int(3 + self.complexity * 5)
		
		for _ in range(numTowns):
			# Placer les villes de préférence dans les plaines
			townPosition = None
			for _ in range(10):  # Essayer 10 fois de trouver une bonne position
				x = random.uniform(100, self.width - 100)
				y = random.uniform(100, self.height - 100)
				
				if self.getTerrainType((x, y)) == "plains":
					townPosition = (x, y)
					break
			
			if townPosition is None:
				# Si pas de plaine trouvée, position aléatoire
				townPosition = (
					random.uniform(100, self.width - 100),
					random.uniform(100, self.height - 100)
				)
			
			poi.append({
				"position": townPosition,
				"importance": 0.8 + random.random() * 0.2  # Haute importance
			})
		
		# Générer les routes principales entre les points les plus importants
		mainPOI = sorted(poi, key=lambda p: p["importance"], reverse=True)[:numTowns]
		
		# Construire un arbre couvrant minimal approximatif
		roads = []
		connected = set([0])  # Commencer par le premier point
		
		while len(connected) < len(mainPOI):
			bestDist = float('inf')
			bestRoad = None
			
			for i in connected:
				for j in range(len(mainPOI)):
					if j not in connected:
						p1 = mainPOI[i]["position"]
						p2 = mainPOI[j]["position"]
						
						dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
						
						if dist < bestDist:
							bestDist = dist
							bestRoad = (p1, p2)
			
			if bestRoad:
				roads.append(bestRoad)
				# Trouver l'indice du point nouvellement connecté
				for j in range(len(mainPOI)):
					if j not in connected and mainPOI[j]["position"] == bestRoad[1]:
						connected.add(j)
						break
		
		# Ajouter quelques routes secondaires
		numSecondaryRoads = int(self.complexity * 10)
		
		for _ in range(numSecondaryRoads):
			if len(poi) < 2:
				break
				
			# Sélectionner deux points aléatoires
			i, j = random.sample(range(len(poi)), 2)
			p1 = poi[i]["position"]
			p2 = poi[j]["position"]
			
			# Ne pas ajouter si la route est trop longue
			dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
			
			if dist < 300:
				roads.append((p1, p2))
		
		# Stocker les routes
		self.features["roads"] = roads
	
	def _createObstaclesFromFeatures(self) -> List[Dict[str, Any]]:
		"""
		Crée des obstacles basés sur les caractéristiques du terrain.
		
		Returns:
			List[Dict[str, Any]]: Liste des obstacles
		"""
		obstacles = []
		
		# Ajouter des obstacles pour les montagnes
		for mountain in self.features["mountains"]:
			# Créer plusieurs obstacles pour représenter la chaîne de montagnes
			numPeaks = min(8, max(3, int(mountain["peaks"] / 2)))
			
			for _ in range(numPeaks):
				# Position aléatoire dans la chaîne
				angle = random.uniform(0, 2 * math.pi)
				distance = random.uniform(0, mountain["radius"] * 0.8)
				
				posX = mountain["center"][0] + math.cos(angle) * distance
				posY = mountain["center"][1] + math.sin(angle) * distance
				
				# S'assurer que la position est dans les limites
				posX = max(50, min(self.width - 50, posX))
				posY = max(50, min(self.height - 50, posY))
				
				# Taille proportionnelle à la hauteur
				sizeScale = 0.5 + mountain["height"] * 0.5
				size = random.uniform(20, 40) * sizeScale
				
				obstacles.append({
					"position": (posX, posY),
					"radius": size,
					"traversable": False
				})
		
		# Ajouter des obstacles pour les lacs
		for lake in self.features["lakes"]:
			# Un grand obstacle pour représenter le lac
			obstacles.append({
				"position": lake["center"],
				"radius": lake["radius"] * 0.8,
				"traversable": True  # Les lacs sont traversables mais avec pénalité
			})
		
		return obstacles
	
	def _createObstacleClusters(self) -> List[Dict[str, Any]]:
		"""
		Crée des clusters d'obstacles basés sur les caractéristiques du terrain.
		
		Returns:
			List[Dict[str, Any]]: Liste des clusters d'obstacles
		"""
		clusters = []
		
		# Créer des clusters pour les forêts
		for forest in self.features["forests"]:
			clusters.append({
				"center": forest["center"],
				"radius": forest["radius"],
				"density": forest["density"],
				"type": "forest"
			})
		
		# Ajouter d'autres types de clusters si nécessaire
		# (par exemple, zones rocheuses, ruines, etc.)
		
		return clusters
	
	def _createResourceHotspots(self) -> List[Dict[str, Any]]:
		"""
		Crée des zones riches en ressources.
		
		Returns:
			List[Dict[str, Any]]: Liste des hotspots de ressources
		"""
		hotspots = []
		
		# 1. Ressources près des forêts
		for forest in self.features["forests"]:
			if random.random() < 0.7:  # 70% de chance d'avoir des ressources
				hotspots.append({
					"center": forest["center"],
					"radius": forest["radius"] * 0.7,
					"density": forest["density"] * 0.8,
					"quality": random.uniform(0.6, 0.9),
					"type": "forest_resource"
				})
		
		# 2. Ressources près des lacs
		for lake in self.features["lakes"]:
			if random.random() < 0.8:  # 80% de chance d'avoir des ressources
				# Placer les ressources sur le rivage
				angle = random.uniform(0, 2 * math.pi)
				posX = lake["center"][0] + math.cos(angle) * lake["radius"] * 1.2
				posY = lake["center"][1] + math.sin(angle) * lake["radius"] * 1.2
				
				# S'assurer que la position est dans les limites
				posX = max(50, min(self.width - 50, posX))
				posY = max(50, min(self.height - 50, posY))
				
				hotspots.append({
					"center": (posX, posY),
					"radius": lake["radius"] * 0.5,
					"density": 0.6,
					"quality": random.uniform(0.7, 1.0),
					"type": "water_resource"
				})
		
		# 3. Ressources rares dans les montagnes
		for mountain in self.features["mountains"]:
			if random.random() < 0.4:  # 40% de chance d'avoir des ressources
				hotspots.append({
					"center": mountain["center"],
					"radius": mountain["radius"] * 0.5,
					"density": 0.3,  # Faible densité
					"quality": random.uniform(0.8, 1.5),  # Mais haute qualité
					"type": "mountain_resource"
				})
		
		# 4. Quelques hotspots aléatoires
		numRandomHotspots = int(3 + self.complexity * 5)
		
		for _ in range(numRandomHotspots):
			posX = random.uniform(100, self.width - 100)
			posY = random.uniform(100, self.height - 100)
			
			hotspots.append({
				"center": (posX, posY),
				"radius": random.uniform(50, 150),
				"density": random.uniform(0.3, 0.8),
				"quality": random.uniform(0.6, 1.2),
				"type": "random_resource"
			})
		
		return hotspots
	
	def getTerrainType(self, position: Tuple[float, float]) -> str:
		"""
		Détermine le type de terrain à une position donnée.
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			
		Returns:
			str: Type de terrain
		"""
		if self.terrainTypeMap is None:
			return "plains"
			
		# Convertir la position en indices de grille
		x = int(position[0] / self.gridResolution)
		y = int(position[1] / self.gridResolution)
		
		# Vérifier les limites
		gridHeight, gridWidth = self.terrainTypeMap.shape
		
		if x < 0 or x >= gridWidth or y < 0 or y >= gridHeight:
			return "plains"  # Par défaut
		
		return self.terrainTypeMap[y, x]
	
	def getHeight(self, position: Tuple[float, float]) -> float:
		"""
		Récupère la hauteur du terrain à une position donnée.
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			
		Returns:
			float: Hauteur (0-1)
		"""
		if self.heightMap is None:
			return 0.5
			
		# Convertir la position en indices de grille
		x = int(position[0] / self.gridResolution)
		y = int(position[1] / self.gridResolution)
		
		# Vérifier les limites
		gridHeight, gridWidth = self.heightMap.shape
		
		if x < 0 or x >= gridWidth or y < 0 or y >= gridHeight:
			return 0.5  # Par défaut
		
		return self.heightMap[y, x]
	
	def getMoisture(self, position: Tuple[float, float]) -> float:
		"""
		Récupère l'humidité du terrain à une position donnée.
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			
		Returns:
			float: Humidité (0-1)
		"""
		if self.moistureMap is None:
			return 0.5
			
		# Convertir la position en indices de grille
		x = int(position[0] / self.gridResolution)
		y = int(position[1] / self.gridResolution)
		
		# Vérifier les limites
		gridHeight, gridWidth = self.moistureMap.shape
		
		if x < 0 or x >= gridWidth or y < 0 or y >= gridHeight:
			return 0.5  # Par défaut
		
		return self.moistureMap[y, x]
	
	def isOnRoad(self, position: Tuple[float, float], margin: float = 15.0) -> bool:
		"""
		Vérifie si une position est sur ou près d'une route.
		
		Args:
			position (Tuple[float, float]): Position à vérifier
			margin (float): Marge autour de la route
			
		Returns:
			bool: True si la position est sur ou près d'une route
		"""
		for start, end in self.features.get("roads", []):
			# Calculer la distance point-segment
			x, y = position
			x1, y1 = start
			x2, y2 = end
			
			# Vecteur de la ligne
			line_vec = (x2 - x1, y2 - y1)
			line_len = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
			
			if line_len == 0:
				# Les points sont identiques
				dist = math.sqrt((x - x1)**2 + (y - y1)**2)
			else:
				# Normaliser le vecteur
				line_vec = (line_vec[0] / line_len, line_vec[1] / line_len)
				
				# Vecteur du point au début de la ligne
				point_vec = (x - x1, y - y1)
				
				# Projection du point sur la ligne
				proj_len = point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]
				
				# Limiter la projection à la longueur de la ligne
				proj_len = max(0, min(line_len, proj_len))
				
				# Point projeté
				proj_x = x1 + line_vec[0] * proj_len
				proj_y = y1 + line_vec[1] * proj_len
				
				# Distance entre le point et sa projection
				dist = math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
			
			if dist <= margin:
				return True
		
		return False
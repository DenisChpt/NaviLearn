#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module implémentant la mémoire de connaissances pour les agents collectifs.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
from collections import deque


class KnowledgeMemory:
	"""
	Mémoire de connaissances pour stocker et organiser les informations recueillies par les agents.
	
	Cette classe permet:
	- Le stockage structuré d'observations
	- La gestion des connaissances avec dégradation temporelle
	- La fusion d'informations provenant de différentes sources
	- Le calcul de statistiques sur les connaissances acquises
	"""
	
	def __init__(self, capacity: int = 1000, spatialResolution: float = 20.0) -> None:
		"""
		Initialise une nouvelle mémoire de connaissances.
		
		Args:
			capacity (int): Capacité maximale de la mémoire (nombre d'entrées)
			spatialResolution (float): Résolution spatiale pour la fusion d'observations
		"""
		self.capacity = capacity
		self.spatialResolution = spatialResolution
		self.entries = deque(maxlen=capacity)
		self.exploredPositions = set()  # Positions visitées pour la couverture
		self.avoidanceRegions = []  # Régions à éviter
		self.lastUpdateTime = 0
		self.memoryQuadrants = {}  # Pour une organisation spatiale des connaissances
	
	def addEntry(self, entry: Dict[str, Any]) -> None:
		"""
		Ajoute une nouvelle entrée dans la mémoire.
		
		Args:
			entry (Dict[str, Any]): Entrée à ajouter, contenant au minimum:
				- type: Type d'entrée (ex: "resource", "obstacle", "agent")
				- position: Coordonnées (x, y) si applicable
				- timestamp: Horodatage de l'observation
				- certainty: Niveau de certitude (0-1)
				- source: ID de la source (agent ayant fourni l'information)
		"""
		# Mettre à jour l'horodatage du dernier ajout
		self.lastUpdateTime = max(self.lastUpdateTime, entry.get("timestamp", 0))
		
		# Si une entrée similaire existe déjà, la fusionner plutôt qu'ajouter
		if self._shouldMergeEntry(entry):
			self._mergeEntry(entry)
		else:
			# Ajouter la nouvelle entrée
			self.entries.append(entry)
			
			# Si c'est une position d'agent, l'ajouter aux positions explorées
			if entry.get("type") == "agent_position" and "position" in entry:
				self.exploredPositions.add(self._discretizePosition(entry["position"]))
				
				# Mettre à jour l'organisation spatiale
				self._updateSpatialIndex(entry)
	
	def addPositionEntry(self, position: Tuple[float, float], timestamp: float) -> None:
		"""
		Ajoute une entrée pour la position actuelle de l'agent.
		
		Args:
			position (Tuple[float, float]): Position (x, y)
			timestamp (float): Horodatage
		"""
		entry = {
			"type": "agent_position",
			"position": position,
			"timestamp": timestamp,
			"certainty": 1.0,
			"source": "self"
		}
		
		self.addEntry(entry)
		
		# Ajouter aux positions explorées
		self.exploredPositions.add(self._discretizePosition(position))
	
	def addAvoidanceRegion(self, region: List[Tuple[float, float]]) -> None:
		"""
		Ajoute une région à éviter.
		
		Args:
			region (List[Tuple[float, float]]): Liste de positions définissant la région
		"""
		if region:
			self.avoidanceRegions.append({
				"positions": region,
				"timestamp": self.lastUpdateTime
			})
	
	def _discretizePosition(self, position: Tuple[float, float]) -> Tuple[int, int]:
		"""
		Discrétise une position pour le stockage spatial.
		
		Args:
			position (Tuple[float, float]): Position (x, y)
			
		Returns:
			Tuple[int, int]: Position discrétisée
		"""
		x, y = position
		return (int(x // self.spatialResolution), int(y // self.spatialResolution))
	
	def _updateSpatialIndex(self, entry: Dict[str, Any]) -> None:
		"""
		Met à jour l'index spatial des connaissances.
		
		Args:
			entry (Dict[str, Any]): Entrée à indexer
		"""
		if "position" not in entry:
			return
			
		# Déterminer le quadrant
		position = entry["position"]
		discretePos = self._discretizePosition(position)
		
		# Ajouter l'entrée au quadrant
		if discretePos not in self.memoryQuadrants:
			self.memoryQuadrants[discretePos] = []
			
		self.memoryQuadrants[discretePos].append(entry)
	
	def _shouldMergeEntry(self, newEntry: Dict[str, Any]) -> bool:
		"""
		Détermine si une nouvelle entrée devrait être fusionnée avec une entrée existante.
		
		Args:
			newEntry (Dict[str, Any]): Nouvelle entrée
			
		Returns:
			bool: True si l'entrée devrait être fusionnée, False sinon
		"""
		# On ne fusionne que certains types d'entrées
		if newEntry.get("type") not in ["resource", "obstacle", "feedback"]:
			return False
			
		# L'entrée doit avoir une position
		if "position" not in newEntry:
			return False
			
		# Chercher des entrées similaires
		newPos = newEntry["position"]
		entriesOfSameType = [e for e in self.entries if e.get("type") == newEntry.get("type")]
		
		for existingEntry in entriesOfSameType:
			if "position" in existingEntry:
				existingPos = existingEntry["position"]
				
				# Calculer la distance
				dx = existingPos[0] - newPos[0]
				dy = existingPos[1] - newPos[1]
				distance = math.sqrt(dx*dx + dy*dy)
				
				# Fusionner si la distance est inférieure à la résolution spatiale
				if distance < self.spatialResolution:
					return True
		
		return False
	
	def _mergeEntry(self, newEntry: Dict[str, Any]) -> None:
		"""
		Fusionne une nouvelle entrée avec une entrée existante similaire.
		
		Args:
			newEntry (Dict[str, Any]): Nouvelle entrée à fusionner
		"""
		# Trouver l'entrée existante à fusionner
		newPos = newEntry["position"]
		entriesOfSameType = [e for e in self.entries if e.get("type") == newEntry.get("type")]
		
		for i, existingEntry in enumerate(entriesOfSameType):
			if "position" in existingEntry:
				existingPos = existingEntry["position"]
				
				# Calculer la distance
				dx = existingPos[0] - newPos[0]
				dy = existingPos[1] - newPos[1]
				distance = math.sqrt(dx*dx + dy*dy)
				
				if distance < self.spatialResolution:
					# Fusionner les informations
					mergedEntry = existingEntry.copy()
					
					# Mettre à jour la position (moyenne pondérée par la certitude)
					w1 = existingEntry.get("certainty", 0.5)
					w2 = newEntry.get("certainty", 0.5)
					totalWeight = w1 + w2
					
					mergedX = (existingPos[0] * w1 + newPos[0] * w2) / totalWeight
					mergedY = (existingPos[1] * w1 + newPos[1] * w2) / totalWeight
					mergedEntry["position"] = (mergedX, mergedY)
					
					# Mettre à jour la certitude (augmente avec les observations multiples)
					mergedEntry["certainty"] = min(1.0, existingEntry.get("certainty", 0.5) + 
												newEntry.get("certainty", 0.5) * 0.2)
					
					# Prendre l'horodatage le plus récent
					mergedEntry["timestamp"] = max(existingEntry.get("timestamp", 0), 
												newEntry.get("timestamp", 0))
					
					# Mettre à jour les valeurs (pour les ressources)
					if "value" in existingEntry and "value" in newEntry:
						mergedEntry["value"] = (existingEntry["value"] * w1 + 
											  newEntry["value"] * w2) / totalWeight
					
					# Conserver la provenance des données
					if "source" in existingEntry and "source" in newEntry:
						if existingEntry["source"] != newEntry["source"]:
							mergedEntry["source"] = f"{existingEntry['source']},{newEntry['source']}"
					
					# Remplacer l'entrée existante
					self.entries.remove(existingEntry)
					self.entries.append(mergedEntry)
					
					# Mettre à jour l'index spatial
					self._updateSpatialIndex(mergedEntry)
					return
	
	def getEntries(self, entryType: Optional[str] = None) -> List[Dict[str, Any]]:
		"""
		Récupère toutes les entrées, éventuellement filtrées par type.
		
		Args:
			entryType (Optional[str]): Type d'entrée à filtrer, ou None pour toutes
			
		Returns:
			List[Dict[str, Any]]: Liste des entrées correspondantes
		"""
		if entryType is None:
			return list(self.entries)
		else:
			return [e for e in self.entries if e.get("type") == entryType]
	
	def getRecentEntries(self, count: int, minCertainty: float = 0.0) -> List[Dict[str, Any]]:
		"""
		Récupère les entrées les plus récentes avec un niveau de certitude minimum.
		
		Args:
			count (int): Nombre maximum d'entrées à récupérer
			minCertainty (float): Niveau de certitude minimum (0-1)
			
		Returns:
			List[Dict[str, Any]]: Liste des entrées récentes
		"""
		sortedEntries = sorted(
			[e for e in self.entries if e.get("certainty", 0) >= minCertainty],
			key=lambda e: e.get("timestamp", 0),
			reverse=True
		)
		
		return sortedEntries[:count]
	
	def getEntriesInRegion(self, region: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
		"""
		Récupère les entrées situées dans une région spécifique.
		
		Args:
			region (Tuple[float, float, float, float]): Région (x1, y1, x2, y2)
			
		Returns:
			List[Dict[str, Any]]: Liste des entrées dans la région
		"""
		x1, y1, x2, y2 = region
		result = []
		
		for entry in self.entries:
			if "position" in entry:
				x, y = entry["position"]
				if x1 <= x <= x2 and y1 <= y <= y2:
					result.append(entry)
		
		return result
	
	def getExploredRegions(self, minTimestamp: float = 0) -> List[Tuple[float, float]]:
		"""
		Récupère les positions explorées après un certain temps.
		
		Args:
			minTimestamp (float): Horodatage minimum
			
		Returns:
			List[Tuple[float, float]]: Liste des positions explorées
		"""
		exploredPositions = []
		
		for entry in self.entries:
			if (entry.get("type") == "agent_position" and 
				entry.get("timestamp", 0) >= minTimestamp and
				"position" in entry):
				exploredPositions.append(entry["position"])
		
		return exploredPositions
	
	def getRecentAgentPositions(self, minTimestamp: float) -> List[Tuple[float, float]]:
		"""
		Récupère les positions récentes des agents.
		
		Args:
			minTimestamp (float): Horodatage minimum
			
		Returns:
			List[Tuple[float, float]]: Liste des positions d'agents
		"""
		positions = []
		
		for entry in self.entries:
			if (entry.get("type") == "agent_position" and 
				entry.get("timestamp", 0) >= minTimestamp and
				"position" in entry):
				positions.append(entry["position"])
		
		return positions
	
	def getFreshness(self, currentTime: float) -> float:
		"""
		Calcule la fraîcheur moyenne des connaissances.
		
		Args:
			currentTime (float): Temps actuel
			
		Returns:
			float: Indice de fraîcheur (0-1), 1 étant le plus frais
		"""
		if not self.entries:
			return 0.0
			
		totalAge = 0.0
		entryCount = len(self.entries)
		
		for entry in self.entries:
			age = currentTime - entry.get("timestamp", 0)
			totalAge += max(0, min(1000, age))  # Limiter l'âge à 1000 unités de temps
		
		averageAge = totalAge / entryCount
		freshness = 1.0 - (averageAge / 1000.0)
		
		return max(0.0, freshness)
	
	def getCoverageRatio(self, worldWidth: float, worldHeight: float, resolution: Optional[float] = None) -> float:
		"""
		Calcule le ratio de couverture de l'environnement.
		
		Args:
			worldWidth (float): Largeur du monde
			worldHeight (float): Hauteur du monde
			resolution (Optional[float]): Résolution pour le calcul (utilise spatialResolution par défaut)
			
		Returns:
			float: Ratio de couverture (0-1)
		"""
		if resolution is None:
			resolution = self.spatialResolution
			
		# Calculer le nombre total de cellules dans le monde
		totalCells = (worldWidth / resolution) * (worldHeight / resolution)
		
		# Compter le nombre de cellules explorées
		exploredCells = len(self.exploredPositions)
		
		return min(1.0, exploredCells / totalCells)
	
	def getLastUpdateTime(self) -> float:
		"""
		Retourne l'horodatage du dernier ajout à la mémoire.
		
		Returns:
			float: Horodatage du dernier ajout
		"""
		return self.lastUpdateTime
	
	def clearTemporaryData(self) -> None:
		"""
		Efface les données temporaires tout en conservant les connaissances importantes.
		"""
		# Conserver uniquement les connaissances spatiales importantes
		importantEntries = [
			entry for entry in self.entries
			if entry.get("type") in ["resource", "obstacle"] and entry.get("certainty", 0) > 0.7
		]
		
		# Réinitialiser la mémoire
		self.entries = deque(importantEntries, maxlen=self.capacity)
		
		# Conserver les positions explorées
		# (on pourrait aussi les réinitialiser si on veut encourager la ré-exploration)
		
		# Vider les régions à éviter
		self.avoidanceRegions = []
	
	def decay(self, currentTime: float, decayRate: float = 0.001) -> None:
		"""
		Applique une dégradation temporelle aux connaissances.
		
		Args:
			currentTime (float): Temps actuel
			decayRate (float): Taux de dégradation par unité de temps
		"""
		for entry in self.entries:
			age = currentTime - entry.get("timestamp", currentTime)
			
			# Appliquer la dégradation à la certitude
			if "certainty" in entry:
				decayFactor = 1.0 - (decayRate * age)
				entry["certainty"] = max(0.1, entry["certainty"] * decayFactor)
	
	def prune(self, minCertainty: float = 0.2) -> None:
		"""
		Supprime les entrées dont la certitude est tombée sous un certain seuil.
		
		Args:
			minCertainty (float): Certitude minimale à conserver
		"""
		# Créer une nouvelle liste d'entrées
		newEntries = deque(
			[entry for entry in self.entries if entry.get("certainty", 0) >= minCertainty],
			maxlen=self.capacity
		)
		
		self.entries = newEntries
	
	def getResourceDistribution(self) -> Dict[Tuple[int, int], float]:
		"""
		Calcule la distribution spatiale des ressources connues.
		
		Returns:
			Dict[Tuple[int, int], float]: Carte de densité des ressources
		"""
		resourceMap = {}
		
		for entry in self.entries:
			if entry.get("type") == "resource" and "position" in entry:
				pos = self._discretizePosition(entry["position"])
				value = entry.get("value", 1.0) * entry.get("certainty", 0.5)
				
				if pos in resourceMap:
					resourceMap[pos] += value
				else:
					resourceMap[pos] = value
		
		return resourceMap
	
	def getHeatmap(self, entryType: str, worldWidth: float, worldHeight: float, 
				 resolution: Optional[float] = None) -> np.ndarray:
		"""
		Génère une carte de chaleur pour un type d'entrée spécifique.
		
		Args:
			entryType (str): Type d'entrée à visualiser
			worldWidth (float): Largeur du monde
			worldHeight (float): Hauteur du monde
			resolution (Optional[float]): Résolution de la carte
			
		Returns:
			np.ndarray: Carte de chaleur 2D
		"""
		if resolution is None:
			resolution = self.spatialResolution
			
		# Dimensions de la carte
		width_cells = int(worldWidth / resolution) + 1
		height_cells = int(worldHeight / resolution) + 1
		
		# Initialiser la carte
		heatmap = np.zeros((height_cells, width_cells))
		
		# Remplir la carte
		for entry in self.entries:
			if entry.get("type") == entryType and "position" in entry:
				x, y = entry["position"]
				cell_x = min(width_cells - 1, int(x / resolution))
				cell_y = min(height_cells - 1, int(y / resolution))
				
				value = entry.get("certainty", 0.5)
				if entryType == "resource" and "value" in entry:
					value *= entry["value"]
				
				heatmap[cell_y, cell_x] += value
		
		# Normaliser
		if np.max(heatmap) > 0:
			heatmap = heatmap / np.max(heatmap)
		
		return heatmap
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module gérant les conditions météorologiques dynamiques de l'environnement.
"""

import math
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class WeatherSystem:
	"""
	Système météorologique dynamique qui simule différentes conditions
	affectant les agents et l'environnement.
	
	Ce système gère:
	- La génération de conditions météorologiques changeantes
	- Les transitions graduelles entre différents états météorologiques
	- Les effets de la météo sur la visibilité, la vitesse et d'autres facteurs
	"""
	
	# Définition des types de météo
	WEATHER_TYPES = {
		"clear": {
			"visibility": 1.0,
			"windStrength": 0.1,
			"temperature": 0.7,
			"rainIntensity": 0.0,
			"fogIntensity": 0.0,
			"snowIntensity": 0.0
		},
		"cloudy": {
			"visibility": 0.8,
			"windStrength": 0.3,
			"temperature": 0.5,
			"rainIntensity": 0.0,
			"fogIntensity": 0.2,
			"snowIntensity": 0.0
		},
		"rainy": {
			"visibility": 0.6,
			"windStrength": 0.5,
			"temperature": 0.4,
			"rainIntensity": 0.7,
			"fogIntensity": 0.3,
			"snowIntensity": 0.0
		},
		"stormy": {
			"visibility": 0.4,
			"windStrength": 0.9,
			"temperature": 0.3,
			"rainIntensity": 0.9,
			"fogIntensity": 0.3,
			"snowIntensity": 0.0
		},
		"foggy": {
			"visibility": 0.3,
			"windStrength": 0.2,
			"temperature": 0.4,
			"rainIntensity": 0.1,
			"fogIntensity": 0.9,
			"snowIntensity": 0.0
		},
		"snowy": {
			"visibility": 0.5,
			"windStrength": 0.4,
			"temperature": 0.1,
			"rainIntensity": 0.0,
			"fogIntensity": 0.3,
			"snowIntensity": 0.7
		},
		"blizzard": {
			"visibility": 0.2,
			"windStrength": 0.8,
			"temperature": 0.0,
			"rainIntensity": 0.0,
			"fogIntensity": 0.5,
			"snowIntensity": 0.9
		}
	}
	
	def __init__(
		self, 
		enabled: bool = True,
		changeFrequency: float = 0.002,  # Probabilité de changement par pas de temps
		transitionSpeed: float = 0.01,    # Vitesse de transition entre états
		initialWeather: Optional[str] = None
	) -> None:
		"""
		Initialise le système météorologique.
		
		Args:
			enabled (bool): Active ou désactive les effets météorologiques
			changeFrequency (float): Probabilité de changement par pas de temps
			transitionSpeed (float): Vitesse de transition entre états
			initialWeather (Optional[str]): Type de météo initial, ou None pour aléatoire
		"""
		self.enabled = enabled
		self.changeFrequency = changeFrequency
		self.transitionSpeed = transitionSpeed
		
		# Sélectionner la météo initiale
		if initialWeather is not None and initialWeather in self.WEATHER_TYPES:
			self.currentWeatherType = initialWeather
		else:
			self.currentWeatherType = random.choice(list(self.WEATHER_TYPES.keys()))
		
		# État météorologique actuel (valeurs réelles)
		self.currentConditions = self.WEATHER_TYPES[self.currentWeatherType].copy()
		
		# État cible lors d'une transition
		self.targetWeatherType = self.currentWeatherType
		self.targetConditions = self.WEATHER_TYPES[self.targetWeatherType].copy()
		
		# État de transition
		self.isTransitioning = False
		self.transitionProgress = 0.0
		
		# Variations aléatoires pour ajouter du réalisme
		self.noiseFrequency = 0.05
		self.noiseAmplitude = 0.1
		self.timeOffset = random.uniform(0, 1000)
		
		# Historique météorologique
		self.weatherHistory = []
		self.maxHistoryLength = 100
		
		# Effets sur les agents
		self.speedModifier = 1.0
		self.visibilityModifier = 1.0
		
		# Ajouter des variations initiales
		self._applyRandomVariations()
	
	def reset(self) -> None:
		"""
		Réinitialise le système météorologique.
		"""
		# Sélectionner une nouvelle météo initiale
		self.currentWeatherType = random.choice(list(self.WEATHER_TYPES.keys()))
		self.currentConditions = self.WEATHER_TYPES[self.currentWeatherType].copy()
		
		# Réinitialiser l'état de transition
		self.targetWeatherType = self.currentWeatherType
		self.targetConditions = self.currentConditions.copy()
		self.isTransitioning = False
		self.transitionProgress = 0.0
		
		# Réinitialiser l'historique
		self.weatherHistory = []
		
		# Nouvelle origine temporelle pour les variations
		self.timeOffset = random.uniform(0, 1000)
		
		# Appliquer des variations initiales
		self._applyRandomVariations()
	
	def update(self, currentTime: int) -> None:
		"""
		Met à jour les conditions météorologiques.
		
		Args:
			currentTime (int): Temps actuel de la simulation
		"""
		if not self.enabled:
			return
		
		# Décider si on change de météo
		if not self.isTransitioning and random.random() < self.changeFrequency:
			self._initiateWeatherChange()
		
		# Mettre à jour la transition en cours
		if self.isTransitioning:
			self._updateTransition()
		
		# Appliquer des variations aléatoires pour le réalisme
		self._applyRandomVariations(currentTime)
		
		# Enregistrer l'historique
		self._recordHistory()
		
		# Mettre à jour les modificateurs d'effet
		self._updateEffectModifiers()
	
	def _initiateWeatherChange(self) -> None:
		"""
		Initie un changement de conditions météorologiques.
		"""
		# Sélectionner un nouveau type de météo (différent de l'actuel)
		possibleTypes = list(self.WEATHER_TYPES.keys())
		possibleTypes.remove(self.currentWeatherType)
		
		# Prendre en compte les transitions logiques
		# (ex: de clear à stormy est peu probable, mais clear à cloudy est plus probable)
		weightedTypes = self._getWeightedTypes()
		
		# Sélectionner le nouveau type
		self.targetWeatherType = random.choices(
			list(weightedTypes.keys()),
			weights=list(weightedTypes.values()),
			k=1
		)[0]
		
		# Définir les conditions cibles
		self.targetConditions = self.WEATHER_TYPES[self.targetWeatherType].copy()
		
		# Initialiser la transition
		self.isTransitioning = True
		self.transitionProgress = 0.0
	
	def _getWeightedTypes(self) -> Dict[str, float]:
		"""
		Calcule des probabilités pondérées pour les transitions météorologiques.
		
		Returns:
			Dict[str, float]: Types météo avec leur probabilité de sélection
		"""
		weights = {}
		currentType = self.currentWeatherType
		
		# Définir les probabilités de transition entre types de météo
		transitionMatrix = {
			"clear": {"cloudy": 0.6, "foggy": 0.3, "rainy": 0.1, "stormy": 0.0, "snowy": 0.0, "blizzard": 0.0},
			"cloudy": {"clear": 0.4, "foggy": 0.2, "rainy": 0.3, "stormy": 0.1, "snowy": 0.0, "blizzard": 0.0},
			"rainy": {"clear": 0.1, "cloudy": 0.4, "foggy": 0.2, "stormy": 0.3, "snowy": 0.0, "blizzard": 0.0},
			"stormy": {"clear": 0.0, "cloudy": 0.2, "rainy": 0.6, "foggy": 0.1, "snowy": 0.1, "blizzard": 0.0},
			"foggy": {"clear": 0.3, "cloudy": 0.4, "rainy": 0.2, "stormy": 0.0, "snowy": 0.1, "blizzard": 0.0},
			"snowy": {"clear": 0.1, "cloudy": 0.3, "foggy": 0.2, "rainy": 0.0, "stormy": 0.0, "blizzard": 0.4},
			"blizzard": {"clear": 0.0, "cloudy": 0.1, "foggy": 0.1, "rainy": 0.0, "stormy": 0.1, "snowy": 0.7}
		}
		
		# Utiliser la matrice de transition
		for targetType in self.WEATHER_TYPES:
			if targetType != currentType:
				weights[targetType] = transitionMatrix.get(currentType, {}).get(targetType, 0.1)
		
		return weights
	
	def _updateTransition(self) -> None:
		"""
		Met à jour la transition entre deux états météorologiques.
		"""
		# Progresser dans la transition
		self.transitionProgress += self.transitionSpeed
		
		if self.transitionProgress >= 1.0:
			# Transition terminée
			self.currentWeatherType = self.targetWeatherType
			self.currentConditions = self.targetConditions.copy()
			self.isTransitioning = False
			self.transitionProgress = 0.0
		else:
			# Interpoler entre les conditions actuelles et cibles
			for key in self.currentConditions:
				if key in self.targetConditions:
					self.currentConditions[key] = (
						(1 - self.transitionProgress) * self.currentConditions[key] +
						self.transitionProgress * self.targetConditions[key]
					)
	
	def _applyRandomVariations(self, currentTime: int = 0) -> None:
		"""
		Applique de petites variations aléatoires aux conditions actuelles.
		
		Args:
			currentTime (int): Temps actuel pour le bruit cohérent
		"""
		# Utiliser le temps pour générer du bruit cohérent
		t = currentTime * self.noiseFrequency + self.timeOffset
		
		# Appliquer des variations à chaque paramètre
		for key in self.currentConditions:
			# Générer un bruit sinusoïdal
			noise = math.sin(t + hash(key) % 100) * self.noiseAmplitude
			
			# Appliquer le bruit
			self.currentConditions[key] = max(0.0, min(1.0, self.currentConditions[key] + noise))
	
	def _recordHistory(self) -> None:
		"""
		Enregistre l'état météorologique actuel dans l'historique.
		"""
		# Ajouter l'état actuel
		self.weatherHistory.append({
			"type": self.currentWeatherType,
			"conditions": self.currentConditions.copy()
		})
		
		# Limiter la taille de l'historique
		if len(self.weatherHistory) > self.maxHistoryLength:
			self.weatherHistory.pop(0)
	
	def _updateEffectModifiers(self) -> None:
		"""
		Met à jour les modificateurs d'effets basés sur les conditions actuelles.
		"""
		# Modificateur de vitesse (vent, pluie, neige)
		windEffect = self.currentConditions["windStrength"] * 0.3
		rainEffect = self.currentConditions["rainIntensity"] * 0.4
		snowEffect = self.currentConditions["snowIntensity"] * 0.6
		
		# Plus de vent, pluie ou neige = mouvement plus lent
		self.speedModifier = 1.0 - max(windEffect, rainEffect, snowEffect)
		self.speedModifier = max(0.4, self.speedModifier)  # Au moins 40% de la vitesse normale
		
		# Modificateur de visibilité (brouillard, pluie, neige)
		fogEffect = self.currentConditions["fogIntensity"] * 0.7
		rainEffect = self.currentConditions["rainIntensity"] * 0.4
		snowEffect = self.currentConditions["snowIntensity"] * 0.5
		
		# Plus de brouillard, pluie ou neige = visibilité réduite
		self.visibilityModifier = 1.0 - max(fogEffect, rainEffect, snowEffect)
		self.visibilityModifier = max(0.3, self.visibilityModifier)  # Au moins 30% de visibilité
	
	def getCurrentConditions(self) -> Dict[str, float]:
		"""
		Retourne les conditions météorologiques actuelles.
		
		Returns:
			Dict[str, float]: Conditions actuelles
		"""
		if not self.enabled:
			# Retourner des conditions par défaut si désactivé
			return {
				"visibility": 1.0,
				"windStrength": 0.0,
				"temperature": 0.5,
				"rainIntensity": 0.0,
				"fogIntensity": 0.0,
				"snowIntensity": 0.0,
				"type": "clear",
				"speedModifier": 1.0,
				"visibilityModifier": 1.0
			}
		
		# Copier les conditions actuelles
		conditions = self.currentConditions.copy()
		
		# Ajouter des informations supplémentaires
		conditions["type"] = self.currentWeatherType
		conditions["speedModifier"] = self.speedModifier
		conditions["visibilityModifier"] = self.visibilityModifier
		
		return conditions
	
	def getSpeedModifier(self) -> float:
		"""
		Retourne le modificateur de vitesse actuel.
		
		Returns:
			float: Modificateur de vitesse (0-1)
		"""
		return self.speedModifier if self.enabled else 1.0
	
	def getVisibilityModifier(self) -> float:
		"""
		Retourne le modificateur de visibilité actuel.
		
		Returns:
			float: Modificateur de visibilité (0-1)
		"""
		return self.visibilityModifier if self.enabled else 1.0
	
	def getWeatherTrend(self, timeSpan: int = 50) -> Dict[str, Any]:
		"""
		Analyse la tendance météorologique sur une période donnée.
		
		Args:
			timeSpan (int): Nombre d'étapes à analyser
			
		Returns:
			Dict[str, Any]: Informations sur la tendance météorologique
		"""
		if not self.weatherHistory:
			return {"trend": "stable", "direction": "none"}
			
		# Limiter à l'historique disponible
		historySpan = min(len(self.weatherHistory), timeSpan)
		
		if historySpan < 5:  # Pas assez de données
			return {"trend": "unknown", "direction": "none"}
			
		# Analyser les tendances pour chaque condition
		trends = {}
		
		for key in self.currentConditions:
			values = [entry["conditions"].get(key, 0) for entry in self.weatherHistory[-historySpan:]]
			
			if len(values) >= 2:
				# Calculer la tendance linéaire simple
				x = list(range(len(values)))
				slope = np.polyfit(x, values, 1)[0]
				
				# Déterminer la direction
				if abs(slope) < 0.001:
					trends[key] = "stable"
				elif slope > 0:
					trends[key] = "increasing"
				else:
					trends[key] = "decreasing"
		
		# Déterminer la tendance globale
		significant_keys = ["rainIntensity", "snowIntensity", "fogIntensity", "windStrength"]
		active_conditions = []
		
		for key in significant_keys:
			if self.currentConditions.get(key, 0) > 0.3:
				active_conditions.append(key)
		
		if not active_conditions:
			# Conditions claires
			if any(trends.get(key) == "increasing" for key in significant_keys):
				return {"trend": "deteriorating", "direction": "worsening"}
			else:
				return {"trend": "stable", "direction": "fair"}
		else:
			# Conditions actives
			worsening = sum(1 for key in active_conditions if trends.get(key) == "increasing")
			improving = sum(1 for key in active_conditions if trends.get(key) == "decreasing")
			
			if worsening > improving:
				return {"trend": "worsening", "direction": "deteriorating"}
			elif improving > worsening:
				return {"trend": "improving", "direction": "clearing"}
			else:
				return {"trend": "stable", "direction": "mixed"}
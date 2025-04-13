#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour la gestion des configurations du système.
"""

import os
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
	"""
	Gestionnaire de configuration pour le projet NaviLearn.
	
	Cette classe est responsable de:
	- Charger les configurations depuis un fichier YAML
	- Fournir des valeurs par défaut pour les paramètres manquants
	- Valider les paramètres de configuration
	"""
	
	def __init__(self, configPath: str) -> None:
		"""
		Initialise le gestionnaire de configuration.
		
		Args:
			configPath (str): Chemin vers le fichier de configuration YAML
		"""
		self.configPath = configPath
		self.config = None
		
		# Configuration par défaut
		self.defaultConfig = {
			"environment": {
				"width": 1000,
				"height": 1000,
				"obstacleCount": 30,
				"resourceCount": 15,
				"weatherEnabled": True,
				"terrainComplexity": 0.5
			},
			"agents": {
				"count": 10,
				"sensorRange": 150,
				"communicationRange": 200,
				"maxSpeed": 5.0,
				"maxTurnRate": 0.4,
				"memorySize": 1000
			},
			"reinforcement_learning": {
				"algorithm": "dqn",
				"stateDimension": 32,
				"actionDimension": 9,
				"learningRate": 0.001,
				"gamma": 0.99,
				"epsilon": 1.0,
				"epsilonDecay": 0.995,
				"minEpsilon": 0.01,
				"batchSize": 64,
				"bufferSize": 100000,
				"prioritizedExperience": True,
				"targetUpdateFrequency": 10,
				
				# Paramètres spécifiques à PPO
				"ppoEpsilon": 0.2,
				"valueCoefficient": 0.5,
				"entropyCoefficient": 0.01,
				"clipRange": 0.2,
				"epochsPerUpdate": 4
			},
			"training": {
				"episodes": 100,
				"evaluationFrequency": 10,
				"saveFrequency": 20
			},
			"rendering": {
				"enabled": True,
				"windowWidth": 1280,
				"windowHeight": 720,
				"targetFPS": 60,
				"showStats": True,
				"showPaths": True,
				"showSensors": True,
				"showCommunication": True,
				"use3D": True,
				"graphicalLevel": 2
			},
			"logging": {
				"level": "INFO",
				"saveStats": True,
				"statsFile": "stats/stats.csv"
			},
			"profiling": {
				"enabled": False,
				"sampleInterval": 100
			}
		}
	
	def loadConfig(self) -> Dict[str, Any]:
		"""
		Charge la configuration depuis le fichier YAML.
		
		Returns:
			Dict[str, Any]: Configuration chargée avec valeurs par défaut pour les clés manquantes
		"""
		# Vérifier si le fichier existe
		if not os.path.exists(self.configPath):
			print(f"Fichier de configuration non trouvé: {self.configPath}")
			print("Utilisation de la configuration par défaut.")
			self.config = self.defaultConfig
			return self.config
		
		# Charger le fichier YAML
		try:
			with open(self.configPath, 'r') as file:
				self.config = yaml.safe_load(file)
		except Exception as e:
			print(f"Erreur lors du chargement de la configuration: {e}")
			print("Utilisation de la configuration par défaut.")
			self.config = self.defaultConfig
			return self.config
		
		# Fusionner avec les valeurs par défaut pour les clés manquantes
		self.config = self._mergeWithDefaults(self.config, self.defaultConfig)
		
		# Valider la configuration
		self._validateConfig()
		
		return self.config
	
	def _mergeWithDefaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Fusionne la configuration chargée avec les valeurs par défaut.
		
		Args:
			config (Dict[str, Any]): Configuration chargée
			defaults (Dict[str, Any]): Configuration par défaut
			
		Returns:
			Dict[str, Any]: Configuration fusionnée
		"""
		result = defaults.copy()
		
		# Si config est None, retourner les valeurs par défaut
		if config is None:
			return result
		
		# Parcourir récursivement les clés
		for key, value in config.items():
			if key in result and isinstance(value, dict) and isinstance(result[key], dict):
				# Fusionner récursivement les sous-dictionnaires
				result[key] = self._mergeWithDefaults(value, result[key])
			else:
				# Remplacer directement la valeur
				result[key] = value
		
		return result
	
	def _validateConfig(self) -> None:
		"""
		Valide les paramètres de configuration.
		
		Vérifie que les paramètres essentiels ont des valeurs valides
		et avertit en cas de valeurs potentiellement problématiques.
		"""
		# Vérifier les dimensions de l'environnement
		if self.config["environment"]["width"] <= 0 or self.config["environment"]["height"] <= 0:
			print("AVERTISSEMENT: Les dimensions de l'environnement doivent être positives.")
			self.config["environment"]["width"] = max(1, self.config["environment"]["width"])
			self.config["environment"]["height"] = max(1, self.config["environment"]["height"])
		
		# Vérifier le nombre d'agents
		if self.config["agents"]["count"] <= 0:
			print("AVERTISSEMENT: Le nombre d'agents doit être positif.")
			self.config["agents"]["count"] = 1
		
		# Vérifier l'algorithme d'apprentissage
		if self.config["reinforcement_learning"]["algorithm"] not in ["dqn", "ppo"]:
			print(f"AVERTISSEMENT: Algorithme inconnu: {self.config['reinforcement_learning']['algorithm']}")
			print("Utilisation de l'algorithme par défaut: dqn")
			self.config["reinforcement_learning"]["algorithm"] = "dqn"
		
		# Vérifier les paramètres d'apprentissage
		if self.config["reinforcement_learning"]["gamma"] <= 0 or self.config["reinforcement_learning"]["gamma"] >= 1:
			print("AVERTISSEMENT: Le facteur gamma doit être compris entre 0 et 1.")
			self.config["reinforcement_learning"]["gamma"] = 0.99
		
		# Vérifier les paramètres de rendu
		if self.config["rendering"]["windowWidth"] <= 0 or self.config["rendering"]["windowHeight"] <= 0:
			print("AVERTISSEMENT: Les dimensions de la fenêtre doivent être positives.")
			self.config["rendering"]["windowWidth"] = max(640, self.config["rendering"]["windowWidth"])
			self.config["rendering"]["windowHeight"] = max(480, self.config["rendering"]["windowHeight"])
	
	def saveConfig(self, configPath: Optional[str] = None) -> bool:
		"""
		Sauvegarde la configuration actuelle dans un fichier YAML.
		
		Args:
			configPath (Optional[str]): Chemin du fichier de sortie, ou None pour utiliser le chemin actuel
			
		Returns:
			bool: True si la sauvegarde a réussi, False sinon
		"""
		if configPath is None:
			configPath = self.configPath
		
		try:
			# Créer le répertoire si nécessaire
			directory = os.path.dirname(configPath)
			if directory and not os.path.exists(directory):
				os.makedirs(directory)
			
			# Sauvegarder la configuration
			with open(configPath, 'w') as file:
				yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)
			
			return True
		except Exception as e:
			print(f"Erreur lors de la sauvegarde de la configuration: {e}")
			return False
	
	def getParam(self, section: str, key: str, default: Any = None) -> Any:
		"""
		Récupère un paramètre de configuration.
		
		Args:
			section (str): Section de la configuration
			key (str): Clé du paramètre
			default (Any): Valeur par défaut si le paramètre n'existe pas
			
		Returns:
			Any: Valeur du paramètre ou valeur par défaut
		"""
		if self.config is None:
			self.loadConfig()
		
		try:
			return self.config[section][key]
		except (KeyError, TypeError):
			return default
	
	def setParam(self, section: str, key: str, value: Any) -> None:
		"""
		Définit un paramètre de configuration.
		
		Args:
			section (str): Section de la configuration
			key (str): Clé du paramètre
			value (Any): Nouvelle valeur
		"""
		if self.config is None:
			self.loadConfig()
		
		# Créer la section si elle n'existe pas
		if section not in self.config:
			self.config[section] = {}
		
		# Définir la valeur
		self.config[section][key] = value
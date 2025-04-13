#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module implémentant le réseau de valeur pour l'apprentissage par renforcement.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class ValueNetwork(nn.Module):
	"""
	Réseau de valeur pour estimer la valeur d'état dans les algorithmes d'apprentissage par renforcement.
	
	Ce réseau prend un vecteur d'état en entrée et produit une estimation de la valeur (V-function).
	"""
	
	def __init__(
		self, 
		stateDimension: int,
		hiddenLayers: List[int] = [128, 64],
		activationFunction: str = "relu",
		useLayerNorm: bool = False,
		dropoutRate: float = 0.0,
		initStd: float = 0.1
	) -> None:
		"""
		Initialise le réseau de valeur.
		
		Args:
			stateDimension (int): Dimension du vecteur d'état
			hiddenLayers (List[int]): Nombre de neurones dans chaque couche cachée
			activationFunction (str): Fonction d'activation pour les couches cachées
			useLayerNorm (bool): Utiliser la normalisation des couches
			dropoutRate (float): Taux de dropout (0-1)
			initStd (float): Écart-type pour l'initialisation des poids
		"""
		super(ValueNetwork, self).__init__()
		
		self.stateDimension = stateDimension
		self.hiddenLayers = hiddenLayers
		self.activationFunction = activationFunction
		self.useLayerNorm = useLayerNorm
		self.dropoutRate = dropoutRate
		
		# Construire le réseau
		layers = []
		
		# Couche d'entrée
		inputDim = stateDimension
		
		# Couches cachées
		for hiddenDim in hiddenLayers:
			layers.append(nn.Linear(inputDim, hiddenDim))
			
			if useLayerNorm:
				layers.append(nn.LayerNorm(hiddenDim))
				
			# Fonction d'activation
			if activationFunction == "relu":
				layers.append(nn.ReLU())
			elif activationFunction == "tanh":
				layers.append(nn.Tanh())
			elif activationFunction == "leaky_relu":
				layers.append(nn.LeakyReLU(0.1))
			
			# Dropout si spécifié
			if dropoutRate > 0:
				layers.append(nn.Dropout(dropoutRate))
				
			inputDim = hiddenDim
		
		# Couche de sortie - valeur scalaire
		self.featureExtractor = nn.Sequential(*layers)
		self.valueHead = nn.Linear(inputDim, 1)
		
		# Initialisation des poids
		self._initializeWeights(initStd)
	
	def _initializeWeights(self, std: float = 0.1) -> None:
		"""
		Initialise les poids du réseau.
		
		Args:
			std (float): Écart-type pour l'initialisation
		"""
		# Initialisation de Xavier/Glorot pour les couches internes
		for layer in self.featureExtractor:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_normal_(layer.weight)
				nn.init.zeros_(layer.bias)
		
		# Initialisation spécifique pour la couche de sortie
		nn.init.normal_(self.valueHead.weight, mean=0.0, std=std)
		nn.init.zeros_(self.valueHead.bias)
	
	def forward(self, state: torch.Tensor) -> torch.Tensor:
		"""
		Propage l'état à travers le réseau.
		
		Args:
			state (torch.Tensor): Vecteur d'état
			
		Returns:
			torch.Tensor: Estimation de la valeur
		"""
		# Extraire les caractéristiques
		features = self.featureExtractor(state)
		
		# Calculer la valeur
		value = self.valueHead(features)
		
		return value.squeeze(-1)  # Supprimer la dimension supplémentaire
	
	def saveModel(self, path: str) -> None:
		"""
		Sauvegarde le modèle dans un fichier.
		
		Args:
			path (str): Chemin du fichier de sauvegarde
		"""
		torch.save({
			'state_dict': self.state_dict(),
			'state_dimension': self.stateDimension,
			'hidden_layers': self.hiddenLayers,
			'activation_function': self.activationFunction,
			'use_layer_norm': self.useLayerNorm,
			'dropout_rate': self.dropoutRate
		}, path)
	
	@classmethod
	def loadModel(cls, path: str, device: torch.device = torch.device('cpu')) -> 'ValueNetwork':
		"""
		Charge un modèle depuis un fichier.
		
		Args:
			path (str): Chemin du fichier de sauvegarde
			device (torch.device): Périphérique sur lequel charger le modèle
			
		Returns:
			ValueNetwork: Instance du modèle chargé
		"""
		checkpoint = torch.load(path, map_location=device)
		
		model = cls(
			stateDimension=checkpoint['state_dimension'],
			hiddenLayers=checkpoint['hidden_layers'],
			activationFunction=checkpoint['activation_function'],
			useLayerNorm=checkpoint['use_layer_norm'],
			dropoutRate=checkpoint['dropout_rate']
		)
		
		model.load_state_dict(checkpoint['state_dict'])
		model.to(device)
		
		return model


class QNetwork(nn.Module):
	"""
	Réseau Q pour estimer les valeurs d'action dans les algorithmes d'apprentissage par renforcement.
	
	Ce réseau prend un vecteur d'état en entrée et produit des estimations de valeur pour chaque action
	(Q-function).
	"""
	
	def __init__(
		self, 
		stateDimension: int,
		actionDimension: int,
		hiddenLayers: List[int] = [128, 64],
		activationFunction: str = "relu",
		useLayerNorm: bool = False,
		dueling: bool = False,
		initStd: float = 0.1
	) -> None:
		"""
		Initialise le réseau Q.
		
		Args:
			stateDimension (int): Dimension du vecteur d'état
			actionDimension (int): Dimension du vecteur d'action (nombre d'actions discrètes)
			hiddenLayers (List[int]): Nombre de neurones dans chaque couche cachée
			activationFunction (str): Fonction d'activation pour les couches cachées
			useLayerNorm (bool): Utiliser la normalisation des couches
			dueling (bool): Utiliser l'architecture duel (avantage + valeur)
			initStd (float): Écart-type pour l'initialisation des poids
		"""
		super(QNetwork, self).__init__()
		
		self.stateDimension = stateDimension
		self.actionDimension = actionDimension
		self.hiddenLayers = hiddenLayers
		self.activationFunction = activationFunction
		self.useLayerNorm = useLayerNorm
		self.dueling = dueling
		
		# Construire le réseau
		layers = []
		
		# Couche d'entrée
		inputDim = stateDimension
		
		# Couches cachées
		for hiddenDim in hiddenLayers:
			layers.append(nn.Linear(inputDim, hiddenDim))
			
			if useLayerNorm:
				layers.append(nn.LayerNorm(hiddenDim))
				
			# Fonction d'activation
			if activationFunction == "relu":
				layers.append(nn.ReLU())
			elif activationFunction == "tanh":
				layers.append(nn.Tanh())
			elif activationFunction == "leaky_relu":
				layers.append(nn.LeakyReLU(0.1))
				
			inputDim = hiddenDim
		
		# Extraire les caractéristiques communes
		self.featureExtractor = nn.Sequential(*layers)
		
		# Architecture standard ou duel
		if dueling:
			# Architecture duel: avantage + valeur
			self.valueHead = nn.Linear(inputDim, 1)
			self.advantageHead = nn.Linear(inputDim, actionDimension)
		else:
			# Architecture standard: Q-values directes
			self.qHead = nn.Linear(inputDim, actionDimension)
		
		# Initialisation des poids
		self._initializeWeights(initStd)
	
	def _initializeWeights(self, std: float = 0.1) -> None:
		"""
		Initialise les poids du réseau.
		
		Args:
			std (float): Écart-type pour l'initialisation
		"""
		# Initialisation de Xavier/Glorot pour les couches internes
		for layer in self.featureExtractor:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_normal_(layer.weight)
				nn.init.zeros_(layer.bias)
		
		# Initialisation spécifique pour les couches de sortie
		if self.dueling:
			nn.init.normal_(self.valueHead.weight, mean=0.0, std=std)
			nn.init.zeros_(self.valueHead.bias)
			
			nn.init.normal_(self.advantageHead.weight, mean=0.0, std=std)
			nn.init.zeros_(self.advantageHead.bias)
		else:
			nn.init.normal_(self.qHead.weight, mean=0.0, std=std)
			nn.init.zeros_(self.qHead.bias)
	
	def forward(self, state: torch.Tensor) -> torch.Tensor:
		"""
		Propage l'état à travers le réseau.
		
		Args:
			state (torch.Tensor): Vecteur d'état
			
		Returns:
			torch.Tensor: Estimations Q pour chaque action
		"""
		# Extraire les caractéristiques
		features = self.featureExtractor(state)
		
		if self.dueling:
			# Architecture duel
			value = self.valueHead(features)
			advantage = self.advantageHead(features)
			
			# Combiner valeur et avantage (trick de moyenne)
			qValues = value + (advantage - advantage.mean(dim=-1, keepdim=True))
		else:
			# Architecture standard
			qValues = self.qHead(features)
		
		return qValues
	
	def selectAction(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
		"""
		Sélectionne une action selon la politique epsilon-greedy.
		
		Args:
			state (torch.Tensor): Vecteur d'état
			epsilon (float): Probabilité d'exploration (0-1)
			
		Returns:
			int: Action sélectionnée
		"""
		# Exploration aléatoire
		if np.random.random() < epsilon:
			return np.random.randint(0, self.actionDimension)
		
		# Exploitation
		with torch.no_grad():
			qValues = self.forward(state)
			return int(torch.argmax(qValues).item())
	
	def saveModel(self, path: str) -> None:
		"""
		Sauvegarde le modèle dans un fichier.
		
		Args:
			path (str): Chemin du fichier de sauvegarde
		"""
		torch.save({
			'state_dict': self.state_dict(),
			'state_dimension': self.stateDimension,
			'action_dimension': self.actionDimension,
			'hidden_layers': self.hiddenLayers,
			'activation_function': self.activationFunction,
			'use_layer_norm': self.useLayerNorm,
			'dueling': self.dueling
		}, path)
	
	@classmethod
	def loadModel(cls, path: str, device: torch.device = torch.device('cpu')) -> 'QNetwork':
		"""
		Charge un modèle depuis un fichier.
		
		Args:
			path (str): Chemin du fichier de sauvegarde
			device (torch.device): Périphérique sur lequel charger le modèle
			
		Returns:
			QNetwork: Instance du modèle chargé
		"""
		checkpoint = torch.load(path, map_location=device)
		
		model = cls(
			stateDimension=checkpoint['state_dimension'],
			actionDimension=checkpoint['action_dimension'],
			hiddenLayers=checkpoint['hidden_layers'],
			activationFunction=checkpoint['activation_function'],
			useLayerNorm=checkpoint['use_layer_norm'],
			dueling=checkpoint['dueling']
		)
		
		model.load_state_dict(checkpoint['state_dict'])
		model.to(device)
		
		return model
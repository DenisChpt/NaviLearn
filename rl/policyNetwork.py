#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module implémentant le réseau de politique pour l'apprentissage par renforcement.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class PolicyNetwork(nn.Module):
	"""
	Réseau de politique pour sélectionner des actions dans l'espace d'action discret.
	
	Ce réseau neuronal prend un vecteur d'état en entrée et produit une distribution
	de probabilité sur les actions possibles.
	"""
	
	def __init__(
		self, 
		stateDimension: int,
		actionDimension: int,
		hiddenLayers: List[int] = [128, 64],
		activationFunction: str = "relu",
		outputActivation: str = "softmax",
		initStd: float = 0.1
	) -> None:
		"""
		Initialise le réseau de politique.
		
		Args:
			stateDimension (int): Dimension du vecteur d'état
			actionDimension (int): Dimension du vecteur d'action (nombre d'actions discrètes)
			hiddenLayers (List[int]): Nombre de neurones dans chaque couche cachée
			activationFunction (str): Fonction d'activation pour les couches cachées
			outputActivation (str): Fonction d'activation pour la couche de sortie
			initStd (float): Écart-type pour l'initialisation des poids
		"""
		super(PolicyNetwork, self).__init__()
		
		self.stateDimension = stateDimension
		self.actionDimension = actionDimension
		self.hiddenLayers = hiddenLayers
		self.activationFunction = activationFunction
		self.outputActivation = outputActivation
		
		# Construire le réseau
		layers = []
		
		# Couche d'entrée
		inputDim = stateDimension
		
		# Couches cachées
		for hiddenDim in hiddenLayers:
			layers.append(nn.Linear(inputDim, hiddenDim))
			if activationFunction == "relu":
				layers.append(nn.ReLU())
			elif activationFunction == "tanh":
				layers.append(nn.Tanh())
			elif activationFunction == "leaky_relu":
				layers.append(nn.LeakyReLU(0.1))
			inputDim = hiddenDim
		
		# Couche de sortie
		self.featureExtractor = nn.Sequential(*layers)
		self.outputLayer = nn.Linear(inputDim, actionDimension)
		
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
		nn.init.normal_(self.outputLayer.weight, mean=0.0, std=std)
		nn.init.zeros_(self.outputLayer.bias)
	
	def forward(self, state: torch.Tensor) -> torch.Tensor:
		"""
		Propage l'état à travers le réseau.
		
		Args:
			state (torch.Tensor): Vecteur d'état
			
		Returns:
			torch.Tensor: Distribution de probabilité sur les actions
		"""
		# Extraire les caractéristiques
		features = self.featureExtractor(state)
		
		# Calculer les logits
		logits = self.outputLayer(features)
		
		# Appliquer la fonction d'activation de sortie
		if self.outputActivation == "softmax":
			return F.softmax(logits, dim=-1)
		elif self.outputActivation == "tanh":
			return torch.tanh(logits)
		else:
			return logits
	
	def selectAction(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
		"""
		Sélectionne une action à partir de l'état actuel.
		
		Args:
			state (torch.Tensor): Vecteur d'état
			deterministic (bool): Si True, sélectionne l'action la plus probable,
								  sinon échantillonne selon la distribution
			
		Returns:
			Tuple[int, torch.Tensor]: Action sélectionnée et distribution de probabilité
		"""
		# Obtenir la distribution de probabilité
		with torch.no_grad():
			probs = self.forward(state)
		
		# Sélectionner l'action
		if deterministic:
			# Sélectionner l'action avec la probabilité la plus élevée
			action = int(torch.argmax(probs, dim=-1).item())
		else:
			# Échantillonner selon la distribution
			action = int(torch.multinomial(probs, 1).item())
		
		return action, probs
	
	def evaluateActions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Évalue les log-probabilités des actions et l'entropie.
		
		Args:
			states (torch.Tensor): Batch de vecteurs d'état
			actions (torch.Tensor): Batch d'actions
			
		Returns:
			Tuple[torch.Tensor, torch.Tensor]: Log-probabilités et entropie
		"""
		# Obtenir les distributions de probabilité
		probs = self.forward(states)
		
		# Calculer les log-probabilités des actions
		dist = torch.distributions.Categorical(probs)
		logProbs = dist.log_prob(actions)
		
		# Calculer l'entropie
		entropy = dist.entropy()
		
		return logProbs, entropy
	
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
			'output_activation': self.outputActivation
		}, path)
	
	@classmethod
	def loadModel(cls, path: str, device: torch.device = torch.device('cpu')) -> 'PolicyNetwork':
		"""
		Charge un modèle depuis un fichier.
		
		Args:
			path (str): Chemin du fichier de sauvegarde
			device (torch.device): Périphérique sur lequel charger le modèle
			
		Returns:
			PolicyNetwork: Instance du modèle chargé
		"""
		checkpoint = torch.load(path, map_location=device)
		
		model = cls(
			stateDimension=checkpoint['state_dimension'],
			actionDimension=checkpoint['action_dimension'],
			hiddenLayers=checkpoint['hidden_layers'],
			activationFunction=checkpoint['activation_function'],
			outputActivation=checkpoint['output_activation']
		)
		
		model.load_state_dict(checkpoint['state_dict'])
		model.to(device)
		
		return model


class ContinuousPolicyNetwork(nn.Module):
	"""
	Réseau de politique pour sélectionner des actions dans un espace d'action continu.
	
	Ce réseau neuronal produit une distribution gaussienne sur les actions continues.
	"""
	
	def __init__(
		self, 
		stateDimension: int,
		actionDimension: int,
		hiddenLayers: List[int] = [128, 64],
		activationFunction: str = "tanh",
		minLogStd: float = -20,
		maxLogStd: float = 2,
		initStd: float = 0.1
	) -> None:
		"""
		Initialise le réseau de politique continue.
		
		Args:
			stateDimension (int): Dimension du vecteur d'état
			actionDimension (int): Dimension du vecteur d'action continu
			hiddenLayers (List[int]): Nombre de neurones dans chaque couche cachée
			activationFunction (str): Fonction d'activation pour les couches cachées
			minLogStd (float): Logarithme minimal de l'écart-type
			maxLogStd (float): Logarithme maximal de l'écart-type
			initStd (float): Écart-type pour l'initialisation des poids
		"""
		super(ContinuousPolicyNetwork, self).__init__()
		
		self.stateDimension = stateDimension
		self.actionDimension = actionDimension
		self.hiddenLayers = hiddenLayers
		self.activationFunction = activationFunction
		self.minLogStd = minLogStd
		self.maxLogStd = maxLogStd
		
		# Construire le réseau
		layers = []
		
		# Couche d'entrée
		inputDim = stateDimension
		
		# Couches cachées
		for hiddenDim in hiddenLayers:
			layers.append(nn.Linear(inputDim, hiddenDim))
			if activationFunction == "relu":
				layers.append(nn.ReLU())
			elif activationFunction == "tanh":
				layers.append(nn.Tanh())
			elif activationFunction == "leaky_relu":
				layers.append(nn.LeakyReLU(0.1))
			inputDim = hiddenDim
		
		# Extraction de caractéristiques communes
		self.featureExtractor = nn.Sequential(*layers)
		
		# Couche de sortie pour la moyenne
		self.meanLayer = nn.Linear(inputDim, actionDimension)
		
		# Couche de sortie pour l'écart-type (logarithme)
		self.logStdLayer = nn.Linear(inputDim, actionDimension)
		
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
		nn.init.xavier_normal_(self.meanLayer.weight)
		nn.init.zeros_(self.meanLayer.bias)
		
		nn.init.xavier_normal_(self.logStdLayer.weight)
		nn.init.zeros_(self.logStdLayer.bias)
	
	def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Propage l'état à travers le réseau.
		
		Args:
			state (torch.Tensor): Vecteur d'état
			
		Returns:
			Tuple[torch.Tensor, torch.Tensor]: Moyenne et écart-type de la distribution
		"""
		# Extraire les caractéristiques
		features = self.featureExtractor(state)
		
		# Calculer la moyenne et l'écart-type
		mean = self.meanLayer(features)
		logStd = self.logStdLayer(features)
		
		# Limiter l'écart-type pour la stabilité
		logStd = torch.clamp(logStd, self.minLogStd, self.maxLogStd)
		std = torch.exp(logStd)
		
		return mean, std
	
	def selectAction(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Tuple]:
		"""
		Sélectionne une action à partir de l'état actuel.
		
		Args:
			state (torch.Tensor): Vecteur d'état
			deterministic (bool): Si True, retourne la moyenne, sinon échantillonne
			
		Returns:
			Tuple[torch.Tensor, Tuple]: Action sélectionnée et paramètres de distribution
		"""
		# Obtenir les paramètres de distribution
		with torch.no_grad():
			mean, std = self.forward(state)
		
		# Sélectionner l'action
		if deterministic:
			# Retourner la moyenne directement
			action = mean
		else:
			# Échantillonner selon la distribution normale
			normal = torch.distributions.Normal(mean, std)
			action = normal.sample()
		
		return action, (mean, std)
	
	def evaluateActions(
		self, 
		states: torch.Tensor, 
		actions: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Évalue les log-probabilités des actions et l'entropie.
		
		Args:
			states (torch.Tensor): Batch de vecteurs d'état
			actions (torch.Tensor): Batch d'actions
			
		Returns:
			Tuple[torch.Tensor, torch.Tensor]: Log-probabilités et entropie
		"""
		# Obtenir les paramètres de distribution
		mean, std = self.forward(states)
		
		# Créer la distribution
		dist = torch.distributions.Normal(mean, std)
		
		# Calculer les log-probabilités des actions
		logProbs = dist.log_prob(actions).sum(dim=-1)
		
		# Calculer l'entropie
		entropy = dist.entropy().sum(dim=-1)
		
		return logProbs, entropy
	
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
			'min_log_std': self.minLogStd,
			'max_log_std': self.maxLogStd
		}, path)
	
	@classmethod
	def loadModel(cls, path: str, device: torch.device = torch.device('cpu')) -> 'ContinuousPolicyNetwork':
		"""
		Charge un modèle depuis un fichier.
		
		Args:
			path (str): Chemin du fichier de sauvegarde
			device (torch.device): Périphérique sur lequel charger le modèle
			
		Returns:
			ContinuousPolicyNetwork: Instance du modèle chargé
		"""
		checkpoint = torch.load(path, map_location=device)
		
		model = cls(
			stateDimension=checkpoint['state_dimension'],
			actionDimension=checkpoint['action_dimension'],
			hiddenLayers=checkpoint['hidden_layers'],
			activationFunction=checkpoint['activation_function'],
			minLogStd=checkpoint['min_log_std'],
			maxLogStd=checkpoint['max_log_std']
		)
		
		model.load_state_dict(checkpoint['state_dict'])
		model.to(device)
		
		return model
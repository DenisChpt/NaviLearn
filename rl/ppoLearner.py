#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module implémentant l'algorithme PPO (Proximal Policy Optimization) pour l'apprentissage par renforcement.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Deque
from collections import deque, namedtuple

from rl.policyNetwork import PolicyNetwork
from rl.valueNetwork import ValueNetwork
from agents.collectiveAgent import CollectiveAgent


# Structure pour stocker les transitions (expériences)
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob'])


class PPOMemory:
	"""
	Mémoire pour stocker les trajectoires pour l'algorithme PPO.
	"""
	
	def __init__(self, batchSize: int = 64):
		"""
		Initialise la mémoire PPO.
		
		Args:
			batchSize (int): Taille des batchs d'apprentissage
		"""
		self.states = []
		self.actions = []
		self.rewards = []
		self.nextStates = []
		self.dones = []
		self.logProbs = []
		self.batchSize = batchSize
	
	def add(
		self, 
		state: np.ndarray, 
		action: int, 
		reward: float, 
		nextState: np.ndarray, 
		done: bool, 
		logProb: float
	) -> None:
		"""
		Ajoute une transition à la mémoire.
		
		Args:
			state (np.ndarray): État
			action (int): Action
			reward (float): Récompense
			nextState (np.ndarray): État suivant
			done (bool): Indique si l'épisode est terminé
			logProb (float): Log-probabilité de l'action
		"""
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.nextStates.append(nextState)
		self.dones.append(done)
		self.logProbs.append(logProb)
	
	def clear(self) -> None:
		"""Vide la mémoire."""
		self.states = []
		self.actions = []
		self.rewards = []
		self.nextStates = []
		self.dones = []
		self.logProbs = []
	
	def __len__(self) -> int:
		"""Retourne la taille de la mémoire."""
		return len(self.states)
	
	def getBatches(self, device: torch.device) -> List:
		"""
		Génère des batchs pour l'apprentissage.
		
		Args:
			device (torch.device): Périphérique pour les tenseurs
			
		Returns:
			List: Liste des batchs
		"""
		batchStart = np.arange(0, len(self.states), self.batchSize)
		indices = np.arange(len(self.states), dtype=np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batchSize] for i in batchStart]
		
		return [
			(
				torch.tensor(np.array(self.states)[batch], dtype=torch.float32).to(device),
				torch.tensor(np.array(self.actions)[batch]).to(device),
				torch.tensor(np.array(self.rewards)[batch], dtype=torch.float32).to(device),
				torch.tensor(np.array(self.nextStates)[batch], dtype=torch.float32).to(device),
				torch.tensor(np.array(self.dones)[batch], dtype=torch.float32).to(device),
				torch.tensor(np.array(self.logProbs)[batch], dtype=torch.float32).to(device)
			)
			for batch in batches
		]


class PPOLearner:
	"""
	Implémentation de l'algorithme PPO (Proximal Policy Optimization).
	
	PPO est un algorithme de politique sur mesure qui utilise un clipping pour limiter
	les mises à jour de politique, offrant un bon équilibre entre facilité d'utilisation,
	échantillonnage et performances.
	"""
	
	def __init__(
		self,
		agents: List[CollectiveAgent],
		stateDimension: int,
		actionDimension: int,
		learningRate: float = 0.0003,
		gamma: float = 0.99,
		epsilon: float = 0.2,  # Paramètre de clipping pour PPO
		valueCoefficient: float = 0.5,  # Coefficient pour la perte de valeur
		entropyCoefficient: float = 0.01,  # Coefficient pour le terme d'entropie
		clipRange: float = 0.2,  # Plage de clipping pour la fonction objectif
		gaeParameter: float = 0.95,  # Paramètre lambda pour GAE (Generalized Advantage Estimation)
		batchSize: int = 64,
		memorySizeMultiplier: int = 10,  # Multiplicateur pour déterminer la taille de la mémoire
		normAdvantages: bool = True,  # Normaliser les avantages
		epochsPerUpdate: int = 4,  # Nombre d'époques d'apprentissage par mise à jour
		miniBatchSize: int = 64,  # Taille des mini-batchs pour l'apprentissage
		maxGradNorm: float = 0.5,  # Norme maximale pour le clipping de gradient
		updateFrequency: int = 2048,  # Nombre d'étapes avant mise à jour
		device: torch.device = None
	) -> None:
		"""
		Initialise l'algorithme PPO.
		
		Args:
			agents (List[CollectiveAgent]): Liste des agents à entraîner
			stateDimension (int): Dimension du vecteur d'état
			actionDimension (int): Dimension du vecteur d'action (nombre d'actions discrètes)
			learningRate (float): Taux d'apprentissage
			gamma (float): Facteur d'actualisation pour les récompenses futures
			epsilon (float): Paramètre de clipping pour PPO
			valueCoefficient (float): Coefficient pour la perte de valeur
			entropyCoefficient (float): Coefficient pour le terme d'entropie
			clipRange (float): Plage de clipping pour la fonction objectif
			gaeParameter (float): Paramètre lambda pour GAE
			batchSize (int): Taille des batchs d'apprentissage
			memorySizeMultiplier (int): Multiplicateur pour déterminer la taille de la mémoire
			normAdvantages (bool): Normaliser les avantages
			epochsPerUpdate (int): Nombre d'époques d'apprentissage par mise à jour
			miniBatchSize (int): Taille des mini-batchs pour l'apprentissage
			maxGradNorm (float): Norme maximale pour le clipping de gradient
			updateFrequency (int): Nombre d'étapes avant mise à jour
			device (torch.device): Périphérique d'exécution (CPU/GPU)
		"""
		# Configuration
		self.agents = agents
		self.stateDimension = stateDimension
		self.actionDimension = actionDimension
		self.learningRate = learningRate
		self.gamma = gamma
		self.epsilon = epsilon
		self.valueCoefficient = valueCoefficient
		self.entropyCoefficient = entropyCoefficient
		self.clipRange = clipRange
		self.gaeParameter = gaeParameter
		self.batchSize = batchSize
		self.normAdvantages = normAdvantages
		self.epochsPerUpdate = epochsPerUpdate
		self.miniBatchSize = miniBatchSize
		self.maxGradNorm = maxGradNorm
		self.updateFrequency = updateFrequency
		
		# Compteurs
		self.trainingSteps = 0
		self.episodeCount = 0
		self.totalSteps = 0
		self.updateCount = 0
		
		# Initialiser le périphérique
		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = device
			
		print(f"PPO utilisant le périphérique: {self.device}")
		
		# Initialiser les réseaux
		self.policyNetwork = PolicyNetwork(
			stateDimension=stateDimension,
			actionDimension=actionDimension,
			hiddenLayers=[128, 64],
			activationFunction="tanh"
		).to(self.device)
		
		self.valueNetwork = ValueNetwork(
			stateDimension=stateDimension,
			hiddenLayers=[128, 64],
			activationFunction="tanh"
		).to(self.device)
		
		# Initialiser les optimiseurs
		self.policyOptimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate)
		self.valueOptimizer = optim.Adam(self.valueNetwork.parameters(), lr=learningRate)
		
		# Initialiser la mémoire
		memorySize = batchSize * memorySizeMultiplier
		self.memory = PPOMemory(batchSize=miniBatchSize)
		
		# Suivi des récompenses pour la normalisation
		self.rewardScaler = RunningMeanStd(shape=())
		self.rewardHistory = deque(maxlen=100)
		
		# Métriques de performance
		self.valueLosses = deque(maxlen=100)
		self.policyLosses = deque(maxlen=100)
		self.entropyLosses = deque(maxlen=100)
		self.advantages = deque(maxlen=100)
		self.ratios = deque(maxlen=100)
	
	def getAction(self, agent: CollectiveAgent, state: np.ndarray) -> Tuple[int, float]:
		"""
		Choisit une action selon la politique actuelle.
		
		Args:
			agent (CollectiveAgent): Agent pour lequel choisir l'action
			state (np.ndarray): État actuel
			
		Returns:
			Tuple[int, float]: Action choisie et sa log-probabilité
		"""
		# Convertir l'état en tensor
		stateTensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
		
		# Obtenir la distribution de probabilité
		self.policyNetwork.eval()
		with torch.no_grad():
			action, probs = self.policyNetwork.selectAction(stateTensor)
			dist = Categorical(probs)
			logProb = dist.log_prob(torch.tensor([action], device=self.device)).item()
		self.policyNetwork.train()
		
		return action, logProb
	
	def storeExperience(
		self, 
		agent: CollectiveAgent, 
		state: np.ndarray, 
		action: int, 
		reward: float, 
		nextState: np.ndarray, 
		done: bool,
		logProb: float
	) -> None:
		"""
		Stocke une expérience dans la mémoire PPO.
		
		Args:
			agent (CollectiveAgent): Agent qui a généré l'expérience
			state (np.ndarray): État
			action (int): Action effectuée
			reward (float): Récompense reçue
			nextState (np.ndarray): État suivant
			done (bool): Indique si l'état suivant est terminal
			logProb (float): Log-probabilité de l'action
		"""
		# Normaliser la récompense
		self.rewardScaler.update(np.array([reward]))
		normalizedReward = reward / np.sqrt(self.rewardScaler.var + 1e-8)
		
		# Stocker l'expérience
		self.memory.add(state, action, normalizedReward, nextState, done, logProb)
		
		# Stocker la récompense brute pour le suivi
		self.rewardHistory.append(reward)
		
		# Incrémenter le compteur d'étapes
		self.totalSteps += 1
	
	def learn(self) -> Optional[Dict[str, float]]:
		"""
		Effectue une mise à jour si le moment est venu.
		
		Returns:
			Optional[Dict[str, float]]: Métriques d'apprentissage si une mise à jour a été effectuée, None sinon
		"""
		# Vérifier si c'est le moment de mettre à jour
		if len(self.memory) < self.updateFrequency:
			return None
			
		# Incrémenter le compteur de mises à jour
		self.updateCount += 1
		
		# Incrémenter le compteur d'étapes d'apprentissage
		self.trainingSteps += len(self.memory)
		
		# Calculer les avantages et les retours
		advantages, returns = self._computeAdvantagesAndReturns()
		
		# Normaliser les avantages
		if self.normAdvantages and len(advantages) > 1:
			advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
		
		# Conversion en tenseurs
		states = torch.tensor(np.array(self.memory.states), dtype=torch.float32).to(self.device)
		actions = torch.tensor(np.array(self.memory.actions)).to(self.device)
		oldLogProbs = torch.tensor(np.array(self.memory.logProbs), dtype=torch.float32).to(self.device)
		
		# Générer des batchs pour l'apprentissage
		batches = self.memory.getBatches(self.device)
		
		# Métriques pour le suivi
		totalPolicyLoss = 0
		totalValueLoss = 0
		totalEntropyLoss = 0
		totalClippedRatio = 0
		batchCount = 0
		
		# Effectuer plusieurs époques d'apprentissage
		for _ in range(self.epochsPerUpdate):
			for stateBatch, actionBatch, _, _, _, oldLogProbBatch in batches:
				batchCount += 1
				
				# Indices pour récupérer les avantages et retours correspondants
				batchIndices = np.arange(len(stateBatch))
				advantageBatch = advantages[batchIndices]
				returnBatch = returns[batchIndices]
				
				# Mise à jour de la politique
				# 1. Obtenir les distributions actuelles
				self.policyNetwork.train()
				probs = self.policyNetwork(stateBatch)
				dist = Categorical(probs)
				currLogProbs = dist.log_prob(actionBatch)
				entropy = dist.entropy().mean()
				
				# 2. Calculer les ratios de probabilité
				ratios = torch.exp(currLogProbs - oldLogProbBatch)
				self.ratios.extend(ratios.detach().cpu().numpy())
				
				# 3. Calculer les composantes de la perte surrogate
				surr1 = ratios * advantageBatch
				surr2 = torch.clamp(ratios, 1-self.clipRange, 1+self.clipRange) * advantageBatch
				
				# 4. Perte de politique
				policyLoss = -torch.min(surr1, surr2).mean()
				
				# 5. Compter les ratios clippés
				clippedCount = (ratios < 1-self.clipRange).sum() + (ratios > 1+self.clipRange).sum()
				totalClippedRatio += clippedCount.item() / len(ratios)
				
				# 6. Ajouter l'entropie pour encourager l'exploration
				entropyLoss = -entropy * self.entropyCoefficient
				
				# 7. Perte totale de politique
				totalLoss = policyLoss + entropyLoss
				
				# 8. Optimisation de la politique
				self.policyOptimizer.zero_grad()
				totalLoss.backward()
				torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.maxGradNorm)
				self.policyOptimizer.step()
				
				# Mise à jour du réseau de valeur
				# 1. Obtenir les valeurs d'état prédites
				self.valueNetwork.train()
				valuesPred = self.valueNetwork(stateBatch)
				
				# 2. Calculer la perte de valeur (MSE)
				valueLoss = ((valuesPred - returnBatch) ** 2).mean() * self.valueCoefficient
				
				# 3. Optimisation du réseau de valeur
				self.valueOptimizer.zero_grad()
				valueLoss.backward()
				torch.nn.utils.clip_grad_norm_(self.valueNetwork.parameters(), self.maxGradNorm)
				self.valueOptimizer.step()
				
				# Suivi des métriques
				totalPolicyLoss += policyLoss.item()
				totalValueLoss += valueLoss.item()
				totalEntropyLoss += entropyLoss.item()
				
				# Stocker pour les métriques
				self.policyLosses.append(policyLoss.item())
				self.valueLosses.append(valueLoss.item())
				self.entropyLosses.append(entropyLoss.item())
				self.advantages.extend(advantageBatch.detach().cpu().numpy())
		
		# Vider la mémoire après l'apprentissage
		self.memory.clear()
		
		# Calculer les moyennes pour les métriques
		avgPolicyLoss = totalPolicyLoss / batchCount
		avgValueLoss = totalValueLoss / batchCount
		avgEntropyLoss = totalEntropyLoss / batchCount
		avgClippedRatio = totalClippedRatio / batchCount
		
		# Retourner les métriques
		return {
			'policy_loss': avgPolicyLoss,
			'value_loss': avgValueLoss,
			'entropy_loss': avgEntropyLoss,
			'clipped_ratio': avgClippedRatio,
			'advantage_mean': advantages.mean().item(),
			'advantage_std': advantages.std().item(),
			'return_mean': returns.mean().item(),
			'return_std': returns.std().item()
		}
	
	def _computeAdvantagesAndReturns(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Calcule les avantages et les retours pour les expériences en mémoire.
		
		Returns:
			Tuple[torch.Tensor, torch.Tensor]: Avantages et retours calculés
		"""
		# Convertir les états et récompenses en tenseurs
		states = torch.tensor(np.array(self.memory.states), dtype=torch.float32).to(self.device)
		nextStates = torch.tensor(np.array(self.memory.nextStates), dtype=torch.float32).to(self.device)
		rewards = torch.tensor(np.array(self.memory.rewards), dtype=torch.float32).to(self.device)
		dones = torch.tensor(np.array(self.memory.dones), dtype=torch.float32).to(self.device)
		
		# Obtenir les valeurs d'état
		with torch.no_grad():
			values = self.valueNetwork(states).detach()
			nextValues = self.valueNetwork(nextStates).detach()
		
		# Calculer les retours et avantages avec GAE
		advantages = torch.zeros_like(rewards)
		returns = torch.zeros_like(rewards)
		lastGae = 0
		
		# Parcourir les expériences en ordre inverse
		for t in reversed(range(len(rewards))):
			# Pour le dernier état, ou si l'épisode est terminé
			if t == len(rewards) - 1 or dones[t]:
				nextValue = 0
			else:
				nextValue = nextValues[t]
			
			# Delta: r + gamma * V(s') - V(s)
			delta = rewards[t] + self.gamma * nextValue * (1 - dones[t]) - values[t]
			
			# GAE: δt + (γλ)δt+1 + (γλ)²δt+2 + ...
			lastGae = delta + self.gamma * self.gaeParameter * (1 - dones[t]) * lastGae
			advantages[t] = lastGae
			
			# Retours: avantage + valeur
			returns[t] = advantages[t] + values[t]
		
		return advantages, returns
	
	def endEpisode(self) -> None:
		"""
		Effectue les opérations de fin d'épisode.
		"""
		# Incrémenter le compteur d'épisodes
		self.episodeCount += 1
	
	def saveModel(self, path: str) -> None:
		"""
		Sauvegarde le modèle dans un fichier.
		
		Args:
			path (str): Chemin du fichier de sauvegarde
		"""
		# Créer le répertoire si nécessaire
		os.makedirs(os.path.dirname(path), exist_ok=True)
		
		# Sauvegarder les réseaux et les paramètres
		torch.save({
			'policy_network_state_dict': self.policyNetwork.state_dict(),
			'value_network_state_dict': self.valueNetwork.state_dict(),
			'policy_optimizer_state_dict': self.policyOptimizer.state_dict(),
			'value_optimizer_state_dict': self.valueOptimizer.state_dict(),
			'training_steps': self.trainingSteps,
			'episode_count': self.episodeCount,
			'total_steps': self.totalSteps,
			'update_count': self.updateCount,
			'reward_scaler_mean': self.rewardScaler.mean,
			'reward_scaler_var': self.rewardScaler.var,
			'reward_scaler_count': self.rewardScaler.count,
			'config': {
				'state_dimension': self.stateDimension,
				'action_dimension': self.actionDimension,
				'learning_rate': self.learningRate,
				'gamma': self.gamma,
				'epsilon': self.epsilon,
				'value_coefficient': self.valueCoefficient,
				'entropy_coefficient': self.entropyCoefficient,
				'clip_range': self.clipRange,
				'gae_parameter': self.gaeParameter,
				'batch_size': self.batchSize,
				'norm_advantages': self.normAdvantages,
				'epochs_per_update': self.epochsPerUpdate,
				'mini_batch_size': self.miniBatchSize,
				'max_grad_norm': self.maxGradNorm,
				'update_frequency': self.updateFrequency
			}
		}, path)
		
		print(f"Modèle sauvegardé dans {path}")
	
	def loadModel(self, path: str) -> None:
		"""
		Charge un modèle depuis un fichier.
		
		Args:
			path (str): Chemin du fichier à charger
		"""
		# Vérifier que le fichier existe
		if not os.path.exists(path):
			print(f"Erreur: Le fichier {path} n'existe pas.")
			return
		
		# Charger les données
		checkpoint = torch.load(path, map_location=self.device)
		
		# Vérifier la compatibilité des dimensions
		configData = checkpoint.get('config', {})
		if (configData.get('state_dimension') != self.stateDimension or
			configData.get('action_dimension') != self.actionDimension):
			print("Avertissement: Les dimensions du modèle chargé sont différentes.")
			print(f"  Modèle: {configData.get('state_dimension')}x{configData.get('action_dimension')}")
			print(f"  Actuel: {self.stateDimension}x{self.actionDimension}")
		
		# Charger les poids des réseaux
		self.policyNetwork.load_state_dict(checkpoint['policy_network_state_dict'])
		self.valueNetwork.load_state_dict(checkpoint['value_network_state_dict'])
		
		# Charger l'état des optimiseurs
		self.policyOptimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
		self.valueOptimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
		
		# Charger les paramètres
		self.trainingSteps = checkpoint.get('training_steps', 0)
		self.episodeCount = checkpoint.get('episode_count', 0)
		self.totalSteps = checkpoint.get('total_steps', 0)
		self.updateCount = checkpoint.get('update_count', 0)
		
		# Charger l'état du normaliseur de récompenses
		mean = checkpoint.get('reward_scaler_mean', 0)
		var = checkpoint.get('reward_scaler_var', 1)
		count = checkpoint.get('reward_scaler_count', 0)
		self.rewardScaler.mean = mean
		self.rewardScaler.var = var
		self.rewardScaler.count = count
		
		print(f"Modèle chargé depuis {path}")
		print(f"  Épisodes: {self.episodeCount}, Étapes: {self.totalSteps}, Mises à jour: {self.updateCount}")
	
	def getStats(self) -> Dict[str, Any]:
		"""
		Retourne les statistiques d'apprentissage.
		
		Returns:
			Dict[str, Any]: Statistiques actuelles
		"""
		return {
			'training_steps': self.trainingSteps,
			'episode_count': self.episodeCount,
			'total_steps': self.totalSteps,
			'update_count': self.updateCount,
			'memory_size': len(self.memory),
			'reward_mean': np.mean(self.rewardHistory) if self.rewardHistory else 0,
			'reward_std': np.std(self.rewardHistory) if self.rewardHistory else 0,
			'policy_loss_mean': np.mean(self.policyLosses) if self.policyLosses else 0,
			'value_loss_mean': np.mean(self.valueLosses) if self.valueLosses else 0,
			'entropy_loss_mean': np.mean(self.entropyLosses) if self.entropyLosses else 0,
			'advantage_mean': np.mean(self.advantages) if self.advantages else 0,
			'advantage_std': np.std(self.advantages) if self.advantages else 0,
			'ratio_mean': np.mean(self.ratios) if self.ratios else 0,
			'ratio_std': np.std(self.ratios) if self.ratios else 0,
			'learning_rate': self.learningRate
		}
	
	def updateLearningRate(self, newLR: float) -> None:
		"""
		Met à jour le taux d'apprentissage des optimiseurs.
		
		Args:
			newLR (float): Nouveau taux d'apprentissage
		"""
		for param_group in self.policyOptimizer.param_groups:
			param_group['lr'] = newLR
			
		for param_group in self.valueOptimizer.param_groups:
			param_group['lr'] = newLR
		
		self.learningRate = newLR


class RunningMeanStd:
	"""
	Calcule la moyenne et l'écart-type mobiles pour la normalisation.
	"""
	
	def __init__(self, shape=()):
		"""
		Initialise le calcul de statistiques mobiles.
		
		Args:
			shape (tuple): Forme des données
		"""
		self.mean = np.zeros(shape, dtype=np.float64)
		self.var = np.ones(shape, dtype=np.float64)
		self.count = 0
	
	def update(self, x: np.ndarray) -> None:
		"""
		Met à jour les statistiques avec de nouvelles données.
		
		Args:
			x (np.ndarray): Nouvelles observations
		"""
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]
		
		self.update_from_moments(batch_mean, batch_var, batch_count)
	
	def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
		"""
		Met à jour à partir des moments statistiques.
		
		Args:
			batch_mean (np.ndarray): Moyenne du batch
			batch_var (np.ndarray): Variance du batch
			batch_count (int): Taille du batch
		"""
		delta = batch_mean - self.mean
		total_count = self.count + batch_count
		
		new_mean = self.mean + delta * batch_count / total_count
		
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
		
		self.mean = new_mean
		self.var = M2 / total_count
		self.count = total_count
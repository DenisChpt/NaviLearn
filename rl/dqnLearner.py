#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module implémentant l'algorithme DQN (Deep Q-Network) pour l'apprentissage par renforcement.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import Dict, List, Tuple, Optional, Union, Any

from rl.valueNetwork import QNetwork
from rl.replayBuffer import ReplayBuffer, PrioritizedReplayBuffer, MultiAgentReplayBuffer
from agents.collectiveAgent import CollectiveAgent


class DQNLearner:
	"""
	Implémentation de l'algorithme DQN (Deep Q-Network) avec diverses améliorations.
	
	Cette classe implémente:
	- DQN de base
	- Double DQN (DDQN)
	- Dueling Network Architecture
	- Prioritized Experience Replay (PER)
	- Multi-agent DQN
	"""
	
	def __init__(
		self,
		agents: List[CollectiveAgent],
		stateDimension: int,
		actionDimension: int,
		learningRate: float = 0.001,
		gamma: float = 0.99,
		epsilon: float = 1.0,
		epsilonDecay: float = 0.995,
		minEpsilon: float = 0.01,
		tau: float = 0.001,  # Pour la mise à jour douce du réseau cible
		batchSize: int = 64,
		bufferSize: int = 100000,
		updateFrequency: int = 4,  # Fréquence d'entraînement (étapes)
		targetUpdateFrequency: int = 10,  # Fréquence de mise à jour du réseau cible (épisodes)
		doubleDQN: bool = True,
		duelingNetwork: bool = True,
		prioritizedExperience: bool = False,
		alpha: float = 0.6,  # Exposant de priorité pour PER
		beta: float = 0.4,  # Exposant d'importance-sampling pour PER
		multiAgent: bool = True,
		explorationDecay: str = "linear",  # "linear", "exponential", "cosine"
		device: torch.device = None
	) -> None:
		"""
		Initialise l'algorithme DQN.
		
		Args:
			agents (List[CollectiveAgent]): Liste des agents à entraîner
			stateDimension (int): Dimension du vecteur d'état
			actionDimension (int): Dimension du vecteur d'action (nombre d'actions discrètes)
			learningRate (float): Taux d'apprentissage du réseau
			gamma (float): Facteur d'actualisation pour les récompenses futures
			epsilon (float): Valeur initiale du paramètre d'exploration epsilon-greedy
			epsilonDecay (float): Facteur de décroissance d'epsilon
			minEpsilon (float): Valeur minimale d'epsilon
			tau (float): Facteur de mise à jour douce pour le réseau cible
			batchSize (int): Taille des batchs d'apprentissage
			bufferSize (int): Taille du tampon de replay
			updateFrequency (int): Fréquence d'entraînement (étapes)
			targetUpdateFrequency (int): Fréquence de mise à jour du réseau cible (épisodes)
			doubleDQN (bool): Utiliser Double DQN
			duelingNetwork (bool): Utiliser Dueling Network Architecture
			prioritizedExperience (bool): Utiliser Prioritized Experience Replay
			alpha (float): Exposant de priorité pour PER
			beta (float): Exposant d'importance-sampling pour PER
			multiAgent (bool): Mode multi-agents
			explorationDecay (str): Type de décroissance d'epsilon
			device (torch.device): Périphérique d'exécution (CPU/GPU)
		"""
		# Configuration
		self.agents = agents
		self.stateDimension = stateDimension
		self.actionDimension = actionDimension
		self.learningRate = learningRate
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilonDecay = epsilonDecay
		self.minEpsilon = minEpsilon
		self.tau = tau
		self.batchSize = batchSize
		self.updateFrequency = updateFrequency
		self.targetUpdateFrequency = targetUpdateFrequency
		self.doubleDQN = doubleDQN
		self.duelingNetwork = duelingNetwork
		self.prioritizedExperience = prioritizedExperience
		self.alpha = alpha
		self.beta = beta
		self.multiAgent = multiAgent
		self.explorationDecay = explorationDecay
		
		# Compteurs
		self.trainingSteps = 0
		self.episodeCount = 0
		self.totalSteps = 0
		
		# Initialiser le périphérique
		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = device
			
		print(f"DQN utilisant le périphérique: {self.device}")
		
		# Initialiser les réseaux
		self.qNetwork = QNetwork(
			stateDimension=stateDimension,
			actionDimension=actionDimension,
			hiddenLayers=[128, 64],
			dueling=duelingNetwork
		).to(self.device)
		
		# Réseau cible (pour la stabilité)
		self.targetQNetwork = QNetwork(
			stateDimension=stateDimension,
			actionDimension=actionDimension,
			hiddenLayers=[128, 64],
			dueling=duelingNetwork
		).to(self.device)
		
		# Initialiser le réseau cible avec les mêmes poids
		self.targetQNetwork.load_state_dict(self.qNetwork.state_dict())
		self.targetQNetwork.eval()  # Mode évaluation (pas de gradient)
		
		# Initialiser l'optimiseur
		self.optimizer = optim.Adam(self.qNetwork.parameters(), lr=learningRate)
		
		# Initialiser le tampon de replay
		if multiAgent:
			# Tampon multi-agents
			self.replayBuffer = MultiAgentReplayBuffer(
				capacity=bufferSize,
				batchSize=batchSize,
				stateDimension=stateDimension,
				balanceAgents=True
			)
		elif prioritizedExperience:
			# Tampon avec priorité
			self.replayBuffer = PrioritizedReplayBuffer(
				capacity=bufferSize,
				batchSize=batchSize,
				stateDimension=stateDimension,
				alpha=alpha,
				beta=beta
			)
		else:
			# Tampon standard
			self.replayBuffer = ReplayBuffer(
				capacity=bufferSize,
				batchSize=batchSize,
				stateDimension=stateDimension
			)
	
	def getAction(self, agent: CollectiveAgent, state: np.ndarray) -> int:
		"""
		Choisit une action selon la politique epsilon-greedy.
		
		Args:
			agent (CollectiveAgent): Agent pour lequel choisir l'action
			state (np.ndarray): État actuel
			
		Returns:
			int: Action choisie
		"""
		# Convertir l'état en tensor
		stateTensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
		
		# Exploration vs exploitation
		if np.random.random() < self.epsilon:
			# Exploration: action aléatoire
			action = np.random.randint(0, self.actionDimension)
		else:
			# Exploitation: meilleure action selon le réseau Q
			self.qNetwork.eval()
			with torch.no_grad():
				actionValues = self.qNetwork(stateTensor)
			self.qNetwork.train()
			
			# Sélectionner l'action avec la valeur Q maximale
			action = torch.argmax(actionValues).item()
		
		return action
	
	def storeExperience(
		self, 
		agent: CollectiveAgent, 
		state: np.ndarray, 
		action: int, 
		reward: float, 
		nextState: np.ndarray, 
		done: bool
	) -> None:
		"""
		Stocke une expérience dans le tampon de replay.
		
		Args:
			agent (CollectiveAgent): Agent qui a généré l'expérience
			state (np.ndarray): État
			action (int): Action effectuée
			reward (float): Récompense reçue
			nextState (np.ndarray): État suivant
			done (bool): Indique si l'état suivant est terminal
		"""
		if self.multiAgent:
			# Mode multi-agents: stocker avec l'ID de l'agent
			self.replayBuffer.add(agent.agentId, state, action, reward, nextState, done)
		else:
			# Mode standard
			self.replayBuffer.add(state, action, reward, nextState, done)
		
		# Incrémenter le compteur d'étapes
		self.totalSteps += 1
	
	def learn(self) -> Optional[float]:
		"""
		Effectue une étape d'apprentissage si le moment est venu.
		
		Returns:
			Optional[float]: Perte d'apprentissage si une mise à jour a été effectuée, None sinon
		"""
		# Mettre à jour seulement à la fréquence spécifiée
		if self.totalSteps % self.updateFrequency != 0:
			return None
			
		# Vérifier si on a assez d'expériences
		if len(self.replayBuffer) < self.batchSize:
			return None
			
		# Incrémenter le compteur d'étapes d'apprentissage
		self.trainingSteps += 1
		
		# Échantillonner des expériences
		if self.prioritizedExperience:
			states, actions, rewards, nextStates, dones, weights, indices = (
				self.replayBuffer.sample(device=self.device)
			)
		else:
			states, actions, rewards, nextStates, dones = (
				self.replayBuffer.sample(device=self.device)
			)
			weights = torch.ones_like(rewards, device=self.device)  # Poids uniformes
		
		# Calculer les Q-values pour l'état actuel
		qValues = self.qNetwork(states).gather(1, actions)
		
		# Calculer les Q-values cibles
		with torch.no_grad():
			if self.doubleDQN:
				# Double DQN: utiliser le réseau principal pour choisir l'action
				# et le réseau cible pour évaluer sa valeur
				nextActions = self.qNetwork(nextStates).argmax(1, keepdim=True)
				nextQValues = self.targetQNetwork(nextStates).gather(1, nextActions)
			else:
				# DQN standard: utiliser le réseau cible pour choisir et évaluer
				nextQValues = self.targetQNetwork(nextStates).max(1, keepdim=True)[0]
			
			# Calculer les valeurs cibles
			targets = rewards + (self.gamma * nextQValues * (1 - dones))
		
		# Calculer la perte
		td_errors = targets - qValues
		loss = (weights * td_errors.pow(2)).mean()
		
		# Mise à jour des priorités pour PER
		if self.prioritizedExperience:
			# Mettre à jour les priorités en fonction des erreurs TD
			priorityUpdates = torch.abs(td_errors).detach().cpu().numpy()
			self.replayBuffer.updatePriorities(indices, priorityUpdates)
		
		# Optimisation
		self.optimizer.zero_grad()
		loss.backward()
		
		# Limiter la norme du gradient pour la stabilité (clipping)
		torch.nn.utils.clip_grad_norm_(self.qNetwork.parameters(), 1.0)
		
		self.optimizer.step()
		
		# Mise à jour douce du réseau cible (soft update)
		if self.tau > 0:
			self._softUpdate()
		
		return loss.item()
	
	def endEpisode(self) -> None:
		"""
		Effectue les opérations de fin d'épisode.
		"""
		# Incrémenter le compteur d'épisodes
		self.episodeCount += 1
		
		# Mise à jour complète du réseau cible à la fréquence spécifiée
		if self.tau == 0 and self.episodeCount % self.targetUpdateFrequency == 0:
			self._hardUpdate()
		
		# Décroissance d'epsilon
		self._updateEpsilon()
	
	def _softUpdate(self) -> None:
		"""
		Mise à jour douce du réseau cible (Polyak averaging).
		"""
		for targetParam, param in zip(self.targetQNetwork.parameters(), self.qNetwork.parameters()):
			targetParam.data.copy_(self.tau * param.data + (1.0 - self.tau) * targetParam.data)
	
	def _hardUpdate(self) -> None:
		"""
		Mise à jour complète du réseau cible.
		"""
		self.targetQNetwork.load_state_dict(self.qNetwork.state_dict())
	
	def _updateEpsilon(self) -> None:
		"""
		Met à jour le paramètre d'exploration epsilon selon la stratégie choisie.
		"""
		if self.explorationDecay == "exponential":
			# Décroissance exponentielle
			self.epsilon = max(self.minEpsilon, self.epsilon * self.epsilonDecay)
		elif self.explorationDecay == "linear":
			# Décroissance linéaire
			decayFactor = 1.0 - (self.episodeCount / 1000.0)  # 1000 épisodes pour atteindre minEpsilon
			self.epsilon = max(self.minEpsilon, decayFactor)
		elif self.explorationDecay == "cosine":
			# Décroissance en cosinus
			maxEpisodes = 2000  # Nombre d'épisodes pour atteindre minEpsilon
			progress = min(1.0, self.episodeCount / maxEpisodes)
			cosineFactor = 0.5 * (1.0 + math.cos(math.pi * progress))
			self.epsilon = self.minEpsilon + (1.0 - self.minEpsilon) * cosineFactor
	
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
			'qnetwork_state_dict': self.qNetwork.state_dict(),
			'target_qnetwork_state_dict': self.targetQNetwork.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'epsilon': self.epsilon,
			'training_steps': self.trainingSteps,
			'episode_count': self.episodeCount,
			'total_steps': self.totalSteps,
			'config': {
				'state_dimension': self.stateDimension,
				'action_dimension': self.actionDimension,
				'learning_rate': self.learningRate,
				'gamma': self.gamma,
				'epsilon_decay': self.epsilonDecay,
				'min_epsilon': self.minEpsilon,
				'tau': self.tau,
				'batch_size': self.batchSize,
				'update_frequency': self.updateFrequency,
				'target_update_frequency': self.targetUpdateFrequency,
				'double_dqn': self.doubleDQN,
				'dueling_network': self.duelingNetwork,
				'prioritized_experience': self.prioritizedExperience,
				'alpha': self.alpha,
				'beta': self.beta,
				'multi_agent': self.multiAgent,
				'exploration_decay': self.explorationDecay
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
		self.qNetwork.load_state_dict(checkpoint['qnetwork_state_dict'])
		self.targetQNetwork.load_state_dict(checkpoint['target_qnetwork_state_dict'])
		
		# Charger l'état de l'optimiseur
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		
		# Charger les paramètres
		self.epsilon = checkpoint.get('epsilon', self.epsilon)
		self.trainingSteps = checkpoint.get('training_steps', 0)
		self.episodeCount = checkpoint.get('episode_count', 0)
		self.totalSteps = checkpoint.get('total_steps', 0)
		
		print(f"Modèle chargé depuis {path}")
		print(f"  Epsilon: {self.epsilon:.4f}, Épisodes: {self.episodeCount}, Étapes: {self.totalSteps}")
	
	def getStats(self) -> Dict[str, Any]:
		"""
		Retourne les statistiques d'apprentissage.
		
		Returns:
			Dict[str, Any]: Statistiques actuelles
		"""
		return {
			'epsilon': self.epsilon,
			'training_steps': self.trainingSteps,
			'episode_count': self.episodeCount,
			'total_steps': self.totalSteps,
			'buffer_size': len(self.replayBuffer),
			'learning_rate': self.optimizer.param_groups[0]['lr']
		}
	
	def updateLearningRate(self, newLR: float) -> None:
		"""
		Met à jour le taux d'apprentissage de l'optimiseur.
		
		Args:
			newLR (float): Nouveau taux d'apprentissage
		"""
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = newLR
		
		self.learningRate = newLR
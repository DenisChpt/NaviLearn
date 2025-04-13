#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module implémentant la mémoire de replay pour l'apprentissage par renforcement.
"""

import math
import numpy as np
import random
from collections import deque, namedtuple
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Deque


# Définir un tuple nommé pour représenter une expérience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
	"""
	Tampon de replay standard pour stocker et échantillonner des expériences.
	
	Le tampon de replay est une technique clé dans les algorithmes d'apprentissage par
	renforcement off-policy comme DQN, qui permet de briser la corrélation temporelle
	des données et d'améliorer la stabilité de l'apprentissage.
	"""
	
	def __init__(
		self, 
		capacity: int = 100000,
		batchSize: int = 64,
		stateDimension: int = 0,  # 0 = autodétection
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise un tampon de replay.
		
		Args:
			capacity (int): Capacité maximale de stockage
			batchSize (int): Taille des batchs lors de l'échantillonnage
			stateDimension (int): Dimension du vecteur d'état (0 = autodétection)
			seed (Optional[int]): Graine aléatoire
		"""
		self.capacity = capacity
		self.batchSize = batchSize
		self.stateDimension = stateDimension
		self.memory = deque(maxlen=capacity)
		self.position = 0
		
		# Initialiser le générateur aléatoire
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
	
	def add(
		self, 
		state: np.ndarray, 
		action: Any, 
		reward: float, 
		nextState: np.ndarray, 
		done: bool
	) -> None:
		"""
		Ajoute une expérience au tampon.
		
		Args:
			state (np.ndarray): État actuel
			action (Any): Action effectuée
			reward (float): Récompense reçue
			nextState (np.ndarray): État suivant
			done (bool): Indique si l'épisode est terminé
		"""
		# Autodétection de la dimension d'état
		if self.stateDimension == 0 and state is not None:
			self.stateDimension = state.shape[0]
		
		# Créer l'expérience
		experience = Experience(state, action, reward, nextState, done)
		
		# Ajouter à la mémoire
		self.memory.append(experience)
	
	def sample(
		self, 
		batchSize: Optional[int] = None,
		device: torch.device = torch.device("cpu")
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Échantillonne un batch d'expériences aléatoires.
		
		Args:
			batchSize (Optional[int]): Taille du batch (utilise self.batchSize si None)
			device (torch.device): Périphérique pour les tenseurs
			
		Returns:
			Tuple[torch.Tensor, ...]: Batch d'états, actions, récompenses, états suivants, indicateurs de fin
		"""
		if batchSize is None:
			batchSize = self.batchSize
			
		# Vérifier si on a assez d'expériences
		if len(self.memory) < batchSize:
			# Retourner des tenseurs vides ou de taille réduite
			actualBatchSize = len(self.memory)
			if actualBatchSize == 0:
				# Aucune expérience disponible
				return (
					torch.zeros((0, self.stateDimension), device=device),
					torch.zeros((0,), dtype=torch.long, device=device),
					torch.zeros((0,), device=device),
					torch.zeros((0, self.stateDimension), device=device),
					torch.zeros((0,), dtype=torch.bool, device=device)
				)
			# Utiliser toutes les expériences disponibles
			batchSize = actualBatchSize
			
		# Échantillonner des expériences aléatoires
		experiences = random.sample(self.memory, batchSize)
		
		# Extraire les composants des expériences
		states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
		
		# Gérer différents types d'actions
		if isinstance(experiences[0].action, int):
			actions = torch.tensor([e.action for e in experiences], 
								 dtype=torch.long).unsqueeze(1).to(device)
		else:
			actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
			
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
		nextStates = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
		
		return states, actions, rewards, nextStates, dones
	
	def __len__(self) -> int:
		"""
		Retourne la taille actuelle du tampon.
		
		Returns:
			int: Nombre d'expériences stockées
		"""
		return len(self.memory)
	
	def isFull(self) -> bool:
		"""
		Vérifie si le tampon est plein.
		
		Returns:
			bool: True si le tampon est plein, False sinon
		"""
		return len(self.memory) == self.capacity
	
	def clear(self) -> None:
		"""
		Vide le tampon.
		"""
		self.memory.clear()
		self.position = 0


class PrioritizedReplayBuffer:
	"""
	Tampon de replay avec échantillonnage prioritaire.
	
	Cette implémentation utilise des priorités basées sur les erreurs TD
	pour échantillonner des expériences plus utiles plus fréquemment.
	"""
	
	def __init__(
		self, 
		capacity: int = 100000,
		batchSize: int = 64,
		stateDimension: int = 0,  # 0 = autodétection
		alpha: float = 0.6,  # Exposant de priorité
		beta: float = 0.4,  # Exposant pour la correction d'importance-sampling
		betaAnnealing: float = 0.001,  # Augmentation de beta par échantillonnage
		epsilon: float = 1e-5,  # Petite constante pour éviter les priorités nulles
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise un tampon de replay prioritaire.
		
		Args:
			capacity (int): Capacité maximale de stockage
			batchSize (int): Taille des batchs lors de l'échantillonnage
			stateDimension (int): Dimension du vecteur d'état (0 = autodétection)
			alpha (float): Contrôle le degré de priorité (0 = uniforme, 1 = priorité complète)
			beta (float): Corrige le biais d'importance-sampling (0 = pas de correction, 1 = correction complète)
			betaAnnealing (float): Augmentation de beta après chaque échantillonnage
			epsilon (float): Petite constante pour éviter les priorités nulles
			seed (Optional[int]): Graine aléatoire
		"""
		self.capacity = capacity
		self.batchSize = batchSize
		self.stateDimension = stateDimension
		self.alpha = alpha
		self.beta = beta
		self.betaAnnealing = betaAnnealing
		self.epsilon = epsilon
		self.maxPriority = 1.0
		
		# Initialiser le générateur aléatoire
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
		
		# Structures de données pour stocker les expériences et priorités
		self.memory = []
		self.priorities = np.zeros(capacity, dtype=np.float32)
		self.position = 0
	
	def add(
		self, 
		state: np.ndarray, 
		action: Any, 
		reward: float, 
		nextState: np.ndarray, 
		done: bool
	) -> None:
		"""
		Ajoute une expérience au tampon avec priorité maximale.
		
		Args:
			state (np.ndarray): État actuel
			action (Any): Action effectuée
			reward (float): Récompense reçue
			nextState (np.ndarray): État suivant
			done (bool): Indique si l'épisode est terminé
		"""
		# Autodétection de la dimension d'état
		if self.stateDimension == 0 and state is not None:
			self.stateDimension = state.shape[0]
		
		# Créer l'expérience
		experience = Experience(state, action, reward, nextState, done)
		
		# Ajouter à la mémoire avec priorité maximale
		if len(self.memory) < self.capacity:
			self.memory.append(experience)
		else:
			self.memory[self.position] = experience
			
		self.priorities[self.position] = self.maxPriority
		self.position = (self.position + 1) % self.capacity
	
	def sample(
		self, 
		batchSize: Optional[int] = None,
		device: torch.device = torch.device("cpu")
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
		"""
		Échantillonne un batch d'expériences selon leurs priorités.
		
		Args:
			batchSize (Optional[int]): Taille du batch (utilise self.batchSize si None)
			device (torch.device): Périphérique pour les tenseurs
			
		Returns:
			Tuple: batch d'états, actions, récompenses, états suivants, indicateurs de fin,
				  poids d'importance-sampling, indices des échantillons
		"""
		if batchSize is None:
			batchSize = self.batchSize
			
		# Vérifier si on a assez d'expériences
		memorySize = len(self.memory)
		if memorySize < batchSize:
			# Retourner des tenseurs vides ou de taille réduite
			if memorySize == 0:
				# Aucune expérience disponible
				return (
					torch.zeros((0, self.stateDimension), device=device),
					torch.zeros((0,), dtype=torch.long, device=device),
					torch.zeros((0,), device=device),
					torch.zeros((0, self.stateDimension), device=device),
					torch.zeros((0,), dtype=torch.bool, device=device),
					torch.zeros((0,), device=device),
					[]
				)
			# Utiliser toutes les expériences disponibles
			batchSize = memorySize
		
		# Calculer les probabilités d'échantillonnage
		priorities = self.priorities[:memorySize]
		probabilities = priorities ** self.alpha
		probabilities = probabilities / np.sum(probabilities)
		
		# Échantillonner les indices selon les priorités
		indices = np.random.choice(memorySize, batchSize, replace=False, p=probabilities)
		
		# Extraire les expériences
		experiences = [self.memory[idx] for idx in indices]
		
		# Calculer les poids d'importance-sampling
		weights = (memorySize * probabilities[indices]) ** (-self.beta)
		weights = weights / np.max(weights)  # Normaliser
		
		# Augmenter beta pour la convergence vers 1
		self.beta = min(1.0, self.beta + self.betaAnnealing)
		
		# Extraire les composants des expériences
		states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
		
		# Gérer différents types d'actions
		if isinstance(experiences[0].action, int):
			actions = torch.tensor([e.action for e in experiences], 
								 dtype=torch.long).unsqueeze(1).to(device)
		else:
			actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
			
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
		nextStates = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
		weights = torch.from_numpy(weights.astype(np.float32)).float().to(device)
		
		return states, actions, rewards, nextStates, dones, weights, indices
	
	def updatePriorities(self, indices: List[int], errors: np.ndarray) -> None:
		"""
		Met à jour les priorités des expériences basées sur les erreurs TD.
		
		Args:
			indices (List[int]): Indices des expériences à mettre à jour
			errors (np.ndarray): Erreurs TD correspondantes
		"""
		for i, idx in enumerate(indices):
			if idx < len(self.memory):  # Vérifier si l'indice est valide
				# Ajouter epsilon pour éviter les priorités nulles
				priority = abs(errors[i]) + self.epsilon
				
				# Mettre à jour la priorité
				self.priorities[idx] = priority
				
				# Mettre à jour la priorité maximale pour les nouvelles expériences
				self.maxPriority = max(self.maxPriority, priority)
	
	def __len__(self) -> int:
		"""
		Retourne la taille actuelle du tampon.
		
		Returns:
			int: Nombre d'expériences stockées
		"""
		return len(self.memory)
	
	def isFull(self) -> bool:
		"""
		Vérifie si le tampon est plein.
		
		Returns:
			bool: True si le tampon est plein, False sinon
		"""
		return len(self.memory) == self.capacity
	
	def clear(self) -> None:
		"""
		Vide le tampon.
		"""
		self.memory = []
		self.priorities = np.zeros(self.capacity, dtype=np.float32)
		self.position = 0
		self.maxPriority = 1.0


class MultiAgentReplayBuffer:
	"""
	Tampon de replay pour l'apprentissage multi-agents.
	
	Cette implémentation stocke des expériences provenant de différents agents
	et peut échantillonner des batchs équilibrés entre agents.
	"""
	
	def __init__(
		self, 
		capacity: int = 100000,
		batchSize: int = 64,
		stateDimension: int = 0,  # 0 = autodétection
		balanceAgents: bool = True,  # Équilibrer les échantillons entre agents
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise un tampon de replay multi-agents.
		
		Args:
			capacity (int): Capacité maximale de stockage
			batchSize (int): Taille des batchs lors de l'échantillonnage
			stateDimension (int): Dimension du vecteur d'état (0 = autodétection)
			balanceAgents (bool): Équilibrer l'échantillonnage entre agents
			seed (Optional[int]): Graine aléatoire
		"""
		self.capacity = capacity
		self.batchSize = batchSize
		self.stateDimension = stateDimension
		self.balanceAgents = balanceAgents
		
		# Initialiser le générateur aléatoire
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
		
		# Mémoire globale
		self.globalMemory = deque(maxlen=capacity)
		
		# Mémoires par agent
		self.agentMemories = {}  # Dict[agent_id, deque]
	
	def add(
		self, 
		agentId: Any,
		state: np.ndarray, 
		action: Any, 
		reward: float, 
		nextState: np.ndarray, 
		done: bool
	) -> None:
		"""
		Ajoute une expérience d'un agent au tampon.
		
		Args:
			agentId (Any): Identifiant de l'agent
			state (np.ndarray): État actuel
			action (Any): Action effectuée
			reward (float): Récompense reçue
			nextState (np.ndarray): État suivant
			done (bool): Indique si l'épisode est terminé
		"""
		# Autodétection de la dimension d'état
		if self.stateDimension == 0 and state is not None:
			self.stateDimension = state.shape[0]
		
		# Créer l'expérience
		experience = Experience(state, action, reward, nextState, done)
		
		# Ajouter à la mémoire globale
		self.globalMemory.append((agentId, experience))
		
		# Ajouter à la mémoire spécifique de l'agent
		if agentId not in self.agentMemories:
			# Créer une nouvelle mémoire pour cet agent
			maxAgentCapacity = self.capacity // 4  # Limiter la mémoire par agent
			self.agentMemories[agentId] = deque(maxlen=maxAgentCapacity)
			
		self.agentMemories[agentId].append(experience)
	
	def sample(
		self, 
		batchSize: Optional[int] = None,
		agentId: Optional[Any] = None,
		device: torch.device = torch.device("cpu")
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Échantillonne un batch d'expériences.
		
		Args:
			batchSize (Optional[int]): Taille du batch (utilise self.batchSize si None)
			agentId (Optional[Any]): Si spécifié, échantillonne uniquement les expériences de cet agent
			device (torch.device): Périphérique pour les tenseurs
			
		Returns:
			Tuple[torch.Tensor, ...]: Batch d'états, actions, récompenses, états suivants, indicateurs de fin
		"""
		if batchSize is None:
			batchSize = self.batchSize
			
		# Déterminer la source des expériences
		if agentId is not None and agentId in self.agentMemories:
			# Échantillonner uniquement pour l'agent spécifié
			memory = self.agentMemories[agentId]
			if len(memory) < batchSize:
				# Pas assez d'échantillons pour cet agent, utiliser ce qui est disponible
				experiences = list(memory)
			else:
				experiences = random.sample(list(memory), batchSize)
		elif self.balanceAgents and len(self.agentMemories) > 1:
			# Échantillonnage équilibré entre agents
			experiences = []
			agentIds = list(self.agentMemories.keys())
			
			# Déterminer le nombre d'échantillons par agent
			samplesPerAgent = max(1, batchSize // len(agentIds))
			remainingSamples = batchSize - samplesPerAgent * len(agentIds)
			
			# Échantillonner pour chaque agent
			for aid in agentIds:
				agentMemory = self.agentMemories[aid]
				numSamples = min(samplesPerAgent, len(agentMemory))
				
				if numSamples > 0:
					agentSamples = random.sample(list(agentMemory), numSamples)
					experiences.extend(agentSamples)
			
			# Ajouter des échantillons supplémentaires si nécessaire
			if remainingSamples > 0 and len(self.globalMemory) > 0:
				additionalSamples = random.sample(
					[e for _, e in self.globalMemory],
					min(remainingSamples, len(self.globalMemory))
				)
				experiences.extend(additionalSamples)
		else:
			# Échantillonnage global standard
			if len(self.globalMemory) < batchSize:
				# Pas assez d'échantillons, utiliser ce qui est disponible
				experiences = [e for _, e in self.globalMemory]
			else:
				# Échantillonner aléatoirement
				globalSamples = random.sample(list(self.globalMemory), batchSize)
				experiences = [e for _, e in globalSamples]
		
		# Vérifier si on a des expériences
		if not experiences:
			# Aucune expérience disponible
			return (
				torch.zeros((0, self.stateDimension), device=device),
				torch.zeros((0,), dtype=torch.long, device=device),
				torch.zeros((0,), device=device),
				torch.zeros((0, self.stateDimension), device=device),
				torch.zeros((0,), dtype=torch.bool, device=device)
			)
		
		# Extraire les composants des expériences
		states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
		
		# Gérer différents types d'actions
		if isinstance(experiences[0].action, int):
			actions = torch.tensor([e.action for e in experiences], 
								 dtype=torch.long).unsqueeze(1).to(device)
		else:
			actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
			
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
		nextStates = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
		
		return states, actions, rewards, nextStates, dones
	
	def __len__(self) -> int:
		"""
		Retourne la taille actuelle du tampon global.
		
		Returns:
			int: Nombre d'expériences stockées
		"""
		return len(self.globalMemory)
	
	def getAgentBufferSize(self, agentId: Any) -> int:
		"""
		Retourne la taille du tampon pour un agent spécifique.
		
		Args:
			agentId (Any): Identifiant de l'agent
			
		Returns:
			int: Nombre d'expériences stockées pour cet agent
		"""
		if agentId in self.agentMemories:
			return len(self.agentMemories[agentId])
		return 0
	
	def clear(self) -> None:
		"""
		Vide tous les tampons.
		"""
		self.globalMemory.clear()
		for agentId in self.agentMemories:
			self.agentMemories[agentId].clear()
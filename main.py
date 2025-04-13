#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NaviLearn - Système d'apprentissage par renforcement pour navigation collective adaptative
Point d'entrée principal du programme.

Ce module initialise et exécute la simulation d'apprentissage par renforcement
avec visualisation OpenGL/Pygame pour observer l'émergence de comportements
de navigation collaboratifs.
"""

import argparse
import os
import sys
import time
from typing import Dict, Any, Optional, List, Tuple

# Ajout du répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.configManager import ConfigManager
from utils.logger import Logger
from utils.profiler import Profiler
from environment.world import World
from agents.smartNavigator import SmartNavigator
from agents.collectiveAgent import CollectiveAgent
from rl.dqnLearner import DQNLearner
from rl.ppoLearner import PPOLearner
from visualization.renderer import Renderer


def parseArguments() -> argparse.Namespace:
	"""
	Parse les arguments de ligne de commande.
	
	Returns:
		argparse.Namespace: Arguments analysés
	"""
	parser = argparse.ArgumentParser(description="NaviLearn - RL pour navigation collective adaptative")
	parser.add_argument("--config", type=str, default="config.yaml", help="Chemin vers le fichier de configuration")
	parser.add_argument("--render", action="store_true", help="Activer le rendu visuel")
	parser.add_argument("--episodes", type=int, help="Nombre d'épisodes d'entraînement")
	parser.add_argument("--load", type=str, help="Charger un modèle pré-entraîné")
	parser.add_argument("--algorithm", type=str, choices=["dqn", "ppo"], help="Algorithme d'apprentissage")
	parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
					  default="INFO", help="Niveau de journalisation")
	
	return parser.parse_args()


def setupEnvironment(config: Dict[str, Any]) -> World:
	"""
	Configure l'environnement de simulation.
	
	Args:
		config (Dict[str, Any]): Configuration chargée
		
	Returns:
		World: Instance du monde simulé
	"""
	worldConfig = config.get("environment", {})
	return World(
		width=worldConfig.get("width", 1000),
		height=worldConfig.get("height", 1000),
		obstacleCount=worldConfig.get("obstacleCount", 30),
		resourceCount=worldConfig.get("resourceCount", 15),
		weatherEnabled=worldConfig.get("weatherEnabled", True),
		terrainComplexity=worldConfig.get("terrainComplexity", 0.7)
	)


def createAgents(world: World, config: Dict[str, Any]) -> List[CollectiveAgent]:
	"""
	Crée une population d'agents pour la simulation.
	
	Args:
		world (World): Environnement de simulation
		config (Dict[str, Any]): Configuration chargée
		
	Returns:
		List[CollectiveAgent]: Liste des agents créés
	"""
	agentConfig = config.get("agents", {})
	agents = []
	
	for i in range(agentConfig.get("count", 10)):
		agent = CollectiveAgent(
			agentId=i,
			world=world,
			sensorRange=agentConfig.get("sensorRange", 150),
			communicationRange=agentConfig.get("communicationRange", 200),
			maxSpeed=agentConfig.get("maxSpeed", 5.0),
			maxTurnRate=agentConfig.get("maxTurnRate", 0.4),
			memorySize=agentConfig.get("memorySize", 1000)
		)
		agents.append(agent)
	
	# Établir les connexions de communication entre agents
	for agent in agents:
		agent.discoverPeers(agents)
	
	return agents


def setupLearner(agents: List[CollectiveAgent], config: Dict[str, Any], algorithm: str) -> Any:
	"""
	Configure l'algorithme d'apprentissage par renforcement.
	
	Args:
		agents (List[CollectiveAgent]): Liste des agents à entraîner
		config (Dict[str, Any]): Configuration chargée
		algorithm (str): Nom de l'algorithme à utiliser (dqn ou ppo)
		
	Returns:
		L'instance du learner correspondant à l'algorithme choisi
	"""
	rlConfig = config.get("reinforcement_learning", {})
	
	if algorithm.lower() == "dqn":
		return DQNLearner(
			agents=agents,
			stateDimension=rlConfig.get("stateDimension", 32),
			actionDimension=rlConfig.get("actionDimension", 9),
			learningRate=rlConfig.get("learningRate", 0.001),
			gamma=rlConfig.get("gamma", 0.99),
			epsilon=rlConfig.get("epsilon", 1.0),
			epsilonDecay=rlConfig.get("epsilonDecay", 0.995),
			minEpsilon=rlConfig.get("minEpsilon", 0.01),
			batchSize=rlConfig.get("batchSize", 64),
			targetUpdateFrequency=rlConfig.get("targetUpdateFrequency", 10),
			prioritizedExperience=rlConfig.get("prioritizedExperience", True)
		)
	elif algorithm.lower() == "ppo":
		return PPOLearner(
			agents=agents,
			stateDimension=rlConfig.get("stateDimension", 32),
			actionDimension=rlConfig.get("actionDimension", 9),
			learningRate=rlConfig.get("learningRate", 0.0003),
			gamma=rlConfig.get("gamma", 0.99),
			epsilon=rlConfig.get("ppoEpsilon", 0.2),
			valueCoefficient=rlConfig.get("valueCoefficient", 0.5),
			entropyCoefficient=rlConfig.get("entropyCoefficient", 0.01),
			clipRange=rlConfig.get("clipRange", 0.2),
			batchSize=rlConfig.get("batchSize", 64),
			epochsPerUpdate=rlConfig.get("epochsPerUpdate", 4)
		)
	else:
		raise ValueError(f"Algorithme inconnu: {algorithm}")


def setupRenderer(world: World, agents: List[CollectiveAgent], config: Dict[str, Any]) -> Optional[Renderer]:
	"""
	Configure le moteur de rendu si le mode visualisation est activé.
	
	Args:
		world (World): Environnement à visualiser
		agents (List[CollectiveAgent]): Agents à visualiser
		config (Dict[str, Any]): Configuration chargée
		
	Returns:
		Optional[Renderer]: Instance du moteur de rendu ou None si désactivé
	"""
	renderConfig = config.get("rendering", {})
	
	if not renderConfig.get("enabled", True):
		return None
	
	return Renderer(
		world=world,
		agents=agents,
		windowWidth=renderConfig.get("windowWidth", 1280),
		windowHeight=renderConfig.get("windowHeight", 720),
		targetFPS=renderConfig.get("targetFPS", 60),
		showStats=renderConfig.get("showStats", True),
		showPaths=renderConfig.get("showPaths", True),
		showSensors=renderConfig.get("showSensors", True),
		showCommunication=renderConfig.get("showCommunication", True),
		use3D=renderConfig.get("use3D", True),
		graphicalLevel=renderConfig.get("graphicalLevel", 2)
	)


def runSimulation(
	world: World, 
	agents: List[CollectiveAgent], 
	learner: Any, 
	renderer: Optional[Renderer],
	episodes: int,
	logger: Logger,
	profiler: Profiler
) -> None:
	"""
	Exécute la boucle principale de simulation et d'apprentissage.
	
	Args:
		world (World): Environnement de simulation
		agents (List[CollectiveAgent]): Agents dans la simulation
		learner (Any): Algorithme d'apprentissage 
		renderer (Optional[Renderer]): Moteur de rendu (peut être None)
		episodes (int): Nombre d'épisodes d'apprentissage
		logger (Logger): Gestionnaire de logs
		profiler (Profiler): Profiler de performance
	"""
	for episode in range(1, episodes + 1):
		logger.info(f"Début de l'épisode {episode}/{episodes}")
		world.reset()
		
		for agent in agents:
			agent.reset()
		
		stepCount = 0
		episodeComplete = False
		totalReward = 0.0
		
		# Mesure le temps d'exécution de l'épisode
		startTime = time.time()
		
		while not episodeComplete and stepCount < world.maxSteps:
			profiler.startSection("step_processing")
			
			# Mise à jour de l'environnement
			profiler.startSection("world_update")
			world.update()
			profiler.endSection("world_update")
			
			# Itération sur chaque agent
			profiler.startSection("agent_processing")
			actionsDict = {}
			
			for agent in agents:
				# Observation de l'environnement
				state = agent.observeEnvironment()
				
				# Décision d'action
				action = learner.getAction(agent, state)
				actionsDict[agent.agentId] = action
				
				# Exécution de l'action
				agent.executeAction(action)
				
				# Calcul de la récompense
				reward = agent.calculateReward()
				totalReward += reward
				
				# Nouvel état après action
				nextState = agent.observeEnvironment()
				
				# Vérifier si l'agent a terminé son objectif
				done = agent.hasCompletedObjective()
				
				# Mémorisation de l'expérience
				learner.storeExperience(agent, state, action, reward, nextState, done)
				
				# Communication entre agents
				agent.shareFeedback(reward)
			
			profiler.endSection("agent_processing")
			
			# Apprentissage à partir des expériences
			profiler.startSection("learning")
			learner.learn()
			profiler.endSection("learning")
			
			# Vérifier si tous les agents ont terminé
			episodeComplete = all(agent.hasCompletedObjective() for agent in agents)
			
			# Affichage si le renderer est actif
			if renderer is not None:
				profiler.startSection("rendering")
				continueSimulation = renderer.render(world, agents, learner)
				profiler.endSection("rendering")
				
				if not continueSimulation:
					logger.info("Simulation interrompue par l'utilisateur")
					return
			
			stepCount += 1
			profiler.endSection("step_processing")
		
		# Fin de l'épisode
		episodeDuration = time.time() - startTime
		logger.info(f"Fin de l'épisode {episode}: {stepCount} étapes, récompense totale: {totalReward:.2f}, durée: {episodeDuration:.2f}s")
		
		# Affiche les statistiques de performance
		profiler.printStats()
		
		# Sauvegarde périodique du modèle
		if episode % 10 == 0:
			learner.saveModel(f"models/model_episode_{episode}.pt")


def main() -> None:
	"""
	Point d'entrée principal du programme.
	"""
	# Traitement des arguments
	args = parseArguments()
	
	# Configuration
	configManager = ConfigManager(args.config)
	config = configManager.loadConfig()
	
	# Mise à jour de la configuration avec les arguments CLI
	if args.episodes is not None:
		config["training"]["episodes"] = args.episodes
	
	if args.render is not None:
		config["rendering"]["enabled"] = args.render
	
	# Logger
	logger = Logger(logLevel=args.log_level)
	
	# Profiler pour l'analyse des performances
	profiler = Profiler(enabled=config.get("profiling", {}).get("enabled", False))
	
	# Afficher les informations de démarrage
	logger.info("Démarrage de NaviLearn - Système d'apprentissage pour navigation collective adaptative")
	logger.info(f"Configuration chargée depuis: {args.config}")
	
	try:
		# Initialisation de l'environnement
		world = setupEnvironment(config)
		logger.info(f"Environnement créé: {world.width}x{world.height}, {world.obstacleCount} obstacles, {world.resourceCount} ressources")
		
		# Création des agents
		agents = createAgents(world, config)
		logger.info(f"{len(agents)} agents créés")
		
		# Choix de l'algorithme d'apprentissage
		algorithm = args.algorithm if args.algorithm else config.get("reinforcement_learning", {}).get("algorithm", "dqn")
		logger.info(f"Algorithme d'apprentissage: {algorithm}")
		
		# Configuration de l'algorithme d'apprentissage
		learner = setupLearner(agents, config, algorithm)
		
		# Chargement d'un modèle pré-entraîné si spécifié
		if args.load:
			logger.info(f"Chargement du modèle: {args.load}")
			learner.loadModel(args.load)
		
		# Configuration du rendu
		renderer = setupRenderer(world, agents, config)
		if renderer:
			logger.info("Visualisation activée")
		
		# Exécution de la simulation
		episodes = config.get("training", {}).get("episodes", 100)
		runSimulation(world, agents, learner, renderer, episodes, logger, profiler)
		
		# Sauvegarde du modèle final
		learner.saveModel("models/model_final.pt")
		logger.info("Modèle final sauvegardé")
		
	except Exception as e:
		logger.error(f"Erreur lors de l'exécution: {e}")
		import traceback
		logger.error(traceback.format_exc())
		return 1
	
	logger.info("Simulation terminée avec succès")
	return 0


if __name__ == "__main__":
	sys.exit(main())
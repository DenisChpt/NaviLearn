#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour le profilage et l'analyse des performances.
"""

import time
import cProfile
import pstats
import io
from typing import Dict, List, Optional, Any, Set


class Profiler:
	"""
	Profiler pour mesurer et analyser les performances du système.
	
	Cette classe permet de:
	- Mesurer le temps d'exécution de différentes sections du code
	- Identifier les goulots d'étranglement
	- Générer des rapports de performance
	"""
	
	def __init__(self, enabled: bool = True, printInterval: int = 100) -> None:
		"""
		Initialise le profiler.
		
		Args:
			enabled (bool): Si False, le profiling est désactivé (pas d'impact sur les performances)
			printInterval (int): Intervalle d'affichage des statistiques (en nombre d'appels)
		"""
		self.enabled = enabled
		self.printInterval = printInterval
		
		# Dictionnaires pour stocker les temps et statistiques
		self.sectionTimers = {}  # Temps de début pour les sections actives
		self.sectionStats = {}   # Statistiques cumulées par section
		
		# Compteurs
		self.callCount = 0
		self.lastPrintTime = time.time()
		
		# État du profiler
		self.isProfiling = False
		self.cProfiler = None
	
	def startSection(self, sectionName: str) -> None:
		"""
		Commence à mesurer le temps d'une section de code.
		
		Args:
			sectionName (str): Nom de la section à mesurer
		"""
		if not self.enabled:
			return
			
		# Enregistrer le temps de début
		self.sectionTimers[sectionName] = time.time()
	
	def endSection(self, sectionName: str) -> None:
		"""
		Termine la mesure du temps d'une section de code.
		
		Args:
			sectionName (str): Nom de la section mesurée
		"""
		if not self.enabled:
			return
			
		# Vérifier si la section a été démarrée
		if sectionName not in self.sectionTimers:
			return
			
		# Calculer le temps écoulé
		startTime = self.sectionTimers[sectionName]
		elapsedTime = time.time() - startTime
		
		# Mettre à jour les statistiques pour cette section
		if sectionName not in self.sectionStats:
			self.sectionStats[sectionName] = {
				"totalTime": 0.0,
				"callCount": 0,
				"minTime": float('inf'),
				"maxTime": 0.0
			}
			
		stats = self.sectionStats[sectionName]
		stats["totalTime"] += elapsedTime
		stats["callCount"] += 1
		stats["minTime"] = min(stats["minTime"], elapsedTime)
		stats["maxTime"] = max(stats["maxTime"], elapsedTime)
		
		# Supprimer le timer
		del self.sectionTimers[sectionName]
		
		# Incrémenter le compteur global
		self.callCount += 1
		
		# Afficher les statistiques périodiquement
		if self.callCount % self.printInterval == 0:
			self.printStats()
	
	def profileFunction(self, func: Any, *args, **kwargs) -> Any:
		"""
		Profile une fonction avec cProfile.
		
		Args:
			func (Any): Fonction à profiler
			*args: Arguments positionnels pour la fonction
			**kwargs: Arguments nommés pour la fonction
			
		Returns:
			Any: Résultat de la fonction
		"""
		if not self.enabled:
			return func(*args, **kwargs)
			
		# Créer un profiler cProfile
		pr = cProfile.Profile()
		
		# Démarrer le profiling
		pr.enable()
		
		# Exécuter la fonction
		result = func(*args, **kwargs)
		
		# Arrêter le profiling
		pr.disable()
		
		# Générer un rapport
		s = io.StringIO()
		ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
		ps.print_stats(20)  # Top 20 fonctions
		
		# Afficher le rapport
		print("\n--- Profiling Results ---")
		print(s.getvalue())
		print("------------------------\n")
		
		return result
	
	def startProfiling(self) -> None:
		"""
		Démarre le profilage global avec cProfile.
		"""
		if not self.enabled or self.isProfiling:
			return
			
		self.cProfiler = cProfile.Profile()
		self.cProfiler.enable()
		self.isProfiling = True
	
	def stopProfiling(self, outputFile: Optional[str] = None) -> None:
		"""
		Arrête le profilage global et génère un rapport.
		
		Args:
			outputFile (Optional[str]): Fichier de sortie pour le rapport, ou None pour la console
		"""
		if not self.enabled or not self.isProfiling or self.cProfiler is None:
			return
			
		self.cProfiler.disable()
		self.isProfiling = False
		
		# Générer un rapport
		if outputFile:
			# Sauvegarder dans un fichier
			self.cProfiler.dump_stats(outputFile)
			print(f"Profiling data saved to {outputFile}")
		else:
			# Afficher dans la console
			s = io.StringIO()
			ps = pstats.Stats(self.cProfiler, stream=s).sort_stats('cumulative')
			ps.print_stats(30)  # Top 30 fonctions
			
			print("\n=== Profiling Results ===")
			print(s.getvalue())
			print("=========================\n")
	
	def printStats(self) -> None:
		"""
		Affiche les statistiques de performance actuelles.
		"""
		if not self.enabled or not self.sectionStats:
			return
			
		# Calculer le temps écoulé depuis le dernier affichage
		currentTime = time.time()
		elapsedTime = currentTime - self.lastPrintTime
		self.lastPrintTime = currentTime
		
		print("\n--- Performance Statistics ---")
		print(f"Interval: {self.printInterval} calls, {elapsedTime:.2f} seconds")
		
		# Trier les sections par temps total
		sortedSections = sorted(
			self.sectionStats.items(),
			key=lambda x: x[1]["totalTime"],
			reverse=True
		)
		
		# En-tête du tableau
		print(f"{'Section':<20} | {'Calls':<8} | {'Total (s)':<10} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10}")
		print("-" * 80)
		
		# Données du tableau
		for section, stats in sortedSections:
			callCount = stats["callCount"]
			totalTime = stats["totalTime"]
			avgTime = (totalTime / callCount) * 1000 if callCount > 0 else 0
			minTime = stats["minTime"] * 1000
			maxTime = stats["maxTime"] * 1000
			
			print(f"{section:<20} | {callCount:<8} | {totalTime:<10.3f} | {avgTime:<10.2f} | {minTime:<10.2f} | {maxTime:<10.2f}")
		
		print("-----------------------------\n")
	
	def reset(self) -> None:
		"""
		Réinitialise toutes les statistiques.
		"""
		self.sectionTimers = {}
		self.sectionStats = {}
		self.callCount = 0
		self.lastPrintTime = time.time()
		
		if self.isProfiling and self.cProfiler is not None:
			self.cProfiler.disable()
			self.cProfiler = cProfile.Profile()
			self.cProfiler.enable()
	
	def getSectionStats(self, sectionName: str) -> Dict[str, Any]:
		"""
		Récupère les statistiques pour une section spécifique.
		
		Args:
			sectionName (str): Nom de la section
			
		Returns:
			Dict[str, Any]: Statistiques de la section, ou dictionnaire vide si non trouvée
		"""
		return self.sectionStats.get(sectionName, {})
	
	def getAllStats(self) -> Dict[str, Dict[str, Any]]:
		"""
		Récupère toutes les statistiques.
		
		Returns:
			Dict[str, Dict[str, Any]]: Toutes les statistiques par section
		"""
		return self.sectionStats
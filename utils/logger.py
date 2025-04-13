#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour la journalisation des événements et messages du système.
"""

import os
import logging
import datetime
from typing import Optional


class Logger:
	"""
	Gestionnaire de journalisation pour le projet NaviLearn.
	
	Cette classe gère:
	- La configuration des journaux
	- L'enregistrement des messages avec différents niveaux
	- La sortie vers la console et/ou des fichiers
	"""
	
	LOG_LEVELS = {
		"DEBUG": logging.DEBUG,
		"INFO": logging.INFO,
		"WARNING": logging.WARNING,
		"ERROR": logging.ERROR,
		"CRITICAL": logging.CRITICAL
	}
	
	def __init__(
		self, 
		logLevel: str = "INFO",
		logToFile: bool = False,
		logDirectory: str = "logs",
		logFormat: Optional[str] = None
	) -> None:
		"""
		Initialise le système de journalisation.
		
		Args:
			logLevel (str): Niveau de journalisation (DEBUG, INFO, WARNING, ERROR, CRITICAL)
			logToFile (bool): Si True, enregistre également les journaux dans un fichier
			logDirectory (str): Répertoire pour les fichiers de journaux
			logFormat (Optional[str]): Format des messages de journal, None pour format par défaut
		"""
		# Convertir le niveau de journal
		self.logLevel = self._getLogLevel(logLevel)
		self.logToFile = logToFile
		self.logDirectory = logDirectory
		
		# Format par défaut des messages
		if logFormat is None:
			self.logFormat = "%(asctime)s [%(levelname)s] %(message)s"
		else:
			self.logFormat = logFormat
		
		# Créer le logger
		self.logger = logging.getLogger("navilearn")
		self.logger.setLevel(self.logLevel)
		
		# Supprimer tous les handlers existants
		for handler in self.logger.handlers[:]:
			self.logger.removeHandler(handler)
		
		# Créer un handler pour la console
		consoleHandler = logging.StreamHandler()
		consoleHandler.setLevel(self.logLevel)
		
		# Définir le format
		formatter = logging.Formatter(self.logFormat)
		consoleHandler.setFormatter(formatter)
		
		# Ajouter le handler au logger
		self.logger.addHandler(consoleHandler)
		
		# Ajouter un handler pour fichier si demandé
		if logToFile:
			self._setupFileHandler()
	
	def _getLogLevel(self, levelName: str) -> int:
		"""
		Convertit un nom de niveau de journal en sa valeur entière.
		
		Args:
			levelName (str): Nom du niveau (DEBUG, INFO, etc.)
			
		Returns:
			int: Valeur entière du niveau de journal
		"""
		return self.LOG_LEVELS.get(levelName.upper(), logging.INFO)
	
	def _setupFileHandler(self) -> None:
		"""
		Configure un handler pour enregistrer les journaux dans un fichier.
		"""
		# Créer le répertoire si nécessaire
		if not os.path.exists(self.logDirectory):
			os.makedirs(self.logDirectory)
		
		# Nom de fichier basé sur la date et l'heure
		timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		logFile = os.path.join(self.logDirectory, f"navilearn_{timestamp}.log")
		
		# Créer le handler
		fileHandler = logging.FileHandler(logFile)
		fileHandler.setLevel(self.logLevel)
		
		# Définir le format
		formatter = logging.Formatter(self.logFormat)
		fileHandler.setFormatter(formatter)
		
		# Ajouter au logger
		self.logger.addHandler(fileHandler)
		
		self.info(f"Log file created: {logFile}")
	
	def debug(self, message: str) -> None:
		"""
		Enregistre un message de niveau DEBUG.
		
		Args:
			message (str): Message à enregistrer
		"""
		self.logger.debug(message)
	
	def info(self, message: str) -> None:
		"""
		Enregistre un message de niveau INFO.
		
		Args:
			message (str): Message à enregistrer
		"""
		self.logger.info(message)
	
	def warning(self, message: str) -> None:
		"""
		Enregistre un message de niveau WARNING.
		
		Args:
			message (str): Message à enregistrer
		"""
		self.logger.warning(message)
	
	def error(self, message: str) -> None:
		"""
		Enregistre un message de niveau ERROR.
		
		Args:
			message (str): Message à enregistrer
		"""
		self.logger.error(message)
	
	def critical(self, message: str) -> None:
		"""
		Enregistre un message de niveau CRITICAL.
		
		Args:
			message (str): Message à enregistrer
		"""
		self.logger.critical(message)
	
	def setLevel(self, level: str) -> None:
		"""
		Change le niveau de journalisation.
		
		Args:
			level (str): Nouveau niveau (DEBUG, INFO, etc.)
		"""
		newLevel = self._getLogLevel(level)
		self.logLevel = newLevel
		self.logger.setLevel(newLevel)
		
		# Mettre à jour tous les handlers
		for handler in self.logger.handlers:
			handler.setLevel(newLevel)
	
	def addLogFile(self, filePath: str) -> None:
		"""
		Ajoute un fichier de journal supplémentaire.
		
		Args:
			filePath (str): Chemin du fichier de journal
		"""
		# Créer le répertoire si nécessaire
		directory = os.path.dirname(filePath)
		if directory and not os.path.exists(directory):
			os.makedirs(directory)
		
		# Créer le handler
		fileHandler = logging.FileHandler(filePath)
		fileHandler.setLevel(self.logLevel)
		
		# Définir le format
		formatter = logging.Formatter(self.logFormat)
		fileHandler.setFormatter(formatter)
		
		# Ajouter au logger
		self.logger.addHandler(fileHandler)
		
		self.info(f"Additional log file added: {filePath}")
	
	def logException(self, exception: Exception, additionalInfo: str = "") -> None:
		"""
		Enregistre une exception avec des informations supplémentaires.
		
		Args:
			exception (Exception): Exception à enregistrer
			additionalInfo (str): Informations supplémentaires
		"""
		message = f"Exception: {type(exception).__name__}: {str(exception)}"
		if additionalInfo:
			message += f" | {additionalInfo}"
		
		self.error(message)
		# On peut également logger le traceback complet
		import traceback
		self.debug(traceback.format_exc())
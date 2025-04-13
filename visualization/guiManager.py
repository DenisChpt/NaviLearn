#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module gérant l'interface utilisateur pour contrôler et visualiser la simulation.
"""

import pygame
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable

from environment.world import World
from agents.collectiveAgent import CollectiveAgent


class GUIManager:
	"""
	Gestionnaire d'interface utilisateur pour la simulation.
	
	Cette classe gère:
	- L'affichage des statistiques et informations
	- Les contrôles interactifs pour la simulation
	- L'affichage des informations sur les agents et l'environnement
	"""
	
	def __init__(
		self,
		screen: pygame.Surface,
		width: int,
		height: int,
		world: World,
		agents: List[CollectiveAgent]
	) -> None:
		"""
		Initialise le gestionnaire d'interface.
		
		Args:
			screen (pygame.Surface): Surface Pygame pour le rendu
			width (int): Largeur de la fenêtre
			height (int): Hauteur de la fenêtre
			world (World): Référence à l'environnement
			agents (List[CollectiveAgent]): Liste des agents
		"""
		self.screen = screen
		self.width = width
		self.height = height
		self.world = world
		self.agents = agents
		
		# Initialiser les polices
		pygame.font.init()
		self.fonts = {
			"small": pygame.font.SysFont("Arial", 12),
			"medium": pygame.font.SysFont("Arial", 16),
			"large": pygame.font.SysFont("Arial", 20),
			"title": pygame.font.SysFont("Arial", 24, bold=True)
		}
		
		# Couleurs pour l'interface
		self.colors = {
			"text": (255, 255, 255),
			"text_highlight": (255, 255, 0),
			"text_muted": (180, 180, 180),
			"background": (40, 40, 50, 180),
			"panel": (30, 30, 40, 220),
			"button": (70, 70, 80),
			"button_hover": (90, 90, 100),
			"button_active": (100, 100, 200),
			"slider": (60, 60, 70),
			"slider_active": (100, 100, 200),
			"border": (100, 100, 120),
			"success": (0, 200, 0),
			"warning": (200, 200, 0),
			"error": (200, 0, 0),
			"graph": [
				(0, 150, 255),    # Bleu
				(0, 200, 0),      # Vert
				(255, 100, 0),    # Orange
				(255, 0, 100),    # Rose
				(150, 100, 200)   # Violet
			]
		}
		
		# État de l'interface
		self.showStatsPanel = True
		self.showControlPanel = True
		self.showAgentInfoPanel = True
		self.showWorldInfoPanel = False
		self.showHelpPanel = False
		
		# Historique des données pour les graphiques
		self.maxHistoryLength = 100
		self.rewardHistory = []
		self.performanceHistory = []
		
		# Composants d'interface
		self.buttons = {}
		self.sliders = {}
		self.checkboxes = {}
		self.activeElement = None
		
		# Créer les composants d'interface
		self._createUIComponents()
	
	def _createUIComponents(self) -> None:
		"""
		Crée les composants d'interface utilisateur.
		"""
		# Boutons
		buttonWidth = 120
		buttonHeight = 30
		buttonMargin = 10
		
		# Définir les boutons
		self.buttons = {
			"toggle_stats": {
				"rect": pygame.Rect(10, 10, buttonWidth, buttonHeight),
				"text": "Statistiques",
				"action": lambda: self._togglePanel("stats"),
				"state": "active" if self.showStatsPanel else "normal"
			},
			"toggle_controls": {
				"rect": pygame.Rect(10, 10 + buttonHeight + buttonMargin, buttonWidth, buttonHeight),
				"text": "Contrôles",
				"action": lambda: self._togglePanel("controls"),
				"state": "active" if self.showControlPanel else "normal"
			},
			"toggle_agent_info": {
				"rect": pygame.Rect(10, 10 + (buttonHeight + buttonMargin) * 2, buttonWidth, buttonHeight),
				"text": "Info Agent",
				"action": lambda: self._togglePanel("agent_info"),
				"state": "active" if self.showAgentInfoPanel else "normal"
			},
			"toggle_world_info": {
				"rect": pygame.Rect(10, 10 + (buttonHeight + buttonMargin) * 3, buttonWidth, buttonHeight),
				"text": "Info Monde",
				"action": lambda: self._togglePanel("world_info"),
				"state": "active" if self.showWorldInfoPanel else "normal"
			},
			"toggle_help": {
				"rect": pygame.Rect(10, 10 + (buttonHeight + buttonMargin) * 4, buttonWidth, buttonHeight),
				"text": "Aide",
				"action": lambda: self._togglePanel("help"),
				"state": "active" if self.showHelpPanel else "normal"
			},
			"pause": {
				"rect": pygame.Rect(self.width - buttonWidth - 10, 10, buttonWidth, buttonHeight),
				"text": "Pause",
				"action": lambda: self._togglePause(),
				"state": "normal"
			},
			"reset": {
				"rect": pygame.Rect(self.width - buttonWidth - 10, 10 + buttonHeight + buttonMargin, buttonWidth, buttonHeight),
				"text": "Réinitialiser",
				"action": lambda: self._resetSimulation(),
				"state": "normal"
			},
			"screenshot": {
				"rect": pygame.Rect(self.width - buttonWidth - 10, 10 + (buttonHeight + buttonMargin) * 2, buttonWidth, buttonHeight),
				"text": "Capture",
				"action": lambda: self._takeScreenshot(),
				"state": "normal"
			}
		}
		
		# Sliders
		sliderWidth = 150
		sliderHeight = 20
		sliderMargin = 10
		
		self.sliders = {
			"speed": {
				"rect": pygame.Rect(self.width - sliderWidth - 10, self.height - 100, sliderWidth, sliderHeight),
				"value": 1.0,  # Vitesse normale
				"min": 0.1,
				"max": 4.0,
				"text": "Vitesse",
				"format": lambda v: f"{v:.1f}x"
			},
			"zoom": {
				"rect": pygame.Rect(self.width - sliderWidth - 10, self.height - 70, sliderWidth, sliderHeight),
				"value": 1.0,  # Zoom normal
				"min": 0.5,
				"max": 2.0,
				"text": "Zoom",
				"format": lambda v: f"{int(v * 100)}%"
			}
		}
		
		# Checkboxes
		checkboxSize = 20
		checkboxMargin = 10
		
		self.checkboxes = {
			"show_paths": {
				"rect": pygame.Rect(self.width - 140, self.height - 160, checkboxSize, checkboxSize),
				"text": "Afficher chemins",
				"checked": True,
				"action": lambda state: self._toggleOption("paths", state)
			},
			"show_sensors": {
				"rect": pygame.Rect(self.width - 140, self.height - 160 + checkboxSize + checkboxMargin, checkboxSize, checkboxSize),
				"text": "Afficher capteurs",
				"checked": True,
				"action": lambda state: self._toggleOption("sensors", state)
			},
			"show_communication": {
				"rect": pygame.Rect(self.width - 140, self.height - 160 + (checkboxSize + checkboxMargin) * 2, checkboxSize, checkboxSize),
				"text": "Afficher communication",
				"checked": True,
				"action": lambda state: self._toggleOption("communication", state)
			}
		}
	
	def render(
		self,
		elapsedTime: float,
		fps: float,
		paused: bool,
		simulationSpeed: float,
		selectedAgent: Optional[int],
		learner: Optional[Any] = None
	) -> None:
		"""
		Effectue le rendu de l'interface utilisateur.
		
		Args:
			elapsedTime (float): Temps écoulé depuis le début de la simulation
			fps (float): Images par seconde actuelles
			paused (bool): Si la simulation est en pause
			simulationSpeed (float): Vitesse de la simulation
			selectedAgent (Optional[int]): ID de l'agent sélectionné, ou None
			learner (Optional[Any]): Algorithme d'apprentissage, pour les statistiques
		"""
		# Mettre à jour l'état des boutons en fonction des paramètres
		self.buttons["pause"]["text"] = "Reprendre" if paused else "Pause"
		self.buttons["pause"]["state"] = "active" if paused else "normal"
		
		# Mettre à jour les sliders
		self.sliders["speed"]["value"] = simulationSpeed
		
		# Mettre à jour les statistiques pour les graphiques
		self._updateStatistics(learner)
		
		# Afficher les panneaux d'information
		if self.showStatsPanel:
			self._renderStatsPanel(elapsedTime, fps, simulationSpeed, learner)
		
		if self.showControlPanel:
			self._renderControlPanel()
		
		if self.showAgentInfoPanel and selectedAgent is not None:
			self._renderAgentInfoPanel(selectedAgent)
		
		if self.showWorldInfoPanel:
			self._renderWorldInfoPanel()
		
		if self.showHelpPanel:
			self._renderHelpPanel()
		
		# Afficher les composants d'interface
		self._renderUIComponents()
	
	def _renderStatsPanel(
		self,
		elapsedTime: float,
		fps: float,
		simulationSpeed: float,
		learner: Optional[Any]
	) -> None:
		"""
		Affiche le panneau de statistiques.
		
		Args:
			elapsedTime (float): Temps écoulé depuis le début de la simulation
			fps (float): Images par seconde actuelles
			simulationSpeed (float): Vitesse de la simulation
			learner (Optional[Any]): Algorithme d'apprentissage
		"""
		# Paramètres du panneau
		panelWidth = 250
		panelHeight = 300
		panelX = 10
		panelY = 50  # En dessous des boutons
		
		# Dessiner le fond du panneau
		panelRect = pygame.Rect(panelX, panelY, panelWidth, panelHeight)
		self._drawPanel(panelRect)
		
		# Titre du panneau
		titleText = self.fonts["title"].render("Statistiques", True, self.colors["text"])
		self.screen.blit(titleText, (panelX + 10, panelY + 10))
		
		# Informations générales
		yOffset = panelY + 45
		lineHeight = 20
		
		# Temps et FPS
		hours = int(elapsedTime // 3600)
		minutes = int((elapsedTime % 3600) // 60)
		seconds = int(elapsedTime % 60)
		timeStr = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
		
		self._drawTextLine("Temps écoulé:", timeStr, panelX + 10, yOffset)
		self._drawTextLine("FPS:", f"{fps:.1f}", panelX + 10, yOffset + lineHeight)
		self._drawTextLine("Vitesse:", f"{simulationSpeed:.1f}x", panelX + 10, yOffset + lineHeight * 2)
		
		# Statistiques de l'environnement
		self._drawTextLine("Ressources restantes:", str(self.world.getRemainingResources()), 
						  panelX + 10, yOffset + lineHeight * 3)
		self._drawTextLine("Ressources collectées:", str(self.world.getResourcesCollected()), 
						  panelX + 10, yOffset + lineHeight * 4)
		
		# Statistiques d'apprentissage si disponibles
		if learner is not None:
			learnerStats = getattr(learner, "getStats", lambda: {})()
			
			yOffset += lineHeight * 5
			self._drawTextLine("Épisodes:", str(learnerStats.get("episode_count", 0)), 
							  panelX + 10, yOffset)
			self._drawTextLine("Étapes totales:", str(learnerStats.get("total_steps", 0)), 
							  panelX + 10, yOffset + lineHeight)
			
			# Affichage de l'epsilon pour DQN ou du learning rate
			if "epsilon" in learnerStats:
				self._drawTextLine("Epsilon:", f"{learnerStats['epsilon']:.4f}", 
								  panelX + 10, yOffset + lineHeight * 2)
			
			if "learning_rate" in learnerStats:
				self._drawTextLine("Taux d'apprentissage:", f"{learnerStats['learning_rate']:.5f}", 
								  panelX + 10, yOffset + lineHeight * 3)
			
			# Ajouter des statistiques avancées selon l'algorithme
			if "policy_loss_mean" in learnerStats:  # Pour PPO
				self._drawTextLine("Perte politique:", f"{learnerStats['policy_loss_mean']:.4f}", 
								  panelX + 10, yOffset + lineHeight * 4)
				self._drawTextLine("Perte valeur:", f"{learnerStats['value_loss_mean']:.4f}", 
								  panelX + 10, yOffset + lineHeight * 5)
		
		# Dessiner les graphiques
		graphRect = pygame.Rect(panelX + 10, panelY + 180, panelWidth - 20, 100)
		self._drawRewardGraph(graphRect, learner)
	
	def _renderControlPanel(self) -> None:
		"""
		Affiche le panneau de contrôle.
		"""
		# Paramètres du panneau
		panelWidth = 250
		panelHeight = 200
		panelX = 10
		panelY = 360  # En dessous du panneau de statistiques
		
		# Dessiner le fond du panneau
		panelRect = pygame.Rect(panelX, panelY, panelWidth, panelHeight)
		self._drawPanel(panelRect)
		
		# Titre du panneau
		titleText = self.fonts["title"].render("Contrôles", True, self.colors["text"])
		self.screen.blit(titleText, (panelX + 10, panelY + 10))
		
		# Informations d'aide sur les contrôles
		yOffset = panelY + 45
		lineHeight = 20
		
		controls = [
			("Espace", "Pause/Reprendre"),
			("+/-", "Ajuster vitesse"),
			("Flèches", "Changer agent"),
			("G", "Grille On/Off"),
			("P", "Chemins On/Off"),
			("S", "Capteurs On/Off"),
			("C", "Communication On/Off"),
			("F", "Suivre agent"),
			("R", "Reset caméra")
		]
		
		for i, (key, action) in enumerate(controls):
			self._drawTextLine(key + ":", action, panelX + 10, yOffset + lineHeight * i)
	
	def _renderAgentInfoPanel(self, selectedAgentId: int) -> None:
		"""
		Affiche le panneau d'information sur l'agent sélectionné.
		
		Args:
			selectedAgentId (int): ID de l'agent sélectionné
		"""
		# Trouver l'agent correspondant
		selectedAgent = next((a for a in self.agents if a.agentId == selectedAgentId), None)
		if selectedAgent is None:
			return
			
		# Paramètres du panneau
		panelWidth = 300
		panelHeight = 350
		panelX = self.width - panelWidth - 10
		panelY = 70  # En dessous des boutons
		
		# Dessiner le fond du panneau
		panelRect = pygame.Rect(panelX, panelY, panelWidth, panelHeight)
		self._drawPanel(panelRect)
		
		# Titre du panneau
		titleText = self.fonts["title"].render(f"Agent #{selectedAgentId}", True, self.colors["text"])
		self.screen.blit(titleText, (panelX + 10, panelY + 10))
		
		# Informations sur l'agent
		yOffset = panelY + 45
		lineHeight = 20
		
		# Position et orientation
		x, y = selectedAgent.position
		heading = math.degrees(selectedAgent.heading) % 360
		self._drawTextLine("Position:", f"({x:.1f}, {y:.1f})", panelX + 10, yOffset)
		self._drawTextLine("Orientation:", f"{heading:.1f}°", panelX + 10, yOffset + lineHeight)
		self._drawTextLine("Vitesse:", f"{selectedAgent.velocity:.2f}", panelX + 10, yOffset + lineHeight * 2)
		
		# Statistiques de performance
		self._drawTextLine("Ressources collectées:", str(selectedAgent.resourcesCollected), 
						  panelX + 10, yOffset + lineHeight * 3)
		self._drawTextLine("Distance parcourue:", f"{selectedAgent.distanceTraveled:.1f}", 
						  panelX + 10, yOffset + lineHeight * 4)
		self._drawTextLine("Collisions:", str(selectedAgent.collisionCount), 
						  panelX + 10, yOffset + lineHeight * 5)
		
		# Ajouter des informations spécifiques à l'agent collectif
		if hasattr(selectedAgent, "assignedRole"):
			roleText = selectedAgent.assignedRole.capitalize()
			self._drawTextLine("Rôle:", roleText, panelX + 10, yOffset + lineHeight * 6)
			
		if hasattr(selectedAgent, "collaborationScore"):
			self._drawTextLine("Score de collaboration:", f"{selectedAgent.collaborationScore:.1f}", 
							  panelX + 10, yOffset + lineHeight * 7)
			
		if hasattr(selectedAgent, "informationShared"):
			self._drawTextLine("Informations partagées:", str(selectedAgent.informationShared), 
							  panelX + 10, yOffset + lineHeight * 8)
			
		if hasattr(selectedAgent, "explorationSkill"):
			self._drawTextLine("Compétence exploration:", f"{selectedAgent.explorationSkill:.2f}", 
							  panelX + 10, yOffset + lineHeight * 9)
			self._drawTextLine("Compétence collecte:", f"{selectedAgent.collectionSkill:.2f}", 
							  panelX + 10, yOffset + lineHeight * 10)
			self._drawTextLine("Compétence communication:", f"{selectedAgent.communicationSkill:.2f}", 
							  panelX + 10, yOffset + lineHeight * 11)
		
		# Dessiner un mini-aperçu des capteurs
		sensorPreviewRect = pygame.Rect(panelX + 10, yOffset + lineHeight * 13, panelWidth - 20, 80)
		pygame.draw.rect(self.screen, self.colors["background"], sensorPreviewRect)
		
		if hasattr(selectedAgent, "obstacleDetectors") and hasattr(selectedAgent, "sensorCount"):
			# Dessiner une représentation des capteurs
			center = (sensorPreviewRect.centerx, sensorPreviewRect.centery)
			radius = min(sensorPreviewRect.width, sensorPreviewRect.height) // 2 - 10
			
			# Dessiner l'agent au centre
			pygame.draw.circle(self.screen, self.colors["text_highlight"], center, 5)
			
			# Dessiner les capteurs
			for i in range(selectedAgent.sensorCount):
				sensorFOV = getattr(selectedAgent, "sensorFOV", 2 * math.pi)
				halfFOV = sensorFOV / 2
				
				sensorAngle = selectedAgent.heading - halfFOV + (sensorFOV * i / (selectedAgent.sensorCount - 1))
				
				# Normaliser la distance du capteur
				distance = selectedAgent.obstacleDetectors[i]
				normalizedDistance = distance / selectedAgent.sensorRange
				
				# Calculer le point final
				endX = center[0] + math.cos(sensorAngle) * radius * normalizedDistance
				endY = center[1] + math.sin(sensorAngle) * radius * normalizedDistance
				
				# Dessiner la ligne du capteur
				pygame.draw.line(self.screen, self.colors["text"], center, (endX, endY), 1)
	
	def _renderWorldInfoPanel(self) -> None:
		"""
		Affiche le panneau d'information sur l'environnement.
		"""
		# Paramètres du panneau
		panelWidth = 300
		panelHeight = 230
		panelX = self.width - panelWidth - 10
		panelY = 430  # En dessous du panneau d'information sur l'agent
		
		# Dessiner le fond du panneau
		panelRect = pygame.Rect(panelX, panelY, panelWidth, panelHeight)
		self._drawPanel(panelRect)
		
		# Titre du panneau
		titleText = self.fonts["title"].render("Environnement", True, self.colors["text"])
		self.screen.blit(titleText, (panelX + 10, panelY + 10))
		
		# Informations sur l'environnement
		yOffset = panelY + 45
		lineHeight = 20
		
		# Taille du monde
		self._drawTextLine("Dimensions:", f"{self.world.width} x {self.world.height}", 
						  panelX + 10, yOffset)
		
		# Obstacles et ressources
		self._drawTextLine("Obstacles:", str(len(self.world.obstacles)), 
						  panelX + 10, yOffset + lineHeight)
		self._drawTextLine("Ressources totales:", str(len(self.world.resources)), 
						  panelX + 10, yOffset + lineHeight * 2)
		self._drawTextLine("Ressources restantes:", str(self.world.getRemainingResources()), 
						  panelX + 10, yOffset + lineHeight * 3)
		
		# Étape actuelle
		self._drawTextLine("Étape actuelle:", str(self.world.currentStep), 
						  panelX + 10, yOffset + lineHeight * 4)
		self._drawTextLine("Étapes maximum:", str(self.world.maxSteps), 
						  panelX + 10, yOffset + lineHeight * 5)
		
		# Conditions météorologiques
		weatherConditions = self.world.getWeatherConditions()
		if weatherConditions:
			weatherType = weatherConditions.get("type", "unknown").capitalize()
			self._drawTextLine("Conditions météo:", weatherType, 
							  panelX + 10, yOffset + lineHeight * 6)
			
			# Afficher les détails météo
			rainIntensity = weatherConditions.get("rainIntensity", 0.0)
			fogIntensity = weatherConditions.get("fogIntensity", 0.0)
			windStrength = weatherConditions.get("windStrength", 0.0)
			
			if rainIntensity > 0.1:
				self._drawTextLine("Pluie:", f"{int(rainIntensity * 100)}%", 
								  panelX + 10, yOffset + lineHeight * 7)
			
			if fogIntensity > 0.1:
				self._drawTextLine("Brouillard:", f"{int(fogIntensity * 100)}%", 
								  panelX + 10, yOffset + lineHeight * 8)
			
			if windStrength > 0.1:
				self._drawTextLine("Vent:", f"{int(windStrength * 100)}%", 
								  panelX + 10, yOffset + lineHeight * 9)
	
	def _renderHelpPanel(self) -> None:
		"""
		Affiche le panneau d'aide.
		"""
		# Paramètres du panneau
		panelWidth = 600
		panelHeight = 400
		panelX = (self.width - panelWidth) // 2
		panelY = (self.height - panelHeight) // 2
		
		# Dessiner le fond du panneau
		panelRect = pygame.Rect(panelX, panelY, panelWidth, panelHeight)
		self._drawPanel(panelRect)
		
		# Titre du panneau
		titleText = self.fonts["title"].render("Aide et Instructions", True, self.colors["text"])
		self.screen.blit(titleText, (panelX + 10, panelY + 10))
		
		# Contenu de l'aide
		yOffset = panelY + 50
		lineHeight = 20
		
		helpText = [
			"Bienvenue dans NaviLearn, un système d'apprentissage par renforcement pour la navigation collective adaptative!",
			"",
			"Contrôles de la simulation:",
			"• ESPACE: Pause/Reprise de la simulation",
			"• +/-: Augmenter/Diminuer la vitesse de simulation",
			"• Flèches GAUCHE/DROITE: Changer d'agent sélectionné",
			"• F: Activer/Désactiver le suivi de l'agent sélectionné",
			"",
			"Contrôles de visualisation:",
			"• G: Activer/Désactiver la grille",
			"• P: Activer/Désactiver l'affichage des chemins",
			"• S: Activer/Désactiver l'affichage des capteurs",
			"• C: Activer/Désactiver l'affichage des communications",
			"• W: Activer/Désactiver les effets météorologiques",
			"",
			"Contrôles de caméra:",
			"• Clic droit + Déplacement: Rotation de la caméra",
			"• Clic milieu + Déplacement: Panoramique",
			"• Molette: Zoom avant/arrière",
			"• R: Réinitialiser la position de la caméra",
			"",
			"Pour plus d'informations sur le projet NaviLearn, consultez la documentation."
		]
		
		for i, line in enumerate(helpText):
			textSurface = self.fonts["medium"].render(line, True, self.colors["text"])
			self.screen.blit(textSurface, (panelX + 20, yOffset + lineHeight * i))
	
	def _renderUIComponents(self) -> None:
		"""
		Effectue le rendu des composants d'interface (boutons, sliders, etc.).
		"""
		# Dessiner les boutons
		for name, button in self.buttons.items():
			self._drawButton(button)
		
		# Dessiner les sliders
		for name, slider in self.sliders.items():
			self._drawSlider(slider)
		
		# Dessiner les checkboxes
		for name, checkbox in self.checkboxes.items():
			self._drawCheckbox(checkbox)
	
	def _drawButton(self, button: Dict[str, Any]) -> None:
		"""
		Dessine un bouton.
		
		Args:
			button (Dict[str, Any]): Informations sur le bouton
		"""
		rect = button["rect"]
		text = button["text"]
		state = button["state"]
		
		# Déterminer la couleur en fonction de l'état
		if state == "active":
			color = self.colors["button_active"]
		elif state == "hover":
			color = self.colors["button_hover"]
		else:
			color = self.colors["button"]
		
		# Dessiner le fond du bouton
		pygame.draw.rect(self.screen, color, rect)
		pygame.draw.rect(self.screen, self.colors["border"], rect, 1)
		
		# Dessiner le texte
		textSurface = self.fonts["medium"].render(text, True, self.colors["text"])
		textRect = textSurface.get_rect(center=rect.center)
		self.screen.blit(textSurface, textRect)
	
	def _drawSlider(self, slider: Dict[str, Any]) -> None:
		"""
		Dessine un slider.
		
		Args:
			slider (Dict[str, Any]): Informations sur le slider
		"""
		rect = slider["rect"]
		value = slider["value"]
		minVal = slider["min"]
		maxVal = slider["max"]
		text = slider["text"]
		formatFunc = slider["format"]
		
		# Dessiner l'étiquette
		labelSurface = self.fonts["small"].render(text, True, self.colors["text"])
		self.screen.blit(labelSurface, (rect.x, rect.y - 15))
		
		# Dessiner la valeur
		valueSurface = self.fonts["small"].render(formatFunc(value), True, self.colors["text"])
		self.screen.blit(valueSurface, (rect.x + rect.width + 5, rect.y))
		
		# Dessiner le fond du slider
		pygame.draw.rect(self.screen, self.colors["slider"], rect)
		pygame.draw.rect(self.screen, self.colors["border"], rect, 1)
		
		# Dessiner la position actuelle
		ratio = (value - minVal) / (maxVal - minVal)
		handleX = rect.x + int(ratio * rect.width)
		handleRect = pygame.Rect(handleX - 5, rect.y - 2, 10, rect.height + 4)
		pygame.draw.rect(self.screen, self.colors["slider_active"], handleRect)
		pygame.draw.rect(self.screen, self.colors["border"], handleRect, 1)
	
	def _drawCheckbox(self, checkbox: Dict[str, Any]) -> None:
		"""
		Dessine une case à cocher.
		
		Args:
			checkbox (Dict[str, Any]): Informations sur la case à cocher
		"""
		rect = checkbox["rect"]
		text = checkbox["text"]
		checked = checkbox["checked"]
		
		# Dessiner le fond de la case
		pygame.draw.rect(self.screen, self.colors["button"], rect)
		pygame.draw.rect(self.screen, self.colors["border"], rect, 1)
		
		# Dessiner la coche si activée
		if checked:
			innerRect = pygame.Rect(rect.x + 4, rect.y + 4, rect.width - 8, rect.height - 8)
			pygame.draw.rect(self.screen, self.colors["button_active"], innerRect)
		
		# Dessiner le texte
		textSurface = self.fonts["small"].render(text, True, self.colors["text"])
		self.screen.blit(textSurface, (rect.x + rect.width + 5, rect.y + 2))
	
	def _drawPanel(self, rect: pygame.Rect) -> None:
		"""
		Dessine un panneau semi-transparent.
		
		Args:
			rect (pygame.Rect): Rectangle définissant le panneau
		"""
		# Créer une surface semi-transparente
		panelSurface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
		panelSurface.fill(self.colors["panel"])
		
		# Dessiner la surface sur l'écran
		self.screen.blit(panelSurface, rect)
		
		# Dessiner une bordure
		pygame.draw.rect(self.screen, self.colors["border"], rect, 1)
	
	def _drawTextLine(self, label: str, value: str, x: int, y: int) -> None:
		"""
		Dessine une ligne de texte avec étiquette et valeur.
		
		Args:
			label (str): Étiquette
			value (str): Valeur
			x (int): Position X
			y (int): Position Y
		"""
		labelSurface = self.fonts["medium"].render(label, True, self.colors["text_muted"])
		self.screen.blit(labelSurface, (x, y))
		
		valueSurface = self.fonts["medium"].render(value, True, self.colors["text"])
		self.screen.blit(valueSurface, (x + 180, y))
	
	def _drawRewardGraph(self, rect: pygame.Rect, learner: Optional[Any]) -> None:
		"""
		Dessine un graphique de récompenses.
		
		Args:
			rect (pygame.Rect): Rectangle définissant la zone du graphique
			learner (Optional[Any]): Algorithme d'apprentissage
		"""
		# Dessiner le fond du graphique
		pygame.draw.rect(self.screen, self.colors["background"], rect)
		pygame.draw.rect(self.screen, self.colors["border"], rect, 1)
		
		# Dessiner les axes
		axisColor = self.colors["text_muted"]
		pygame.draw.line(self.screen, axisColor, (rect.x, rect.y + rect.height - 10), 
						 (rect.x + rect.width, rect.y + rect.height - 10), 1)
		pygame.draw.line(self.screen, axisColor, (rect.x + 10, rect.y), 
						 (rect.x + 10, rect.y + rect.height), 1)
		
		# Si nous avons des données à afficher
		if self.rewardHistory:
			# Normaliser les valeurs pour l'affichage
			maxReward = max(self.rewardHistory)
			minReward = min(self.rewardHistory)
			range_rewards = max(1, maxReward - minReward)
			
			# Dessiner la courbe de récompense
			points = []
			
			for i, reward in enumerate(self.rewardHistory):
				# Calculer la position x
				x = rect.x + 10 + (i * (rect.width - 20) / len(self.rewardHistory))
				
				# Calculer la position y
				y = rect.y + rect.height - 10 - ((reward - minReward) / range_rewards) * (rect.height - 20)
				
				points.append((x, y))
			
			# Dessiner la ligne
			if len(points) >= 2:
				pygame.draw.lines(self.screen, self.colors["graph"][0], False, points, 2)
		
		# Étiquette du graphique
		labelSurface = self.fonts["small"].render("Récompenses", True, self.colors["text"])
		self.screen.blit(labelSurface, (rect.x + 15, rect.y + 5))
	
	def _updateStatistics(self, learner: Optional[Any]) -> None:
		"""
		Met à jour les statistiques pour les graphiques.
		
		Args:
			learner (Optional[Any]): Algorithme d'apprentissage
		"""
		# Obtenir les récompenses récentes des agents
		totalReward = sum(agent.totalReward for agent in self.agents if agent.isActive)
		avgReward = totalReward / max(1, len([a for a in self.agents if a.isActive]))
		
		# Ajouter à l'historique
		self.rewardHistory.append(avgReward)
		
		# Limiter la taille de l'historique
		if len(self.rewardHistory) > self.maxHistoryLength:
			self.rewardHistory.pop(0)
			
		# Mise à jour des performances individuelles des agents
		performances = []
		for agent in self.agents:
			if agent.isActive and hasattr(agent, "efficiencyScore"):
				performances.append(agent.efficiencyScore)
				
		if performances:
			avgPerformance = sum(performances) / len(performances)
			self.performanceHistory.append(avgPerformance)
			
			if len(self.performanceHistory) > self.maxHistoryLength:
				self.performanceHistory.pop(0)
	
	def _togglePanel(self, panelName: str) -> None:
		"""
		Active/désactive un panneau d'interface.
		
		Args:
			panelName (str): Nom du panneau à basculer
		"""
		if panelName == "stats":
			self.showStatsPanel = not self.showStatsPanel
			self.buttons["toggle_stats"]["state"] = "active" if self.showStatsPanel else "normal"
		elif panelName == "controls":
			self.showControlPanel = not self.showControlPanel
			self.buttons["toggle_controls"]["state"] = "active" if self.showControlPanel else "normal"
		elif panelName == "agent_info":
			self.showAgentInfoPanel = not self.showAgentInfoPanel
			self.buttons["toggle_agent_info"]["state"] = "active" if self.showAgentInfoPanel else "normal"
		elif panelName == "world_info":
			self.showWorldInfoPanel = not self.showWorldInfoPanel
			self.buttons["toggle_world_info"]["state"] = "active" if self.showWorldInfoPanel else "normal"
		elif panelName == "help":
			self.showHelpPanel = not self.showHelpPanel
			self.buttons["toggle_help"]["state"] = "active" if self.showHelpPanel else "normal"
	
	def _togglePause(self) -> None:
		"""
		Bascule l'état de pause - action associée au bouton pause.
		Cette méthode sera appelée par le renderer principal.
		"""
		# Ce sera traité par le gestionnaire d'événements principal
		event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE})
		pygame.event.post(event)
	
	def _resetSimulation(self) -> None:
		"""
		Réinitialise la simulation - action associée au bouton reset.
		Cette méthode sera appelée par le renderer principal.
		"""
		# Créer un événement personnalisé
		resetEvent = pygame.event.Event(pygame.USEREVENT, {'action': 'reset'})
		pygame.event.post(resetEvent)
	
	def _takeScreenshot(self) -> None:
		"""
		Prend une capture d'écran - action associée au bouton screenshot.
		Cette méthode sera appelée par le renderer principal.
		"""
		# Créer un événement personnalisé
		screenshotEvent = pygame.event.Event(pygame.USEREVENT, {'action': 'screenshot'})
		pygame.event.post(screenshotEvent)
	
	def _toggleOption(self, option: str, state: bool) -> None:
		"""
		Active/désactive une option de visualisation.
		
		Args:
			option (str): Option à basculer
			state (bool): Nouvel état
		"""
		# Créer un événement personnalisé
		optionEvent = pygame.event.Event(pygame.USEREVENT, {'action': 'toggle_option', 'option': option, 'state': state})
		pygame.event.post(optionEvent)
	
	def handleEvent(self, event: pygame.event.Event) -> bool:
		"""
		Gère les événements de l'interface utilisateur.
		
		Args:
			event (pygame.event.Event): Événement à traiter
			
		Returns:
			bool: True si l'événement a été traité, False sinon
		"""
		# Gestion du clic de souris
		if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
			mousePos = pygame.mouse.get_pos()
			
			# Vérifier si un bouton a été cliqué
			for name, button in self.buttons.items():
				if button["rect"].collidepoint(mousePos):
					# Appeler l'action associée au bouton
					if "action" in button:
						button["action"]()
					return True
			
			# Vérifier si un slider a été cliqué
			for name, slider in self.sliders.items():
				if slider["rect"].collidepoint(mousePos):
					self.activeElement = ("slider", name)
					self._updateSliderValue(name, mousePos[0])
					return True
			
			# Vérifier si une checkbox a été cliquée
			for name, checkbox in self.checkboxes.items():
				if checkbox["rect"].collidepoint(mousePos):
					checkbox["checked"] = not checkbox["checked"]
					if "action" in checkbox:
						checkbox["action"](checkbox["checked"])
					return True
		
		# Gestion du mouvement de souris avec bouton enfoncé
		elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
			if self.activeElement and self.activeElement[0] == "slider":
				sliderName = self.activeElement[1]
				self._updateSliderValue(sliderName, event.pos[0])
				return True
		
		# Gestion du relâchement du bouton de souris
		elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
			if self.activeElement:
				self.activeElement = None
				return True
		
		return False
	
	def _updateSliderValue(self, sliderName: str, xPos: int) -> None:
		"""
		Met à jour la valeur d'un slider.
		
		Args:
			sliderName (str): Nom du slider
			xPos (int): Position X de la souris
		"""
		slider = self.sliders.get(sliderName)
		if not slider:
			return
			
		# Calculer la valeur en fonction de la position de la souris
		rect = slider["rect"]
		minVal = slider["min"]
		maxVal = slider["max"]
		
		# Limiter la position
		xPos = max(rect.x, min(rect.x + rect.width, xPos))
		
		# Calculer la valeur
		ratio = (xPos - rect.x) / rect.width
		value = minVal + ratio * (maxVal - minVal)
		
		# Mettre à jour la valeur
		slider["value"] = value
		
		# Créer un événement pour informer de la modification
		updateEvent = pygame.event.Event(pygame.USEREVENT, {'action': 'slider_changed', 'slider': sliderName, 'value': value})
		pygame.event.post(updateEvent)
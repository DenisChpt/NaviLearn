import math
import numpy as np
import pygame
from pygame.locals import *
import pygame.gfxdraw
import time
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable

# Import OpenGL si disponible
try:
	from OpenGL.GL import *
	from OpenGL.GLU import *
	from OpenGL.GL import shaders
	OPENGL_AVAILABLE = True
except ImportError:
	print("OpenGL non disponible. Utilisation du rendu 2D uniquement.")
	OPENGL_AVAILABLE = False

from environment.world import World
from agents.collectiveAgent import CollectiveAgent
from visualization.camera import Camera
from visualization.shaders import ShaderManager
from visualization.guiManager import GUIManager
from visualization.animations import AnimationManager


class Renderer:
	"""
	Moteur de rendu pour visualiser la simulation.
	
	Cette classe gère:
	- L'initialisation de la fenêtre et du contexte OpenGL
	- Le rendu des agents et de l'environnement en 2D/3D
	- L'interface utilisateur pour contrôler la simulation
	- La gestion des événements utilisateur
	- Les animations et effets visuels
	"""
	
	def __init__(
		self,
		world: World,
		agents: List[CollectiveAgent],
		windowWidth: int = 1280,
		windowHeight: int = 720,
		targetFPS: int = 60,
		showStats: bool = True,
		showPaths: bool = True,
		showSensors: bool = True,
		showCommunication: bool = True,
		use3D: bool = True,
		graphicalLevel: int = 2  # 0=minimal, 1=basic, 2=enhanced, 3=advanced
	) -> None:
		"""
		Initialise le moteur de rendu.
		
		Args:
			world (World): Environnement à visualiser
			agents (List[CollectiveAgent]): Agents à visualiser
			windowWidth (int): Largeur de la fenêtre en pixels
			windowHeight (int): Hauteur de la fenêtre en pixels
			targetFPS (int): Fréquence d'images cible
			showStats (bool): Afficher les statistiques
			showPaths (bool): Afficher les trajectoires des agents
			showSensors (bool): Afficher les capteurs des agents
			showCommunication (bool): Afficher les communications entre agents
			use3D (bool): Utiliser le rendu 3D (OpenGL) si disponible
			graphicalLevel (int): Niveau de détail graphique
		"""
		self.world = world
		self.agents = agents
		self.windowWidth = windowWidth
		self.windowHeight = windowHeight
		self.targetFPS = targetFPS
		self.showStats = showStats
		self.showPaths = showPaths
		self.showSensors = showSensors
		self.showCommunication = showCommunication
		self.graphicalLevel = graphicalLevel
		
		# Déterminer le mode de rendu
		self.use3D = use3D and OPENGL_AVAILABLE
		
		# Initialiser Pygame
		pygame.init()
		pygame.display.set_caption("NaviLearn - Simulation d'apprentissage par renforcement")
		
		# Caméra pour la visualisation
		self.camera = Camera(
			position=(world.width/2, world.height/2, 500),
			target=(world.width/2, world.height/2, 0),
			up=(0, 1, 0),
			fov=45.0,
			aspectRatio=windowWidth/windowHeight,
			nearPlane=1.0,
			farPlane=10000.0
		)

		# Configuration de l'écran
		if self.use3D:
			self.screen = pygame.display.set_mode(
				(windowWidth, windowHeight),
				DOUBLEBUF | OPENGL
			)
			self._initOpenGL()
		else:
			self.screen = pygame.display.set_mode(
				(windowWidth, windowHeight)
			)
		
		
		# Gestionnaire d'interface
		self.guiManager = GUIManager(
			screen=self.screen,
			width=windowWidth,
			height=windowHeight,
			world=world,
			agents=agents
		)
		
		# Gestionnaire d'animations
		self.animationManager = AnimationManager()
		
		# Gestionnaire de shaders (pour OpenGL)
		if self.use3D:
			self.shaderManager = ShaderManager()
			self._loadShaders()
		
		# Textures et ressources graphiques
		self.textures = {}
		self._loadTextures()
		
		# Gestion du temps
		self.clock = pygame.time.Clock()
		self.frameTime = 0.0
		self.elapsedTime = 0.0
		
		# État de l'interface
		self.paused = False
		self.simulationSpeed = 1.0  # 1.0 = vitesse normale
		self.selectedAgent = None
		self.followAgent = False
		self.showGrid = True
		self.showTerrain = True
		self.showWeather = True
		
		# Gestionnaire d'événements personnalisés
		self.customEventHandlers = {}
		
		# Initialiser les mesh et VBOs pour OpenGL
		if self.use3D:
			self._initGeometry()
		
		# Couleurs pour le rendu 2D
		self.colors = {
			"background": (30, 30, 40),
			"grid": (50, 50, 60),
			"text": (255, 255, 255),
			"ui_background": (40, 40, 50, 180),
			"ui_highlight": (80, 80, 200, 220),
			"agent": (0, 150, 255),
			"agent_selected": (255, 215, 0),
			"obstacle": (180, 180, 180),
			"obstacle_dynamic": (200, 100, 50),
			"resource": (0, 255, 0),
			"resource_collected": (100, 100, 100),
			"path": (120, 120, 255, 100),
			"sensor": (255, 255, 255, 80),
			"communication": (255, 255, 0, 150)
		}
	
	def _initOpenGL(self) -> None:
		"""
		Initialise le contexte OpenGL.
		"""
		# Configuration de base
		glEnable(GL_DEPTH_TEST)
		glDepthFunc(GL_LESS)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_POINT_SMOOTH)
		glEnable(GL_LINE_SMOOTH)
		glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
		
		# Configuration de la vue
		glViewport(0, 0, self.windowWidth, self.windowHeight)
		
		# Configuration de la matrice de projection
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(
			self.camera.fov,
			self.camera.aspectRatio,
			self.camera.nearPlane,
			self.camera.farPlane
		)
		
		# Configuration de la matrice de modèle/vue
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		
		# Couleur d'arrière-plan
		glClearColor(0.12, 0.12, 0.15, 1.0)
	
	def _loadShaders(self) -> None:
		"""
		Charge les shaders pour le rendu OpenGL.
		"""
		if not self.use3D:
			return
			
		# Shader de base
		self.shaderManager.addShader(
			"basic",
			"""
			#version 330 core
			layout (location = 0) in vec3 aPos;
			layout (location = 1) in vec3 aColor;
			out vec3 vertexColor;
			uniform mat4 model;
			uniform mat4 view;
			uniform mat4 projection;
			void main() {
				gl_Position = projection * view * model * vec4(aPos, 1.0);
				vertexColor = aColor;
			}
			""",
			"""
			#version 330 core
			in vec3 vertexColor;
			out vec4 FragColor;
			void main() {
				FragColor = vec4(vertexColor, 1.0);
			}
			"""
		)
		
		# Shader pour le terrain
		self.shaderManager.addShader(
			"terrain",
			"""
			#version 330 core
			layout (location = 0) in vec3 aPos;
			layout (location = 1) in vec2 aTexCoord;
			out vec2 TexCoord;
			uniform mat4 model;
			uniform mat4 view;
			uniform mat4 projection;
			void main() {
				gl_Position = projection * view * model * vec4(aPos, 1.0);
				TexCoord = aTexCoord;
			}
			""",
			"""
			#version 330 core
			in vec2 TexCoord;
			out vec4 FragColor;
			uniform sampler2D terrainTexture;
			void main() {
				FragColor = texture(terrainTexture, TexCoord);
			}
			"""
		)
		
		# Shader pour les effets météo
		self.shaderManager.addShader(
			"weather",
			"""
			#version 330 core
			layout (location = 0) in vec3 aPos;
			out vec2 ScreenCoord;
			void main() {
				gl_Position = vec4(aPos, 1.0);
				ScreenCoord = (aPos.xy + 1.0) / 2.0;
			}
			""",
			"""
			#version 330 core
			in vec2 ScreenCoord;
			out vec4 FragColor;
			uniform float time;
			uniform vec2 resolution;
			uniform float rainIntensity;
			uniform float fogIntensity;
			uniform float snowIntensity;
			
			// Fonction de bruit pour les effets
			float hash(vec2 p) {
				return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
			}
			
			void main() {
				vec2 uv = ScreenCoord;
				vec4 color = vec4(0.0);
				
				// Effet de pluie
				if (rainIntensity > 0.0) {
					float rainAmount = rainIntensity * 0.5;
					vec2 positionRain = vec2(uv.x * resolution.x / 50.0, uv.y * resolution.y / 10.0);
					positionRain.y += time * 10.0;
					float rain = hash(floor(positionRain));
					rain = (rain < 0.1 + rainAmount) ? 1.0 : 0.0;
					rain *= (uv.y < 0.95) ? 1.0 : 0.0;  // Pas de pluie en haut de l'écran
					color += vec4(0.7, 0.7, 0.9, rain * rainIntensity * 0.3);
				}
				
				// Effet de brouillard
				if (fogIntensity > 0.0) {
					float fogAmount = fogIntensity * 0.3;
					vec2 positionFog = vec2(uv.x + time * 0.01, uv.y);
					float fog = hash(floor(positionFog * 5.0));
					color += vec4(0.8, 0.8, 0.9, fog * fogAmount);
				}
				
				// Effet de neige
				if (snowIntensity > 0.0) {
					float snowAmount = snowIntensity * 0.5;
					vec2 positionSnow = vec2(
						uv.x * resolution.x / 40.0 + sin(uv.y * 10.0 + time) * 0.5, 
						uv.y * resolution.y / 20.0 + time * 2.0
					);
					float snow = hash(floor(positionSnow));
					snow = (snow < 0.05 + snowAmount) ? 1.0 : 0.0;
					color += vec4(1.0, 1.0, 1.0, snow * snowIntensity * 0.4);
				}
				
				FragColor = color;
			}
			"""
		)
	
	def _loadTextures(self) -> None:
		"""
		Charge les textures et ressources graphiques.
		"""
		# Pour le moment, nous utilisons des primitives géométriques
		# Des textures seraient chargées ici
		pass
	
	def _initGeometry(self) -> None:
		"""
		Initialise les géométries pour le rendu OpenGL.
		"""
		if not self.use3D:
			return
			
		# Géométries prédéfinies à créer ici
		pass
	
	def render(
		self, 
		world: World, 
		agents: List[CollectiveAgent], 
		learner: Optional[Any] = None
	) -> bool:
		"""
		Effectue le rendu de la simulation.
		
		Args:
			world (World): Environnement à visualiser
			agents (List[CollectiveAgent]): Agents à visualiser
			learner (Optional[Any]): Algorithme d'apprentissage (pour les statistiques)
			
		Returns:
			bool: False si la simulation doit s'arrêter, True sinon
		"""
		# Gestion des événements
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				return False
				
			# Traitement des événements personnalisés
			if event.type in self.customEventHandlers:
				self.customEventHandlers[event.type](event)
				
			# Gestion des entrées utilisateur
			self._handleInput(event)
		
		# Mise à jour de la caméra
		if self.followAgent and self.selectedAgent is not None:
			selected = next((a for a in agents if a.agentId == self.selectedAgent), None)
			if selected:
				position = selected.position
				self.camera.setTarget((position[0], position[1], 0))
		
		# Calcul du temps écoulé
		self.frameTime = self.clock.tick(self.targetFPS) / 1000.0
		self.elapsedTime += self.frameTime
		
		# Mise à jour des animations
		self.animationManager.update(self.frameTime)
		
		# Effectuer le rendu
		if self.use3D:
			self._render3D(world, agents, learner)
		else:
			self._render2D(world, agents, learner)
		
		# Afficher l'interface utilisateur
		self.guiManager.render(
			elapsedTime=self.elapsedTime,
			fps=self.clock.get_fps(),
			paused=self.paused,
			simulationSpeed=self.simulationSpeed,
			selectedAgent=self.selectedAgent,
			learner=learner
		)
		
		# Mettre à jour l'affichage
		pygame.display.flip()
		
		return True
	
	def _render2D(
		self, 
		world: World, 
		agents: List[CollectiveAgent], 
		learner: Optional[Any] = None
	) -> None:
		"""
		Effectue le rendu 2D de la simulation.
		
		Args:
			world (World): Environnement à visualiser
			agents (List[CollectiveAgent]): Agents à visualiser
			learner (Optional[Any]): Algorithme d'apprentissage (pour les statistiques)
		"""
		# Effacer l'écran
		self.screen.fill(self.colors["background"])
		
		# Calculer l'échelle et le décalage pour la visualisation
		scaleX = self.windowWidth / world.width
		scaleY = self.windowHeight / world.height
		scale = min(scaleX, scaleY) * 0.9  # Marge de 10%
		
		offsetX = (self.windowWidth - world.width * scale) / 2
		offsetY = (self.windowHeight - world.height * scale) / 2
		
		# Fonction utilitaire pour convertir les coordonnées du monde en coordonnées d'écran
		def worldToScreen(x: float, y: float) -> Tuple[int, int]:
			screenX = int(x * scale + offsetX)
			screenY = int(y * scale + offsetY)
			return (screenX, screenY)
		
		# Dessiner la grille si activée
		if self.showGrid:
			gridSize = 50
			for x in range(0, int(world.width) + 1, gridSize):
				start = worldToScreen(x, 0)
				end = worldToScreen(x, world.height)
				pygame.draw.line(self.screen, self.colors["grid"], start, end, 1)
			
			for y in range(0, int(world.height) + 1, gridSize):
				start = worldToScreen(0, y)
				end = worldToScreen(world.width, y)
				pygame.draw.line(self.screen, self.colors["grid"], start, end, 1)
		
		# Dessiner le terrain si activé
		if self.showTerrain:
			# TODO: Implémentation du rendu de terrain
			pass
		
		# Dessiner les obstacles
		for obstacle in world.obstacles:
			pos = worldToScreen(obstacle.position[0], obstacle.position[1])
			radius = int(obstacle.radius * scale)
			
			color = self.colors["obstacle_dynamic"] if obstacle.isDynamic else self.colors["obstacle"]
			
			if obstacle.traversable:
				# Obstacle traversable: cercle avec bord
				pygame.draw.circle(self.screen, color, pos, radius, 2)
			else:
				# Obstacle solide: cercle plein
				pygame.draw.circle(self.screen, color, pos, radius)
		
		# Dessiner les ressources
		for resource in world.resources:
			pos = worldToScreen(resource.position[0], resource.position[1])
			radius = int(resource.radius * scale)
			
			if not resource.isCollected:
				# Ressource disponible
				color = resource.color
				pygame.draw.circle(self.screen, color, pos, radius)
				
				# Effet de pulsation pour les ressources spéciales
				if resource.pulsating and self.graphicalLevel >= 1:
					pulseRadius = int(resource.getVisualRadius() * scale)
					pygame.draw.circle(self.screen, (*color, 100), pos, pulseRadius, 2)
			elif self.graphicalLevel >= 1:
				# Ressource collectée (visible seulement si niveau graphique > 0)
				color = self.colors["resource_collected"]
				pygame.draw.circle(self.screen, color, pos, radius // 2)
		
		# Dessiner les trajectoires des agents si activé
		if self.showPaths:
			for agent in agents:
				if not agent.isActive:
					continue
					
				# Dessiner la trajectoire (historique des positions)
				if len(agent.positionHistory) >= 2:
					points = [worldToScreen(x, y) for x, y in agent.positionHistory]
					if len(points) >= 2:
						pygame.draw.lines(self.screen, self.colors["path"], False, points, 2)
		
		# Dessiner les capteurs si activé
		if self.showSensors and self.graphicalLevel >= 1:
			for agent in agents:
				if not agent.isActive:
					continue
					
				if hasattr(agent, "obstacleDetectors") and hasattr(agent, "sensorCount"):
					agentPos = worldToScreen(agent.position[0], agent.position[1])
					
					# Calculer l'angle pour chaque capteur
					sensorFOV = getattr(agent, "sensorFOV", 2 * math.pi)
					halfFOV = sensorFOV / 2
					
					for i in range(agent.sensorCount):
						# Calculer l'angle du capteur
						sensorAngle = agent.heading - halfFOV + (sensorFOV * i / (agent.sensorCount - 1))
						
						# Obtenir la distance détectée
						distance = agent.obstacleDetectors[i]
						
						# Calculer le point final du capteur
						endX = agent.position[0] + math.cos(sensorAngle) * distance
						endY = agent.position[1] + math.sin(sensorAngle) * distance
						endPos = worldToScreen(endX, endY)
						
						# Dessiner la ligne du capteur
						pygame.draw.line(self.screen, self.colors["sensor"], agentPos, endPos, 1)
		
		# Dessiner les communications si activé
		if self.showCommunication and self.graphicalLevel >= 1:
			# Créer un ensemble pour éviter les doublons
			drawnConnections = set()
			
			for agent in agents:
				if not agent.isActive:
					continue
					
				# Vérifier les communications récentes
				if hasattr(agent, "lastCommunicationTime"):
					timeSinceComm = world.currentTime - agent.lastCommunicationTime
					
					if timeSinceComm < 100:  # Communication récente
						agentPos = worldToScreen(agent.position[0], agent.position[1])
						
						# Visualiser les agents proches (à portée de communication)
						for otherAgent in agents:
							if agent.agentId != otherAgent.agentId and otherAgent.isActive:
								# Calculer la distance
								distance = agent.getDistanceTo(otherAgent.position)
								
								if distance <= agent.communicationRange:
									# Créer un identifiant unique pour la connexion
									connId = tuple(sorted([agent.agentId, otherAgent.agentId]))
									
									if connId not in drawnConnections:
										drawnConnections.add(connId)
										
										otherPos = worldToScreen(otherAgent.position[0], otherAgent.position[1])
										
										# Dessiner la ligne de communication
										pygame.draw.line(self.screen, self.colors["communication"], agentPos, otherPos, 1)
										
										# Dessiner un cercle pulsant au milieu
										midX = (agent.position[0] + otherAgent.position[0]) / 2
										midY = (agent.position[1] + otherAgent.position[1]) / 2
										midPos = worldToScreen(midX, midY)
										
										pulseSize = int(5 + 3 * math.sin(self.elapsedTime * 10))
										pygame.draw.circle(self.screen, self.colors["communication"], midPos, pulseSize)
		
		# Dessiner les agents
		for agent in agents:
			if not agent.isActive:
				continue
				
			pos = worldToScreen(agent.position[0], agent.position[1])
			
			# Dessiner le corps de l'agent
			selected = (agent.agentId == self.selectedAgent)
			color = self.colors["agent_selected"] if selected else self.colors["agent"]
			
			# Taille en fonction du niveau graphique
			size = int(agent.size * scale)
			if self.graphicalLevel <= 1:
				# Rendu basique: simple cercle
				pygame.draw.circle(self.screen, color, pos, size)
			else:
				# Rendu amélioré: direction visible
				pygame.draw.circle(self.screen, color, pos, size)
				
				# Dessiner la direction
				dirX = agent.position[0] + math.cos(agent.heading) * agent.size * 1.2
				dirY = agent.position[1] + math.sin(agent.heading) * agent.size * 1.2
				dirPos = worldToScreen(dirX, dirY)
				
				pygame.draw.line(self.screen, color, pos, dirPos, 2)
			
			# Ajouter un effet de sélection pour l'agent sélectionné
			if selected and self.graphicalLevel >= 1:
				# Cercle de sélection
				selectionRadius = int(agent.size * scale * 1.5)
				pulseOffset = math.sin(self.elapsedTime * 5) * 2
				pygame.draw.circle(self.screen, self.colors["agent_selected"], pos, selectionRadius + pulseOffset, 2)
			
			# Afficher l'ID de l'agent près de sa position
			if self.graphicalLevel >= 1:
				font = pygame.font.SysFont('Arial', 12)
				idText = font.render(str(agent.agentId), True, self.colors["text"])
				self.screen.blit(idText, (pos[0] + size + 2, pos[1] - 6))
		
		# Effets météorologiques si activés
		if self.showWeather and self.graphicalLevel >= 2:
			weatherConditions = world.getWeatherConditions()
			
			if weatherConditions:
				# Appliquer effets de pluie, neige ou brouillard
				rainIntensity = weatherConditions.get("rainIntensity", 0.0)
				fogIntensity = weatherConditions.get("fogIntensity", 0.0)
				snowIntensity = weatherConditions.get("snowIntensity", 0.0)
				
				# Exemple basique: calque semi-transparent pour le brouillard
				if fogIntensity > 0.2:
					fogSurface = pygame.Surface((self.windowWidth, self.windowHeight), pygame.SRCALPHA)
					fogColor = (255, 255, 255, int(fogIntensity * 60))
					fogSurface.fill(fogColor)
					self.screen.blit(fogSurface, (0, 0))
				
				# Gouttes de pluie simples
				if rainIntensity > 0.2:
					for _ in range(int(rainIntensity * 100)):
						x = random.randint(0, self.windowWidth)
						y = random.randint(0, self.windowHeight)
						length = random.randint(5, 15)
						pygame.draw.line(
							self.screen, 
							(150, 150, 220, 150), 
							(x, y), 
							(x - 2, y + length), 
							1
						)
	
	def _render3D(
		self, 
		world: World, 
		agents: List[CollectiveAgent], 
		learner: Optional[Any] = None
	) -> None:
		"""
		Effectue le rendu 3D de la simulation.
		
		Args:
			world (World): Environnement à visualiser
			agents (List[CollectiveAgent]): Agents à visualiser
			learner (Optional[Any]): Algorithme d'apprentissage (pour les statistiques)
		"""
		# Effacer les buffers
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		
		# Mise à jour de la matrice de vue avec la caméra
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		
		camPos = self.camera.position
		camTarget = self.camera.target
		camUp = self.camera.up
		
		gluLookAt(
			camPos[0], camPos[1], camPos[2],
			camTarget[0], camTarget[1], camTarget[2],
			camUp[0], camUp[1], camUp[2]
		)
		
		# Dessiner le sol (plan XY)
		self._drawGround(world)
		
		# Dessiner la grille si activée
		if self.showGrid:
			self._drawGrid(world)
		
		# Dessiner les obstacles
		for obstacle in world.obstacles:
			self._drawObstacle(obstacle)
		
		# Dessiner les ressources
		for resource in world.resources:
			self._drawResource(resource)
		
		# Dessiner les agents
		for agent in agents:
			if not agent.isActive:
				continue
				
			self._drawAgent(agent, agent.agentId == self.selectedAgent)
			
			# Dessiner les capteurs si activé
			if self.showSensors and self.graphicalLevel >= 1:
				self._drawAgentSensors(agent)
			
			# Dessiner les trajectoires si activé
			if self.showPaths:
				self._drawAgentPath(agent)
		
		# Dessiner les communications si activé
		if self.showCommunication and self.graphicalLevel >= 1:
			self._drawCommunications(agents)
		
		# Dessiner les effets météorologiques
		if self.showWeather and self.graphicalLevel >= 2:
			self._drawWeatherEffects(world)
	
	def _drawGround(self, world: World) -> None:
		"""
		Dessine le sol en 3D.
		
		Args:
			world (World): Environnement de la simulation
		"""
		# Dessiner le plan du sol
		glPushMatrix()
		
		# Couleur du sol
		glColor3f(0.2, 0.2, 0.25)
		
		# Dessiner un rectangle représentant le sol
		glBegin(GL_QUADS)
		glVertex3f(0, 0, -1)
		glVertex3f(world.width, 0, -1)
		glVertex3f(world.width, world.height, -1)
		glVertex3f(0, world.height, -1)
		glEnd()
		
		glPopMatrix()
	
	def _drawGrid(self, world: World) -> None:
		"""
		Dessine une grille sur le sol en 3D.
		
		Args:
			world (World): Environnement de la simulation
		"""
		glPushMatrix()
		
		# Couleur de la grille
		glColor3f(0.3, 0.3, 0.35)
		
		# Dessiner les lignes de la grille
		gridSize = 50.0
		glBegin(GL_LINES)
		
		# Lignes horizontales
		for y in range(0, int(world.height) + 1, int(gridSize)):
			glVertex3f(0, y, 0)
			glVertex3f(world.width, y, 0)
		
		# Lignes verticales
		for x in range(0, int(world.width) + 1, int(gridSize)):
			glVertex3f(x, 0, 0)
			glVertex3f(x, world.height, 0)
			
		glEnd()
		
		glPopMatrix()
	
	def _drawObstacle(self, obstacle: Any) -> None:
		"""
		Dessine un obstacle en 3D.
		
		Args:
			obstacle: Obstacle à dessiner
		"""
		glPushMatrix()
		
		# Positionner l'obstacle
		glTranslatef(obstacle.position[0], obstacle.position[1], 0)
		
		# Couleur selon le type d'obstacle
		if obstacle.isDynamic:
			glColor4f(0.8, 0.4, 0.2, 1.0 if not obstacle.traversable else 0.6)
		else:
			glColor4f(0.7, 0.7, 0.7, 1.0 if not obstacle.traversable else 0.6)
		
		# Dessiner un cylindre pour représenter l'obstacle
		self._drawCylinder(obstacle.radius, obstacle.radius, 20.0, 16)
		
		glPopMatrix()
	
	def _drawResource(self, resource: Any) -> None:
		"""
		Dessine une ressource en 3D.
		
		Args:
			resource: Ressource à dessiner
		"""
		if resource.isCollected:
			# Ressource déjà collectée
			if self.graphicalLevel < 1:
				return  # Ne pas afficher si niveau graphique minimal
				
			glPushMatrix()
			
			# Positionner la ressource
			glTranslatef(resource.position[0], resource.position[1], 0)
			
			# Couleur grisée pour ressource collectée
			glColor4f(0.4, 0.4, 0.4, 0.5)
			
			# Dessiner une sphère aplatie
			self._drawSphere(resource.radius * 0.5, 8, 8)
			
			glPopMatrix()
		else:
			# Ressource disponible
			glPushMatrix()
			
			# Positionner la ressource
			glTranslatef(resource.position[0], resource.position[1], 0)
			
			# Effet de pulsation pour ressources spéciales
			scale = 1.0
			if resource.pulsating and self.graphicalLevel >= 1:
				scale = 1.0 + 0.2 * math.sin(self.elapsedTime * 5)
			
			# Élévation et rotation pour effet visuel
			glRotatef(self.elapsedTime * 30 % 360, 0, 0, 1)
			glTranslatef(0, 0, 5 + 2 * math.sin(self.elapsedTime * 2))
			
			# Couleur de la ressource
			r, g, b = resource.color
			glColor4f(r/255.0, g/255.0, b/255.0, 0.9)
			
			# Dessiner une sphère pour représenter la ressource
			visualRadius = resource.radius * scale
			self._drawSphere(visualRadius, 16, 16)
			
			glPopMatrix()
	
	def _drawAgent(self, agent: Any, selected: bool) -> None:
		"""
		Dessine un agent en 3D.
		
		Args:
			agent: Agent à dessiner
			selected (bool): Si l'agent est sélectionné
		"""
		glPushMatrix()
		
		# Positionner l'agent
		glTranslatef(agent.position[0], agent.position[1], 0)
		
		# Rotation selon l'orientation de l'agent
		glRotatef(math.degrees(agent.heading), 0, 0, 1)
		
		# Couleur selon le statut de sélection
		if selected:
			glColor3f(1.0, 0.84, 0.0)  # Or pour l'agent sélectionné
		else:
			glColor3f(0.0, 0.59, 1.0)  # Bleu pour les agents normaux
		
		# Dessiner le corps de l'agent
		if self.graphicalLevel <= 1:
			# Rendu simple: sphère
			self._drawSphere(agent.size, 12, 12)
		else:
			# Rendu détaillé: corps + direction
			
			# Corps
			self._drawSphere(agent.size, 16, 16)
			
			# Direction (cône)
			glPushMatrix()
			glTranslatef(0, 0, agent.size)
			glRotatef(90, 1, 0, 0)  # Orienter le cône vers l'avant
			self._drawCone(agent.size * 0.6, agent.size * 1.5, 12)
			glPopMatrix()
		
		# Effet visuel pour l'agent sélectionné
		if selected and self.graphicalLevel >= 1:
			# Cercle indicateur pulsant
			glColor4f(1.0, 0.84, 0.0, 0.3 + 0.1 * math.sin(self.elapsedTime * 5))
			glPushMatrix()
			glTranslatef(0, 0, -2)
			self._drawDisc(agent.size * 2.0, 16)
			glPopMatrix()
		
		# Afficher l'ID de l'agent si niveau graphique suffisant
		if self.graphicalLevel >= 1:
			# TODO: Texte 3D pour l'ID (nécessite une implémentation bitmap ou texture)
			pass
		
		glPopMatrix()
	
	def _drawAgentSensors(self, agent: Any) -> None:
		"""
		Dessine les capteurs d'un agent en 3D.
		
		Args:
			agent: Agent dont les capteurs doivent être dessinés
		"""
		if not hasattr(agent, "obstacleDetectors") or not hasattr(agent, "sensorCount"):
			return
			
		glPushMatrix()
		
		# Positionner au niveau de l'agent
		glTranslatef(agent.position[0], agent.position[1], 0)
		
		# Couleur pour les capteurs
		glColor4f(1.0, 1.0, 1.0, 0.3)
		
		# Calculer l'angle pour chaque capteur
		sensorFOV = getattr(agent, "sensorFOV", 2 * math.pi)
		halfFOV = sensorFOV / 2
		
		glBegin(GL_LINES)
		for i in range(agent.sensorCount):
			# Calculer l'angle du capteur
			sensorAngle = agent.heading - halfFOV + (sensorFOV * i / (agent.sensorCount - 1))
			
			# Obtenir la distance détectée
			distance = agent.obstacleDetectors[i]
			
			# Dessiner la ligne du capteur
			glVertex3f(0, 0, agent.size / 2)
			glVertex3f(
				math.cos(sensorAngle) * distance,
				math.sin(sensorAngle) * distance,
				agent.size / 2
			)
		glEnd()
		
		glPopMatrix()
	
	def _drawAgentPath(self, agent: Any) -> None:
		"""
		Dessine la trajectoire d'un agent en 3D.
		
		Args:
			agent: Agent dont la trajectoire doit être dessinée
		"""
		if len(agent.positionHistory) < 2:
			return
			
		glPushMatrix()
		
		# Couleur pour la trajectoire
		glColor4f(0.47, 0.47, 1.0, 0.4)
		
		# Élever légèrement la trajectoire pour éviter le z-fighting
		elevation = 1.0
		
		# Dessiner la ligne de trajectoire
		glBegin(GL_LINE_STRIP)
		for pos in agent.positionHistory:
			glVertex3f(pos[0], pos[1], elevation)
		glEnd()
		
		glPopMatrix()
	
	def _drawCommunications(self, agents: List[Any]) -> None:
		"""
		Dessine les communications entre agents en 3D.
		
		Args:
			agents: Liste des agents
		"""
		# Créer un ensemble pour éviter les doublons
		drawnConnections = set()
		
		glPushMatrix()
		
		# Couleur pour les communications
		glColor4f(1.0, 1.0, 0.0, 0.6)
		
		for agent in agents:
			if not agent.isActive:
				continue
				
			# Vérifier les communications récentes
			if hasattr(agent, "lastCommunicationTime"):
				timeSinceComm = self.world.currentTime - agent.lastCommunicationTime
				
				if timeSinceComm < 100:  # Communication récente
					# Visualiser les agents proches (à portée de communication)
					for otherAgent in agents:
						if agent.agentId != otherAgent.agentId and otherAgent.isActive:
							# Calculer la distance
							distance = agent.getDistanceTo(otherAgent.position)
							
							if distance <= agent.communicationRange:
								# Créer un identifiant unique pour la connexion
								connId = tuple(sorted([agent.agentId, otherAgent.agentId]))
								
								if connId not in drawnConnections:
									drawnConnections.add(connId)
									
									# Dessiner la ligne de communication
									glBegin(GL_LINES)
									glVertex3f(agent.position[0], agent.position[1], agent.size)
									glVertex3f(otherAgent.position[0], otherAgent.position[1], otherAgent.size)
									glEnd()
									
									# Dessiner une sphère pulsante au milieu
									midX = (agent.position[0] + otherAgent.position[0]) / 2
									midY = (agent.position[1] + otherAgent.position[1]) / 2
									
									# Sauvegarder la matrice
									glPushMatrix()
									glTranslatef(midX, midY, agent.size)
									
									# Effet de pulsation
									pulseSize = 5 + 3 * math.sin(self.elapsedTime * 10)
									self._drawSphere(pulseSize, 8, 8)
									
									# Restaurer la matrice
									glPopMatrix()
		
		glPopMatrix()
	
	def _drawWeatherEffects(self, world: Any) -> None:
		"""
		Dessine les effets météorologiques en 3D.
		
		Args:
			world: Environnement contenant les conditions météorologiques
		"""
		weatherConditions = world.getWeatherConditions()
		
		if not weatherConditions:
			return
			
		# Extraire les conditions météo
		rainIntensity = weatherConditions.get("rainIntensity", 0.0)
		fogIntensity = weatherConditions.get("fogIntensity", 0.0)
		snowIntensity = weatherConditions.get("snowIntensity", 0.0)
		
		# Appliquer le brouillard OpenGL
		if fogIntensity > 0.1:
			fogColor = [0.8, 0.8, 0.9, 1.0]
			fogDensity = fogIntensity * 0.01
			
			glEnable(GL_FOG)
			glFogi(GL_FOG_MODE, GL_EXP2)
			glFogfv(GL_FOG_COLOR, fogColor)
			glFogf(GL_FOG_DENSITY, fogDensity)
			glHint(GL_FOG_HINT, GL_NICEST)
		else:
			glDisable(GL_FOG)
		
		# Particules de pluie/neige avec shader si niveau graphique avancé
		if self.graphicalLevel >= 3 and (rainIntensity > 0.2 or snowIntensity > 0.2):
			# Utiliser le shader météo pour des effets avancés
			shader = self.shaderManager.getShader("weather")
			if shader:
				shader.use()
				
				# Définir les paramètres du shader
				shader.setFloat("time", self.elapsedTime)
				shader.setVec2("resolution", (self.windowWidth, self.windowHeight))
				shader.setFloat("rainIntensity", rainIntensity)
				shader.setFloat("fogIntensity", fogIntensity)
				shader.setFloat("snowIntensity", snowIntensity)
				
				# Dessiner un quad plein écran
				glMatrixMode(GL_PROJECTION)
				glPushMatrix()
				glLoadIdentity()
				glMatrixMode(GL_MODELVIEW)
				glPushMatrix()
				glLoadIdentity()
				
				glBegin(GL_QUADS)
				glVertex3f(-1.0, -1.0, 0.0)
				glVertex3f(1.0, -1.0, 0.0)
				glVertex3f(1.0, 1.0, 0.0)
				glVertex3f(-1.0, 1.0, 0.0)
				glEnd()
				
				glPopMatrix()
				glMatrixMode(GL_PROJECTION)
				glPopMatrix()
				glMatrixMode(GL_MODELVIEW)
				
				# Désactiver le shader
				shader.unuse()
	
	def _drawSphere(self, radius: float, slices: int, stacks: int) -> None:
		"""
		Dessine une sphère en 3D.
		
		Args:
			radius (float): Rayon de la sphère
			slices (int): Nombre de divisions horizontales
			stacks (int): Nombre de divisions verticales
		"""
		# Utiliser une quadrique pour dessiner la sphère
		quadric = gluNewQuadric()
		gluQuadricDrawStyle(quadric, GLU_FILL)
		gluQuadricNormals(quadric, GLU_SMOOTH)
		gluSphere(quadric, radius, slices, stacks)
		gluDeleteQuadric(quadric)
	
	def _drawCylinder(self, baseRadius: float, topRadius: float, height: float, slices: int) -> None:
		"""
		Dessine un cylindre ou un cône en 3D.
		
		Args:
			baseRadius (float): Rayon de la base
			topRadius (float): Rayon du sommet
			height (float): Hauteur
			slices (int): Nombre de divisions
		"""
		# Utiliser une quadrique pour dessiner le cylindre
		quadric = gluNewQuadric()
		gluQuadricDrawStyle(quadric, GLU_FILL)
		gluQuadricNormals(quadric, GLU_SMOOTH)
		
		# Dessiner le cylindre
		gluCylinder(quadric, baseRadius, topRadius, height, slices, 1)
		
		# Dessiner les disques aux extrémités pour fermer le cylindre
		gluQuadricOrientation(quadric, GLU_INSIDE)
		gluDisk(quadric, 0, baseRadius, slices, 1)
		
		glPushMatrix()
		glTranslatef(0, 0, height)
		gluQuadricOrientation(quadric, GLU_OUTSIDE)
		gluDisk(quadric, 0, topRadius, slices, 1)
		glPopMatrix()
		
		gluDeleteQuadric(quadric)
	
	def _drawCone(self, radius: float, height: float, slices: int) -> None:
		"""
		Dessine un cône en 3D.
		
		Args:
			radius (float): Rayon de la base
			height (float): Hauteur
			slices (int): Nombre de divisions
		"""
		self._drawCylinder(radius, 0, height, slices)
	
	def _drawDisc(self, radius: float, slices: int) -> None:
		"""
		Dessine un disque en 3D.
		
		Args:
			radius (float): Rayon du disque
			slices (int): Nombre de divisions
		"""
		# Utiliser une quadrique pour dessiner le disque
		quadric = gluNewQuadric()
		gluQuadricDrawStyle(quadric, GLU_FILL)
		gluQuadricNormals(quadric, GLU_SMOOTH)
		gluDisk(quadric, 0, radius, slices, 1)
		gluDeleteQuadric(quadric)
	
	def _handleInput(self, event: pygame.event.Event) -> None:
		"""
		Gère les entrées utilisateur.
		
		Args:
			event (pygame.event.Event): Événement à traiter
		"""
		# Gestion du clavier
		if event.type == pygame.KEYDOWN:
			# Pause/Reprise
			if event.key == pygame.K_SPACE:
				self.paused = not self.paused
			
			# Vitesse de simulation
			elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
				self.simulationSpeed = min(4.0, self.simulationSpeed * 1.5)
			elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
				self.simulationSpeed = max(0.1, self.simulationSpeed / 1.5)
			
			# Affichage
			elif event.key == pygame.K_g:
				self.showGrid = not self.showGrid
			elif event.key == pygame.K_t:
				self.showTerrain = not self.showTerrain
			elif event.key == pygame.K_w:
				self.showWeather = not self.showWeather
			elif event.key == pygame.K_p:
				self.showPaths = not self.showPaths
			elif event.key == pygame.K_s:
				self.showSensors = not self.showSensors
			elif event.key == pygame.K_c:
				self.showCommunication = not self.showCommunication
				
			# Sélection d'agent
			elif event.key == pygame.K_LEFT:
				# Sélectionner l'agent précédent
				if self.selectedAgent is None:
					self.selectedAgent = 0
				else:
					self.selectedAgent = (self.selectedAgent - 1) % len(self.agents)
					
			elif event.key == pygame.K_RIGHT:
				# Sélectionner l'agent suivant
				if self.selectedAgent is None:
					self.selectedAgent = 0
				else:
					self.selectedAgent = (self.selectedAgent + 1) % len(self.agents)
					
			elif event.key == pygame.K_f:
				# Suivre l'agent sélectionné
				self.followAgent = not self.followAgent
				
			# Contrôle de la caméra
			elif event.key == pygame.K_r:
				# Réinitialiser la caméra
				self.camera.reset(self.world.width/2, self.world.height/2)
		
		# Gestion de la souris
		elif event.type == pygame.MOUSEBUTTONDOWN:
			# Sélection d'agent avec le bouton gauche
			if event.button == 1:
				self._selectAgentAtMouse(event.pos)
				
			# Zoom avec la molette
			elif event.button == 4:  # Molette vers le haut
				self.camera.zoom(0.9)
			elif event.button == 5:  # Molette vers le bas
				self.camera.zoom(1.1)
		
		# Gestion du mouvement de souris avec bouton enfoncé
		elif event.type == pygame.MOUSEMOTION:
			# Rotation de caméra avec bouton droit
			if event.buttons[2]:  # Bouton droit
				dx, dy = event.rel
				self.camera.rotate(dx * 0.5, dy * 0.5)
			
			# Panoramique avec bouton milieu
			elif event.buttons[1]:  # Bouton milieu
				dx, dy = event.rel
				self.camera.pan(-dx * 2.0, dy * 2.0)
	
	def _selectAgentAtMouse(self, mousePos: Tuple[int, int]) -> None:
		"""
		Sélectionne un agent à la position de la souris.
		
		Args:
			mousePos (Tuple[int, int]): Position de la souris (x, y)
		"""
		if self.use3D:
			# Sélection 3D via ray-casting
			ray = self._mouseToRay(mousePos)
			
			# Trouver l'agent le plus proche intersectant le rayon
			closestAgent = None
			minDistance = float('inf')
			
			for agent in self.agents:
				if not agent.isActive:
					continue
					
				# Vérifier l'intersection avec la sphère de l'agent
				hit, distance = self._rayIntersectsSphere(
					ray, 
					(agent.position[0], agent.position[1], agent.size/2), 
					agent.size
				)
				
				if hit and distance < minDistance:
					minDistance = distance
					closestAgent = agent
			
			if closestAgent:
				self.selectedAgent = closestAgent.agentId
			else:
				self.selectedAgent = None
		else:
			# Sélection 2D
			x, y = mousePos
			
			# Calculer l'échelle et le décalage pour la visualisation
			scaleX = self.windowWidth / self.world.width
			scaleY = self.windowHeight / self.world.height
			scale = min(scaleX, scaleY) * 0.9  # Marge de 10%
			
			offsetX = (self.windowWidth - self.world.width * scale) / 2
			offsetY = (self.windowHeight - self.world.height * scale) / 2
			
			# Convertir la position de la souris en coordonnées du monde
			worldX = (x - offsetX) / scale
			worldY = (y - offsetY) / scale
			
			# Trouver l'agent le plus proche
			closestAgent = None
			minDistance = float('inf')
			
			for agent in self.agents:
				if not agent.isActive:
					continue
					
				# Calculer la distance
				dx = agent.position[0] - worldX
				dy = agent.position[1] - worldY
				distance = math.sqrt(dx*dx + dy*dy)
				
				# Vérifier si la souris est sur l'agent
				if distance <= agent.size * 1.5 and distance < minDistance:
					minDistance = distance
					closestAgent = agent
			
			if closestAgent:
				self.selectedAgent = closestAgent.agentId
			else:
				self.selectedAgent = None
	
	def _mouseToRay(self, mousePos: Tuple[int, int]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
		"""
		Convertit une position de souris en rayon dans l'espace 3D.
		
		Args:
			mousePos (Tuple[int, int]): Position de la souris (x, y)
			
		Returns:
			Tuple[Tuple[float, float, float], Tuple[float, float, float]]: 
				Origine du rayon et direction normalisée
		"""
		# Obtenir la position de la souris normalisée
		x, y = mousePos
		x = (2.0 * x) / self.windowWidth - 1.0
		y = 1.0 - (2.0 * y) / self.windowHeight
		
		# Calculer le rayon de projection inverse
		# Créer un point homogène dans l'espace de clip
		clipCoords = (x, y, -1.0, 1.0)
		
		# Obtenir les matrices de projection et vue
		projMatrix = glGetDoublev(GL_PROJECTION_MATRIX)
		viewMatrix = glGetDoublev(GL_MODELVIEW_MATRIX)
		
		# Convertir en coordonnées d'œil
		invProj = np.linalg.inv(projMatrix)
		eyeCoords = np.dot(invProj, clipCoords)
		eyeCoords = (eyeCoords[0], eyeCoords[1], -1.0, 0.0)
		
		# Convertir en coordonnées mondiales
		invView = np.linalg.inv(viewMatrix)
		rayWorld = np.dot(invView, eyeCoords)
		rayDir = (rayWorld[0], rayWorld[1], rayWorld[2])
		
		# Normaliser la direction
		length = math.sqrt(rayDir[0]**2 + rayDir[1]**2 + rayDir[2]**2)
		rayDir = (rayDir[0]/length, rayDir[1]/length, rayDir[2]/length)
		
		# Origine du rayon (position de la caméra)
		rayOrigin = self.camera.position
		
		return (rayOrigin, rayDir)
	
	def _rayIntersectsSphere(
		self, 
		ray: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
		sphereCenter: Tuple[float, float, float],
		sphereRadius: float
	) -> Tuple[bool, float]:
		"""
		Vérifie si un rayon intersecte une sphère.
		
		Args:
			ray: Tuple contenant l'origine et la direction du rayon
			sphereCenter: Centre de la sphère
			sphereRadius: Rayon de la sphère
			
		Returns:
			Tuple[bool, float]: True si intersection, et distance à l'intersection
		"""
		rayOrigin, rayDir = ray
		
		# Vecteur de l'origine du rayon au centre de la sphère
		oc = (
			rayOrigin[0] - sphereCenter[0],
			rayOrigin[1] - sphereCenter[1],
			rayOrigin[2] - sphereCenter[2]
		)
		
		# Coefficients de l'équation quadratique
		a = rayDir[0]**2 + rayDir[1]**2 + rayDir[2]**2
		b = 2.0 * (oc[0] * rayDir[0] + oc[1] * rayDir[1] + oc[2] * rayDir[2])
		c = oc[0]**2 + oc[1]**2 + oc[2]**2 - sphereRadius**2
		
		# Discriminant
		discriminant = b**2 - 4 * a * c
		
		if discriminant < 0:
			# Pas d'intersection
			return False, float('inf')
		
		# Calculer les points d'intersection
		t1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
		t2 = (-b + math.sqrt(discriminant)) / (2.0 * a)
		
		# Retourner le point d'intersection le plus proche
		if t1 > 0:
			return True, t1
		elif t2 > 0:
			return True, t2
		else:
			return False, float('inf')
	
	def addCustomEventHandler(self, eventType: int, handler: Callable[[pygame.event.Event], None]) -> None:
		"""
		Ajoute un gestionnaire d'événements personnalisé.
		
		Args:
			eventType (int): Type d'événement Pygame
			handler (Callable): Fonction de traitement de l'événement
		"""
		self.customEventHandlers[eventType] = handler
	
	def takeScreenshot(self, filename: Optional[str] = None) -> None:
		"""
		Prend une capture d'écran de la visualisation actuelle.
		
		Args:
			filename (Optional[str]): Nom du fichier pour la capture d'écran
		"""
		if filename is None:
			# Générer un nom de fichier basé sur la date et l'heure
			timestamp = time.strftime("%Y%m%d-%H%M%S")
			filename = f"screenshot_{timestamp}.png"
		
		# Créer le répertoire si nécessaire
		directory = os.path.dirname(filename)
		if directory and not os.path.exists(directory):
			os.makedirs(directory)
		
		# Sauvegarder la capture d'écran
		pygame.image.save(self.screen, filename)
		print(f"Capture d'écran sauvegardée: {filename}")
	
	def cleanup(self) -> None:
		"""
		Nettoie les ressources avant la fermeture.
		"""
		# Libérer les ressources OpenGL si utilisées
		if self.use3D:
			# Libérer les shaders
			self.shaderManager.cleanup()
			
			# Libérer les textures
			for texture in self.textures.values():
				glDeleteTextures(1, [texture])
		
		# Quitter Pygame
		pygame.quit()
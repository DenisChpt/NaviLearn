#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module gérant les shaders OpenGL pour les effets visuels avancés.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable

# Import OpenGL si disponible
try:
	from OpenGL.GL import *
	from OpenGL.GL import shaders
	OPENGL_AVAILABLE = True
except ImportError:
	print("OpenGL non disponible. Les shaders ne seront pas utilisés.")
	OPENGL_AVAILABLE = False


class Shader:
	"""
	Classe représentant un programme shader OpenGL.
	"""
	
	def __init__(self, vertexCode: str, fragmentCode: str) -> None:
		"""
		Initialise un programme shader.
		
		Args:
			vertexCode (str): Code source du vertex shader
			fragmentCode (str): Code source du fragment shader
		"""
		if not OPENGL_AVAILABLE:
			self.program = None
			return
			
		# Compiler les shaders
		vertexShader = shaders.compileShader(vertexCode, GL_VERTEX_SHADER)
		fragmentShader = shaders.compileShader(fragmentCode, GL_FRAGMENT_SHADER)
		
		# Créer le programme
		self.program = shaders.compileProgram(vertexShader, fragmentShader)
		
		# Stocker les emplacements des uniformes
		self.uniformLocations = {}
	
	def use(self) -> None:
		"""
		Active ce programme shader.
		"""
		if not OPENGL_AVAILABLE or self.program is None:
			return
			
		glUseProgram(self.program)
	
	def unuse(self) -> None:
		"""
		Désactive ce programme shader.
		"""
		if not OPENGL_AVAILABLE:
			return
			
		glUseProgram(0)
	
	def getUniformLocation(self, name: str) -> int:
		"""
		Obtient l'emplacement d'un uniform dans le shader.
		
		Args:
			name (str): Nom de l'uniform
			
		Returns:
			int: Emplacement de l'uniform
		"""
		if not OPENGL_AVAILABLE or self.program is None:
			return -1
			
		if name not in self.uniformLocations:
			self.uniformLocations[name] = glGetUniformLocation(self.program, name)
		
		return self.uniformLocations[name]
	
	def setBool(self, name: str, value: bool) -> None:
		"""
		Définit un uniform booléen.
		
		Args:
			name (str): Nom de l'uniform
			value (bool): Valeur à définir
		"""
		if not OPENGL_AVAILABLE or self.program is None:
			return
			
		glUniform1i(self.getUniformLocation(name), int(value))
	
	def setInt(self, name: str, value: int) -> None:
		"""
		Définit un uniform entier.
		
		Args:
			name (str): Nom de l'uniform
			value (int): Valeur à définir
		"""
		if not OPENGL_AVAILABLE or self.program is None:
			return
			
		glUniform1i(self.getUniformLocation(name), value)
	
	def setFloat(self, name: str, value: float) -> None:
		"""
		Définit un uniform flottant.
		
		Args:
			name (str): Nom de l'uniform
			value (float): Valeur à définir
		"""
		if not OPENGL_AVAILABLE or self.program is None:
			return
			
		glUniform1f(self.getUniformLocation(name), value)
	
	def setVec2(self, name: str, value: Tuple[float, float]) -> None:
		"""
		Définit un uniform vec2.
		
		Args:
			name (str): Nom de l'uniform
			value (Tuple[float, float]): Valeur à définir
		"""
		if not OPENGL_AVAILABLE or self.program is None:
			return
			
		glUniform2f(self.getUniformLocation(name), value[0], value[1])
	
	def setVec3(self, name: str, value: Tuple[float, float, float]) -> None:
		"""
		Définit un uniform vec3.
		
		Args:
			name (str): Nom de l'uniform
			value (Tuple[float, float, float]): Valeur à définir
		"""
		if not OPENGL_AVAILABLE or self.program is None:
			return
			
		glUniform3f(self.getUniformLocation(name), value[0], value[1], value[2])
	
	def setVec4(self, name: str, value: Tuple[float, float, float, float]) -> None:
		"""
		Définit un uniform vec4.
		
		Args:
			name (str): Nom de l'uniform
			value (Tuple[float, float, float, float]): Valeur à définir
		"""
		if not OPENGL_AVAILABLE or self.program is None:
			return
			
		glUniform4f(self.getUniformLocation(name), value[0], value[1], value[2], value[3])
	
	def setMat4(self, name: str, value: Union[List[List[float]], np.ndarray]) -> None:
		"""
		Définit un uniform mat4.
		
		Args:
			name (str): Nom de l'uniform
			value (Union[List[List[float]], np.ndarray]): Matrice 4x4 à définir
		"""
		if not OPENGL_AVAILABLE or self.program is None:
			return
			
		# Convertir en numpy array si nécessaire
		if isinstance(value, list):
			value = np.array(value, dtype=np.float32)
		
		# S'assurer que la matrice est au format float32
		if value.dtype != np.float32:
			value = value.astype(np.float32)
			
		# Définir l'uniform
		glUniformMatrix4fv(self.getUniformLocation(name), 1, GL_FALSE, value)


class ShaderManager:
	"""
	Gestionnaire de shaders pour l'application.
	"""
	
	def __init__(self) -> None:
		"""
		Initialise le gestionnaire de shaders.
		"""
		self.shaders = {}
	
	def addShader(self, name: str, vertexCode: str, fragmentCode: str) -> None:
		"""
		Ajoute un nouveau shader au gestionnaire.
		
		Args:
			name (str): Nom du shader
			vertexCode (str): Code source du vertex shader
			fragmentCode (str): Code source du fragment shader
		"""
		if not OPENGL_AVAILABLE:
			return
			
		try:
			shader = Shader(vertexCode, fragmentCode)
			self.shaders[name] = shader
		except Exception as e:
			print(f"Erreur lors de la compilation du shader '{name}': {e}")
	
	def getShader(self, name: str) -> Optional[Shader]:
		"""
		Récupère un shader par son nom.
		
		Args:
			name (str): Nom du shader
			
		Returns:
			Optional[Shader]: Le shader correspondant, ou None s'il n'existe pas
		"""
		return self.shaders.get(name)
	
	def cleanup(self) -> None:
		"""
		Nettoie les ressources des shaders.
		"""
		if not OPENGL_AVAILABLE:
			return
			
		for name, shader in self.shaders.items():
			if shader.program is not None:
				glDeleteProgram(shader.program)
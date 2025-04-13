#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module définissant la caméra pour la visualisation 3D.
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Any


class Camera:
	"""
	Caméra contrôlable pour la visualisation 3D.
	
	Cette classe gère:
	- La position et l'orientation de la caméra
	- Les transformations de vue et de projection
	- Les contrôles utilisateur (pan, zoom, rotation)
	"""
	
	def __init__(
		self,
		position: Tuple[float, float, float] = (0, 0, 100),
		target: Tuple[float, float, float] = (0, 0, 0),
		up: Tuple[float, float, float] = (0, 1, 0),
		fov: float = 45.0,
		aspectRatio: float = 16/9,
		nearPlane: float = 0.1,
		farPlane: float = 1000.0
	) -> None:
		"""
		Initialise une caméra 3D.
		
		Args:
			position (Tuple[float, float, float]): Position initiale de la caméra
			target (Tuple[float, float, float]): Point ciblé par la caméra
			up (Tuple[float, float, float]): Vecteur "up" pour l'orientation
			fov (float): Champ de vision en degrés
			aspectRatio (float): Ratio largeur/hauteur de la fenêtre
			nearPlane (float): Distance du plan proche de clipping
			farPlane (float): Distance du plan lointain de clipping
		"""
		self.position = position
		self.target = target
		self.up = up
		self.fov = fov
		self.aspectRatio = aspectRatio
		self.nearPlane = nearPlane
		self.farPlane = farPlane
		
		# Angles d'orientation sphérique
		self.yaw = 0.0
		self.pitch = 0.0
		
		# Distance à la cible
		self.distance = self._calculateDistance()
		
		# Calculer les angles initiaux
		self._updateSphericalCoordinates()
	
	def _calculateDistance(self) -> float:
		"""
		Calcule la distance entre la position et la cible.
		
		Returns:
			float: Distance calculée
		"""
		dx = self.position[0] - self.target[0]
		dy = self.position[1] - self.target[1]
		dz = self.position[2] - self.target[2]
		return math.sqrt(dx*dx + dy*dy + dz*dz)
	
	def _updateSphericalCoordinates(self) -> None:
		"""
		Met à jour les angles sphériques basés sur position et target.
		"""
		# Vecteur de la cible à la caméra
		dx = self.position[0] - self.target[0]
		dy = self.position[1] - self.target[1]
		dz = self.position[2] - self.target[2]
		
		# Calculer la distance
		self.distance = math.sqrt(dx*dx + dy*dy + dz*dz)
		
		# Calculer les angles (yaw et pitch)
		self.yaw = math.atan2(dx, dz)
		self.pitch = math.atan2(dy, math.sqrt(dx*dx + dz*dz))
	
	def _updatePosition(self) -> None:
		"""
		Met à jour la position basée sur les angles sphériques et la distance.
		"""
		# Convertir les coordonnées sphériques en cartésiennes
		x = self.target[0] + self.distance * math.sin(self.yaw) * math.cos(self.pitch)
		y = self.target[1] + self.distance * math.sin(self.pitch)
		z = self.target[2] + self.distance * math.cos(self.yaw) * math.cos(self.pitch)
		
		self.position = (x, y, z)
	
	def rotate(self, deltaYaw: float, deltaPitch: float) -> None:
		"""
		Fait tourner la caméra autour de la cible.
		
		Args:
			deltaYaw (float): Changement d'angle yaw en degrés
			deltaPitch (float): Changement d'angle pitch en degrés
		"""
		# Convertir en radians
		dyaw = math.radians(deltaYaw)
		dpitch = math.radians(deltaPitch)
		
		# Mettre à jour les angles
		self.yaw += dyaw
		self.pitch += dpitch
		
		# Limiter l'angle pitch pour éviter le retournement de la caméra
		self.pitch = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.pitch))
		
		# Mettre à jour la position
		self._updatePosition()
	
	def zoom(self, factor: float) -> None:
		"""
		Zoom la caméra en ajustant la distance à la cible.
		
		Args:
			factor (float): Facteur de zoom (>1 pour zoom arrière, <1 pour zoom avant)
		"""
		# Ajuster la distance
		self.distance *= factor
		
		# Limiter la distance pour éviter d'être trop proche ou trop loin
		self.distance = max(self.nearPlane * 2, min(self.farPlane * 0.5, self.distance))
		
		# Mettre à jour la position
		self._updatePosition()
	
	def pan(self, deltaX: float, deltaY: float) -> None:
		"""
		Déplace la caméra et la cible latéralement.
		
		Args:
			deltaX (float): Déplacement horizontal
			deltaY (float): Déplacement vertical
		"""
		# Calculer les vecteurs de la caméra
		forward = self._normalizeVector(self._subtractVectors(self.target, self.position))
		right = self._normalizeVector(self._crossProduct(forward, self.up))
		up = self._normalizeVector(self._crossProduct(right, forward))
		
		# Facteur d'échelle basé sur la distance
		scaleFactor = self.distance * 0.01
		
		# Calculer le déplacement
		rightMove = self._multiplyVector(right, deltaX * scaleFactor)
		upMove = self._multiplyVector(up, deltaY * scaleFactor)
		totalMove = self._addVectors(rightMove, upMove)
		
		# Appliquer le déplacement
		self.position = self._subtractVectors(self.position, totalMove)
		self.target = self._subtractVectors(self.target, totalMove)
		
		# Mettre à jour les angles
		self._updateSphericalCoordinates()
	
	def setTarget(self, target: Tuple[float, float, float]) -> None:
		"""
		Définit une nouvelle cible pour la caméra.
		
		Args:
			target (Tuple[float, float, float]): Nouvelle cible
		"""
		self.target = target
		
		# Mettre à jour la position
		self._updateSphericalCoordinates()
		self._updatePosition()
	
	def setPosition(self, position: Tuple[float, float, float]) -> None:
		"""
		Définit une nouvelle position pour la caméra.
		
		Args:
			position (Tuple[float, float, float]): Nouvelle position
		"""
		self.position = position
		
		# Mettre à jour les angles
		self._updateSphericalCoordinates()
	
	def reset(self, x: float = 0.0, y: float = 0.0) -> None:
		"""
		Réinitialise la caméra à une vue par défaut.
		
		Args:
			x (float): Position X cible
			y (float): Position Y cible
		"""
		# Définir une vue de dessus
		self.target = (x, y, 0.0)
		self.yaw = 0.0
		self.pitch = -math.pi/3  # Angle vers le bas
		self.distance = max(x, y) * 1.5  # Distance adaptée à la taille de la scène
		
		# Mettre à jour la position
		self._updatePosition()
	
	def getViewMatrix(self) -> List[List[float]]:
		"""
		Calcule la matrice de vue pour le rendu.
		
		Returns:
			List[List[float]]: Matrice de vue 4x4
		"""
		# Calculer les vecteurs de la caméra
		forward = self._normalizeVector(
			self._subtractVectors(self.target, self.position)
		)
		right = self._normalizeVector(
			self._crossProduct(forward, self.up)
		)
		newUp = self._normalizeVector(
			self._crossProduct(right, forward)
		)
		
		# Construire la matrice de rotation
		rotation = [
			[right[0], right[1], right[2], 0],
			[newUp[0], newUp[1], newUp[2], 0],
			[-forward[0], -forward[1], -forward[2], 0],
			[0, 0, 0, 1]
		]
		
		# Construire la matrice de translation
		translation = [
			[1, 0, 0, -self.position[0]],
			[0, 1, 0, -self.position[1]],
			[0, 0, 1, -self.position[2]],
			[0, 0, 0, 1]
		]
		
		# Combiner les matrices (rotation * translation)
		viewMatrix = self._multiplyMatrices(rotation, translation)
		
		return viewMatrix
	
	def getProjectionMatrix(self) -> List[List[float]]:
		"""
		Calcule la matrice de projection pour le rendu.
		
		Returns:
			List[List[float]]: Matrice de projection 4x4
		"""
		# Calculer les paramètres de la matrice de projection
		f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
		nf = 1.0 / (self.nearPlane - self.farPlane)
		
		# Construire la matrice de projection (perspective)
		return [
			[f / self.aspectRatio, 0, 0, 0],
			[0, f, 0, 0],
			[0, 0, (self.farPlane + self.nearPlane) * nf, 2 * self.farPlane * self.nearPlane * nf],
			[0, 0, -1, 0]
		]
	
	def getDirection(self) -> Tuple[float, float, float]:
		"""
		Obtient le vecteur de direction de la caméra.
		
		Returns:
			Tuple[float, float, float]: Vecteur de direction normalisé
		"""
		direction = self._subtractVectors(self.target, self.position)
		return self._normalizeVector(direction)
	
	def _addVectors(self, v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> Tuple[float, float, float]:
		"""
		Additionne deux vecteurs.
		
		Args:
			v1 (Tuple[float, float, float]): Premier vecteur
			v2 (Tuple[float, float, float]): Deuxième vecteur
			
		Returns:
			Tuple[float, float, float]: Vecteur résultant
		"""
		return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])
	
	def _subtractVectors(self, v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> Tuple[float, float, float]:
		"""
		Soustrait deux vecteurs.
		
		Args:
			v1 (Tuple[float, float, float]): Premier vecteur
			v2 (Tuple[float, float, float]): Deuxième vecteur
			
		Returns:
			Tuple[float, float, float]: Vecteur résultant
		"""
		return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])
	
	def _multiplyVector(self, v: Tuple[float, float, float], scalar: float) -> Tuple[float, float, float]:
		"""
		Multiplie un vecteur par un scalaire.
		
		Args:
			v (Tuple[float, float, float]): Vecteur
			scalar (float): Scalaire
			
		Returns:
			Tuple[float, float, float]: Vecteur résultant
		"""
		return (v[0] * scalar, v[1] * scalar, v[2] * scalar)
	
	def _dotProduct(self, v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
		"""
		Calcule le produit scalaire de deux vecteurs.
		
		Args:
			v1 (Tuple[float, float, float]): Premier vecteur
			v2 (Tuple[float, float, float]): Deuxième vecteur
			
		Returns:
			float: Produit scalaire
		"""
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
	
	def _crossProduct(self, v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> Tuple[float, float, float]:
		"""
		Calcule le produit vectoriel de deux vecteurs.
		
		Args:
			v1 (Tuple[float, float, float]): Premier vecteur
			v2 (Tuple[float, float, float]): Deuxième vecteur
			
		Returns:
			Tuple[float, float, float]: Produit vectoriel
		"""
		return (
			v1[1] * v2[2] - v1[2] * v2[1],
			v1[2] * v2[0] - v1[0] * v2[2],
			v1[0] * v2[1] - v1[1] * v2[0]
		)
	
	def _vectorLength(self, v: Tuple[float, float, float]) -> float:
		"""
		Calcule la longueur d'un vecteur.
		
		Args:
			v (Tuple[float, float, float]): Vecteur
			
		Returns:
			float: Longueur du vecteur
		"""
		return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
	
	def _normalizeVector(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
		"""
		Normalise un vecteur.
		
		Args:
			v (Tuple[float, float, float]): Vecteur
			
		Returns:
			Tuple[float, float, float]: Vecteur normalisé
		"""
		length = self._vectorLength(v)
		if length < 1e-8:
			return (0, 0, 0)
		return (v[0] / length, v[1] / length, v[2] / length)
	
	def _multiplyMatrices(self, m1: List[List[float]], m2: List[List[float]]) -> List[List[float]]:
		"""
		Multiplie deux matrices 4x4.
		
		Args:
			m1 (List[List[float]]): Première matrice
			m2 (List[List[float]]): Deuxième matrice
			
		Returns:
			List[List[float]]: Matrice résultante
		"""
		result = [[0.0 for _ in range(4)] for _ in range(4)]
		
		for i in range(4):
			for j in range(4):
				for k in range(4):
					result[i][j] += m1[i][k] * m2[k][j]
		
		return result
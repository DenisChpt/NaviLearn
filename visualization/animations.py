#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module gérant les animations pour la visualisation.
"""

import math
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable


class Animation:
	"""
	Classe de base pour une animation.
	"""
	
	def __init__(
		self,
		duration: float,
		looping: bool = False,
		onComplete: Optional[Callable[[], None]] = None
	) -> None:
		"""
		Initialise une animation.
		
		Args:
			duration (float): Durée de l'animation en secondes
			looping (bool): Si True, l'animation se répète
			onComplete (Optional[Callable[[], None]]): Fonction appelée à la fin de l'animation
		"""
		self.duration = max(0.001, duration)  # Éviter la division par zéro
		self.looping = looping
		self.onComplete = onComplete
		
		self.elapsedTime = 0.0
		self.isComplete = False
		self.isPaused = False
	
	def update(self, deltaTime: float) -> None:
		"""
		Met à jour l'animation.
		
		Args:
			deltaTime (float): Temps écoulé depuis la dernière mise à jour
		"""
		if self.isPaused or self.isComplete:
			return
			
		# Mettre à jour le temps écoulé
		self.elapsedTime += deltaTime
		
		# Vérifier si l'animation est terminée
		if self.elapsedTime >= self.duration:
			if self.looping:
				# Recommencer l'animation
				self.elapsedTime %= self.duration
			else:
				# Terminer l'animation
				self.elapsedTime = self.duration
				self.isComplete = True
				
				# Appeler le callback de fin si défini
				if self.onComplete:
					self.onComplete()
	
	def getProgress(self) -> float:
		"""
		Retourne la progression de l'animation (0-1).
		
		Returns:
			float: Progression de l'animation
		"""
		if self.duration <= 0:
			return 1.0
			
		return min(1.0, self.elapsedTime / self.duration)
	
	def getValue(self) -> float:
		"""
		Retourne la valeur actuelle de l'animation.
		À surcharger dans les classes dérivées.
		
		Returns:
			float: Valeur de l'animation
		"""
		return self.getProgress()
	
	def reset(self) -> None:
		"""
		Réinitialise l'animation.
		"""
		self.elapsedTime = 0.0
		self.isComplete = False
	
	def pause(self) -> None:
		"""
		Met l'animation en pause.
		"""
		self.isPaused = True
	
	def resume(self) -> None:
		"""
		Reprend l'animation après une pause.
		"""
		self.isPaused = False


class EasingAnimation(Animation):
	"""
	Animation avec fonction d'interpolation.
	"""
	
	# Types d'interpolation
	EASE_LINEAR = "linear"
	EASE_IN_QUAD = "in_quad"
	EASE_OUT_QUAD = "out_quad"
	EASE_IN_OUT_QUAD = "in_out_quad"
	EASE_IN_CUBIC = "in_cubic"
	EASE_OUT_CUBIC = "out_cubic"
	EASE_IN_OUT_CUBIC = "in_out_cubic"
	EASE_IN_SINE = "in_sine"
	EASE_OUT_SINE = "out_sine"
	EASE_IN_OUT_SINE = "in_out_sine"
	
	def __init__(
		self,
		startValue: float,
		endValue: float,
		duration: float,
		easingFunction: str = "linear",
		looping: bool = False,
		onComplete: Optional[Callable[[], None]] = None
	) -> None:
		"""
		Initialise une animation avec interpolation.
		
		Args:
			startValue (float): Valeur initiale
			endValue (float): Valeur finale
			duration (float): Durée de l'animation en secondes
			easingFunction (str): Fonction d'interpolation à utiliser
			looping (bool): Si True, l'animation se répète
			onComplete (Optional[Callable[[], None]]): Fonction appelée à la fin de l'animation
		"""
		super().__init__(duration, looping, onComplete)
		
		self.startValue = startValue
		self.endValue = endValue
		self.easingFunction = easingFunction
	
	def getValue(self) -> float:
		"""
		Retourne la valeur actuelle de l'animation.
		
		Returns:
			float: Valeur interpolée
		"""
		# Obtenir la progression (0-1)
		t = self.getProgress()
		
		# Appliquer la fonction d'interpolation
		t = self._applyEasing(t)
		
		# Interpoler entre les valeurs initiale et finale
		return self.startValue + (self.endValue - self.startValue) * t
	
	def _applyEasing(self, t: float) -> float:
		"""
		Applique la fonction d'interpolation.
		
		Args:
			t (float): Progression linéaire (0-1)
			
		Returns:
			float: Progression avec interpolation
		"""
		if self.easingFunction == self.EASE_LINEAR:
			return t
		elif self.easingFunction == self.EASE_IN_QUAD:
			return t * t
		elif self.easingFunction == self.EASE_OUT_QUAD:
			return t * (2 - t)
		elif self.easingFunction == self.EASE_IN_OUT_QUAD:
			return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
		elif self.easingFunction == self.EASE_IN_CUBIC:
			return t * t * t
		elif self.easingFunction == self.EASE_OUT_CUBIC:
			t = t - 1
			return t * t * t + 1
		elif self.easingFunction == self.EASE_IN_OUT_CUBIC:
			return 4 * t * t * t if t < 0.5 else (t - 1) * (2 * t - 2) * (2 * t - 2) + 1
		elif self.easingFunction == self.EASE_IN_SINE:
			return 1 - math.cos(t * math.pi / 2)
		elif self.easingFunction == self.EASE_OUT_SINE:
			return math.sin(t * math.pi / 2)
		elif self.easingFunction == self.EASE_IN_OUT_SINE:
			return 0.5 * (1 - math.cos(math.pi * t))
		
		# Par défaut, retour linéaire
		return t


class SequenceAnimation(Animation):
	"""
	Séquence d'animations jouées l'une après l'autre.
	"""
	
	def __init__(
		self,
		animations: List[Animation],
		looping: bool = False,
		onComplete: Optional[Callable[[], None]] = None
	) -> None:
		"""
		Initialise une séquence d'animations.
		
		Args:
			animations (List[Animation]): Liste des animations à jouer en séquence
			looping (bool): Si True, la séquence se répète
			onComplete (Optional[Callable[[], None]]): Fonction appelée à la fin de la séquence
		"""
		# Calculer la durée totale
		totalDuration = sum(anim.duration for anim in animations)
		
		super().__init__(totalDuration, looping, onComplete)
		
		self.animations = animations
		self.currentIndex = 0
	
	def update(self, deltaTime: float) -> None:
		"""
		Met à jour la séquence d'animations.
		
		Args:
			deltaTime (float): Temps écoulé depuis la dernière mise à jour
		"""
		if self.isPaused or self.isComplete:
			return
			
		# Mettre à jour l'animation courante
		currentAnimation = self.animations[self.currentIndex]
		currentAnimation.update(deltaTime)
		
		# Si l'animation courante est terminée, passer à la suivante
		if currentAnimation.isComplete:
			self.currentIndex += 1
			
			# Vérifier si la séquence est terminée
			if self.currentIndex >= len(self.animations):
				if self.looping:
					# Recommencer la séquence
					self._resetAllAnimations()
					self.currentIndex = 0
				else:
					# Terminer la séquence
					self.isComplete = True
					
					# Appeler le callback de fin si défini
					if self.onComplete:
						self.onComplete()
	
	def _resetAllAnimations(self) -> None:
		"""
		Réinitialise toutes les animations de la séquence.
		"""
		for animation in self.animations:
			animation.reset()
	
	def reset(self) -> None:
		"""
		Réinitialise la séquence d'animations.
		"""
		super().reset()
		self._resetAllAnimations()
		self.currentIndex = 0


class ParallelAnimation(Animation):
	"""
	Ensemble d'animations jouées en parallèle.
	"""
	
	def __init__(
		self,
		animations: List[Animation],
		looping: bool = False,
		onComplete: Optional[Callable[[], None]] = None,
		completeWhenAll: bool = True
	) -> None:
		"""
		Initialise un ensemble d'animations en parallèle.
		
		Args:
			animations (List[Animation]): Liste des animations à jouer en parallèle
			looping (bool): Si True, l'ensemble se répète
			onComplete (Optional[Callable[[], None]]): Fonction appelée à la fin
			completeWhenAll (bool): Si True, l'ensemble est terminé quand toutes les animations sont terminées
									Si False, l'ensemble est terminé quand au moins une animation est terminée
		"""
		# Calculer la durée (la plus longue des animations)
		maxDuration = max(anim.duration for anim in animations) if animations else 0.0
		
		super().__init__(maxDuration, looping, onComplete)
		
		self.animations = animations
		self.completeWhenAll = completeWhenAll
	
	def update(self, deltaTime: float) -> None:
		"""
		Met à jour l'ensemble d'animations en parallèle.
		
		Args:
			deltaTime (float): Temps écoulé depuis la dernière mise à jour
		"""
		if self.isPaused or self.isComplete:
			return
			
		# Mettre à jour toutes les animations
		for animation in self.animations:
			animation.update(deltaTime)
		
		# Vérifier si l'ensemble est terminé
		if self.completeWhenAll:
			# Terminé quand toutes les animations sont terminées
			if all(anim.isComplete for anim in self.animations):
				if self.looping:
					# Recommencer toutes les animations
					self._resetAllAnimations()
				else:
					# Terminer l'ensemble
					self.isComplete = True
					
					# Appeler le callback de fin si défini
					if self.onComplete:
						self.onComplete()
		else:
			# Terminé quand au moins une animation est terminée
			if any(anim.isComplete for anim in self.animations):
				if self.looping:
					# Recommencer toutes les animations
					self._resetAllAnimations()
				else:
					# Terminer l'ensemble
					self.isComplete = True
					
					# Appeler le callback de fin si défini
					if self.onComplete:
						self.onComplete()
	
	def _resetAllAnimations(self) -> None:
		"""
		Réinitialise toutes les animations de l'ensemble.
		"""
		for animation in self.animations:
			animation.reset()
	
	def reset(self) -> None:
		"""
		Réinitialise l'ensemble d'animations.
		"""
		super().reset()
		self._resetAllAnimations()


class AnimationManager:
	"""
	Gestionnaire d'animations pour l'application.
	"""
	
	def __init__(self) -> None:
		"""
		Initialise le gestionnaire d'animations.
		"""
		self.animations = {}
		self.nextAnimationId = 0
	
	def update(self, deltaTime: float) -> None:
		"""
		Met à jour toutes les animations actives.
		
		Args:
			deltaTime (float): Temps écoulé depuis la dernière mise à jour
		"""
		# Liste des animations à supprimer
		toRemove = []
		
		# Mettre à jour chaque animation
		for animId, animation in self.animations.items():
			animation.update(deltaTime)
			
			# Si l'animation est terminée et non répétée, la marquer pour suppression
			if animation.isComplete and not animation.looping:
				toRemove.append(animId)
		
		# Supprimer les animations terminées
		for animId in toRemove:
			del self.animations[animId]
	
	def addAnimation(self, animation: Animation) -> int:
		"""
		Ajoute une animation au gestionnaire.
		
		Args:
			animation (Animation): Animation à ajouter
			
		Returns:
			int: Identifiant de l'animation
		"""
		animId = self.nextAnimationId
		self.nextAnimationId += 1
		
		self.animations[animId] = animation
		
		return animId
	
	def removeAnimation(self, animationId: int) -> bool:
		"""
		Supprime une animation du gestionnaire.
		
		Args:
			animationId (int): Identifiant de l'animation à supprimer
			
		Returns:
			bool: True si l'animation a été supprimée, False sinon
		"""
		if animationId in self.animations:
			del self.animations[animationId]
			return True
		return False
	
	def pauseAnimation(self, animationId: int) -> bool:
		"""
		Met en pause une animation.
		
		Args:
			animationId (int): Identifiant de l'animation à mettre en pause
			
		Returns:
			bool: True si l'animation a été mise en pause, False sinon
		"""
		if animationId in self.animations:
			self.animations[animationId].pause()
			return True
		return False
	
	def resumeAnimation(self, animationId: int) -> bool:
		"""
		Reprend une animation mise en pause.
		
		Args:
			animationId (int): Identifiant de l'animation à reprendre
			
		Returns:
			bool: True si l'animation a été reprise, False sinon
		"""
		if animationId in self.animations:
			self.animations[animationId].resume()
			return True
		return False
	
	def resetAnimation(self, animationId: int) -> bool:
		"""
		Réinitialise une animation.
		
		Args:
			animationId (int): Identifiant de l'animation à réinitialiser
			
		Returns:
			bool: True si l'animation a été réinitialisée, False sinon
		"""
		if animationId in self.animations:
			self.animations[animationId].reset()
			return True
		return False
	
	def createEasing(
		self,
		startValue: float,
		endValue: float,
		duration: float,
		easingFunction: str = "linear",
		looping: bool = False,
		onComplete: Optional[Callable[[], None]] = None
	) -> int:
		"""
		Crée et ajoute une animation d'interpolation.
		
		Args:
			startValue (float): Valeur initiale
			endValue (float): Valeur finale
			duration (float): Durée de l'animation en secondes
			easingFunction (str): Fonction d'interpolation à utiliser
			looping (bool): Si True, l'animation se répète
			onComplete (Optional[Callable[[], None]]): Fonction appelée à la fin de l'animation
			
		Returns:
			int: Identifiant de l'animation
		"""
		animation = EasingAnimation(
			startValue=startValue,
			endValue=endValue,
			duration=duration,
			easingFunction=easingFunction,
			looping=looping,
			onComplete=onComplete
		)
		
		return self.addAnimation(animation)
	
	def createSequence(
		self,
		animations: List[Animation],
		looping: bool = False,
		onComplete: Optional[Callable[[], None]] = None
	) -> int:
		"""
		Crée et ajoute une séquence d'animations.
		
		Args:
			animations (List[Animation]): Liste des animations à jouer en séquence
			looping (bool): Si True, la séquence se répète
			onComplete (Optional[Callable[[], None]]): Fonction appelée à la fin de la séquence
			
		Returns:
			int: Identifiant de l'animation
		"""
		animation = SequenceAnimation(
			animations=animations,
			looping=looping,
			onComplete=onComplete
		)
		
		return self.addAnimation(animation)
	
	def createParallel(
		self,
		animations: List[Animation],
		looping: bool = False,
		onComplete: Optional[Callable[[], None]] = None,
		completeWhenAll: bool = True
	) -> int:
		"""
		Crée et ajoute un ensemble d'animations en parallèle.
		
		Args:
			animations (List[Animation]): Liste des animations à jouer en parallèle
			looping (bool): Si True, l'ensemble se répète
			onComplete (Optional[Callable[[], None]]): Fonction appelée à la fin
			completeWhenAll (bool): Si True, l'ensemble est terminé quand toutes les animations sont terminées
									Si False, l'ensemble est terminé quand au moins une animation est terminée
			
		Returns:
			int: Identifiant de l'animation
		"""
		animation = ParallelAnimation(
			animations=animations,
			looping=looping,
			onComplete=onComplete,
			completeWhenAll=completeWhenAll
		)
		
		return self.addAnimation(animation)
	
	def getAnimationValue(self, animationId: int) -> Optional[float]:
		"""
		Récupère la valeur actuelle d'une animation.
		
		Args:
			animationId (int): Identifiant de l'animation
			
		Returns:
			Optional[float]: Valeur actuelle de l'animation, ou None si l'animation n'existe pas
		"""
		if animationId in self.animations:
			return self.animations[animationId].getValue()
		return None
	
	def clearAll(self) -> None:
		"""
		Supprime toutes les animations.
		"""
		self.animations = {}
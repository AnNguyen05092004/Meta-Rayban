"""Perception — biến ảnh thành embedding cho CPM."""

from .embed import RealEmbedder, SyntheticEmbedder, get_embedder

__all__ = ["get_embedder", "SyntheticEmbedder", "RealEmbedder"]

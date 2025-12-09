# Zero-Shot-GNSS-Detection

his repository contains the implementation code for the paper
“A Zero-Shot Framework for GNSS Spoofing and Jamming Detection by Infusing Cluster Semantics into Large Language Models.”

AE.py: dual-branch CNN autoencoder for unsupervised latent feature extraction from multi-dimensional GNSS observations.

feature.py: computes statistical cluster representations based on the latent features.

prompt.py: generates cluster-level semantic descriptions for LLM-based zero-shot detection.

These scripts correspond to the offline part of the framework.
Additional modules (evaluation, visualization, and online inference workflow) will be released in future updates.

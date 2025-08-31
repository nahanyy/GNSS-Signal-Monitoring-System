# GNSS-Signal-Monitoring-System

This repository contains the implementation code for the system described in the paper “Design and Implementation of an LLM-Driven Real-Time GNSS Signal Monitoring System”.

The provided example demonstrates binary classification of GNSS signals (normal vs. spoofing). The GNSS jamming detection part follows a similar structure — you only need to replace it with the corresponding dataset.

Training dataset: derived from the paper “GNSS interference and spoofing dataset”.

VGG.py: model training code (VGG-based classifier).

artifacts_matrix_stats.py: generates Chinese descriptive text of matrix statistical features.

database.py: builds a knowledge base from training samples for later use in LLM-based retrieval-augmented generation (RAG).

Currently, the code provided here corresponds to the offline part of the system. Data preprocessing code and online system code will be released in future updates. 

# DomainAdapation
Domain Adaptation for Semantic Segmentation on Tankerships Using Deep Learning

Semantic segmentation involves the classification of each pixel in an image into a prede-
fined category. In the maritime industry, the automatic segmentation of various elements in
tanker ship imagery can play an important role in applications such as maintenance checks
and safety inspections.

This project addresses the challenge of semantic segmentation in the context of tanker ships,
leveraging a deep learning approach to develop a model capable of segmenting specific areas
within images of tanker ships. The neural network is trained on a dataset composed of
simulated images that provide clear, labeled semantic regions.

The more complex part of this project lies in domain adaptation - the refinement of the model
to transition from simulated images to actual, on-site photographs of tanker ships. Given the
scarcity of annotated real-world data in this domain, the project aims to employ gradient
reversal to bridge the gap between the simulated environment and real-world scenarios,
enabling the trained model to maintain high segmentation accuracy across different data
sources.

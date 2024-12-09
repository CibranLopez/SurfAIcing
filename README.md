# SurfAIcing: Crystal Surface Generator and Analyzer

SlabOptimization is a Python tool designed for the generation and analysis of crystal surfaces based on a given POSCAR (VASP input) structure. It utilizes the `pymatgen` library for crystal structure manipulation and slab generation, enabling researchers to explore and design materials with specific surface properties.

### Surface Energy and Stability

The energy of formation is a crucial parameter for understanding the stability of surfaces. Lower energy indicates greater stability. Surface energy influences various physical properties, such as catalytic activity, adsorption, and reactivity.

### Materials Design and Engineering

Researchers can use the tool to explore and design materials with specific surface properties for applications like catalysts, sensors, and electronic devices. Enables the identification of surfaces with desirable properties, guiding experimentalists in material synthesis.

### Computational Materials Discovery

Supports researchers in exploring a vast space of possible surface configurations, aiding in the discovery of novel materials. Useful for high-throughput calculations in the field of computational materials science.

### Interface Engineering

Important for understanding and engineering material interfaces, which play a significant role in the performance of devices such as solar cells and batteries.

## Features

- **Surface Generation:** Generates slabs up to a specified Miller index using `pymatgen`.
- **Energy Calculation:** Calculates the energy of formation for each slab based on a user-specified energy model or DFT calculations.
- **Sorting and Ranking:** Ranks slabs based on the energy of formation, providing a sorted list with relevant properties.
- **Customization:** Allows users to customize parameters such as minimum slab size, vacuum size, etc.
- **Physics Integration:** Considers surface energy, stability, and various physics-related properties.

Please be aware that the code is under active development, bug reports are welcomed in the GitHub issues!

## Installation

To download the repository and install the dependencies:

```bash
git clone https://github.com/CibranLopez/SlabOptimization.git
cd SlabOptimization
pip3 install -r requirements.txt
```

## Execution

An user-friendly jupyter notebook has been developed, which can be run locally with pytorch dependencies. It generates all slabs and performs geometrical relaxations.

## Authors

This project is being developed by:

 - Cibrán López Álvarez

## Contact, questions and contributing

If you have questions, please don't hesitate to reach out at: cibran.lopez@upc.edu

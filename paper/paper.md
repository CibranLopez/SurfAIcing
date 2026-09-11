---
title: 'SurfAIcing: A Python Workflow for Machine-Learning-Accelerated Surface Energetics, Adsorption, and Band Alignment'
tags:
  - Python
  - materials science
  - condensed matter physics
  - density functional theory
  - machine learning interatomic potentials
  - surface science
  - VASP
authors:
  - name: Cibrán López Álvarez
    orcid: 0000-0003-3949-5058
    affiliation: 1
affiliations:
  - name: Departament de Física, Universitat Politècnica de Catalunya (UPC), Barcelona, Spain
    index: 1
date: 11 September 2026
bibliography: paper.bib
---

# Summary

Surfaces and interfaces control many of the properties that make a material useful in a device: how a battery electrode degrades, how a solar absorber loses electrons to a contact layer, or how a catalyst activates a reactant molecule. Predicting these properties from first principles usually starts by cutting a bulk crystal along many possible orientations (Miller indices) and terminations, since a real sample exposes whichever surface is thermodynamically most stable, and comparing the resulting slabs' energies. `SurfAIcing` automates this screening step and several of the calculations that typically follow it: generating and ranking candidate surface slabs, building coherent heterostructure interfaces between two materials, computing hydrogen adsorption energies on a chosen surface, and aligning valence and conduction band edges to the vacuum level so that surfaces of different materials can be compared on a common energy scale. Each of these steps can fall back to a fast machine-learned interatomic potential (MLIAP) when high-accuracy density functional theory (DFT) results are not yet available, allowing hundreds of candidate slabs or terminations to be pre-screened before committing DFT time to the most promising ones.

# Statement of need

Surface-energy screening is combinatorially expensive: a single bulk structure can yield tens to hundreds of distinct slabs once several Miller indices, terminations, and thicknesses are enumerated, and each one traditionally requires an independent DFT relaxation before it is known which surfaces are thermodynamically favored. Existing tools such as `Surfaxe` [@brlec2021surfaxe], built on `pymatgen` [@ong2013pymatgen], streamline this workflow by automating slab generation, DFT input preparation, and post-processing — including the planar-averaged electrostatic potential used for band alignment, an analysis `SurfAIcing` also performs. These tools, however, assume that every candidate slab will ultimately be evaluated at the DFT level, which is often the actual bottleneck in practice.

`SurfAIcing` targets this bottleneck directly. It pairs the same `pymatgen`-based slab and coherent-interface generation [@zur1984lattice] with a machine-learned interatomic potential — the MACE-MP-0 foundation model [@batatia2024foundation], accessed through `ASE` [@larsen2017ase] — so that surface energies, relaxed geometries, and single-point energies for adsorption or band-alignment calculations can be obtained without DFT when a fast estimate is sufficient, and transparently swapped for DFT output (parsed from VASP's `vasprun.xml` and `LOCPOT` files) once it is available. The result is a single, consistent pipeline that a researcher can use to: (1) rank candidate surfaces of a material by formation energy; (2) build twisted or untwisted heterostructure interfaces between two slabs; (3) estimate hydrogen adsorption energies on a chosen surface; and (4) compute ionization potentials and electron affinities via vacuum-referenced band alignment. This is aimed at researchers studying photovoltaic absorbers, photoelectrochemical surfaces, solid-state electrolyte interfaces, and catalytic surfaces, where the relevant surface or interface termination is frequently not known in advance and a first, ML-accelerated pass materially reduces the number of DFT calculations needed to find it.

# Acknowledgements

The author thanks Claudio Cazorla and Edgardo Saucedo for supervision and helpful discussions during the development of this software.

# References

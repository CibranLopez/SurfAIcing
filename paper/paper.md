---
title: 'SurfAIcing: A Python Workflow for Machine-Learning-Accelerated Surface, Interface, and Adsorption Energetics'
tags:
  - Python
  - materials science
  - condensed matter physics
  - density functional theory
  - machine learning interatomic potentials
  - surface science
  - photocatalysis
  - photovoltaics
  - VASP
authors:
  - name: Cibrán López Álvarez
    orcid: 0000-0003-3949-5058
    affiliation: "1, 2"
  - name: Claudio Cazorla
    affiliation: "1, 2, 3"
affiliations:
  - name: Departament de Física, Universitat Politècnica de Catalunya, 08034 Barcelona, Spain
    index: 1
  - name: Research Center in Multiscale Science and Engineering, Universitat Politècnica de Catalunya, Campus Diagonal-Besòs, 08019 Barcelona, Spain
    index: 2
  - name: Institució Catalana de Recerca i Estudis Avançats (ICREA), Passeig Lluís Companys 23, 08010 Barcelona, Spain
    index: 3
date: 11 September 2026
bibliography: paper.bib
---

# Summary

Surfaces and interfaces control many of the properties that make a material useful in a device: how a battery electrode degrades, how a solar absorber loses electrons to a contact layer, or how a catalyst activates a reactant molecule. Predicting these properties from first principles usually starts by cutting a bulk crystal along many possible orientations (Miller indices) and terminations, since a real sample exposes whichever surface is thermodynamically most stable, and comparing the resulting slabs' energies. `SurfAIcing` automates this screening step and several of the calculations that typically build on it: generating and ranking candidate surface slabs, building coherent heterostructure interfaces between two materials, computing hydrogen adsorption energies on a chosen surface, and aligning valence and conduction band edges to the vacuum level so that surfaces of different materials, or different facets of the same material, can be compared on a common energy scale. Each of these steps can fall back to a fast machine-learned interatomic potential (MLIAP) when high-accuracy density functional theory (DFT) results are not yet available, allowing hundreds of candidate slabs, terminations, or interfaces to be pre-screened before committing DFT time to the most promising ones. The resulting quantities — surface stability, adsorption energetics, and vacuum-referenced band edges — are exactly the ingredients needed to design photocatalytic surfaces and photovoltaic device interfaces.

# Statement of need

Surface-energy screening is combinatorially expensive: a single bulk structure can yield tens to hundreds of distinct slabs once several Miller indices, terminations, and thicknesses are enumerated, and each one traditionally requires an independent DFT relaxation before it is known which surfaces are thermodynamically favored. Existing tools such as `Surfaxe` [@brlec2021surfaxe], built on `pymatgen` [@ong2013pymatgen], streamline this workflow by automating slab generation, DFT input preparation, and post-processing — including the planar-averaged electrostatic potential used for band alignment, an analysis `SurfAIcing` also performs. These tools, however, assume that every candidate slab will ultimately be evaluated at the DFT level, which is often the actual bottleneck in practice.

`SurfAIcing` targets this bottleneck directly. It pairs the same `pymatgen`-based slab and coherent-interface generation [@zur1984lattice] with a machine-learned interatomic potential — the MACE-MP-0 foundation model [@batatia2024foundation], accessed through `ASE` [@larsen2017ase] — so that surface energies, relaxed geometries, and single-point energies for adsorption or band-alignment calculations can be obtained without DFT when a fast estimate is sufficient, and transparently swapped for DFT output (parsed from VASP's [@kresse1996] `vasprun.xml` and `LOCPOT` files) once it is available.

This combination is aimed squarely at two application areas where the relevant surface or interface is rarely known in advance. In heterogeneous catalysis and photocatalysis, activity and selectivity are set by which facet is exposed, how strongly a reactant adsorbs on it, and whether its band edges straddle the redox potentials of the target reaction (e.g. proton or CO$_2$ reduction, water oxidation); screening these properties across many candidate terminations at DFT cost alone is usually prohibitive. In photovoltaics, device efficiency depends on aligning an absorber's band edges with those of its selective contacts, and increasingly on engineering the interface between two absorber compositions or phases directly, so that charge is extracted rather than lost to recombination. `SurfAIcing` was built to make both kinds of screening — surface/adsorption energetics and interface band alignment — tractable across the large compositional and orientational spaces that these problems typically involve. The result is a single, consistent pipeline that a researcher can use to: (1) rank candidate surfaces of a material by formation energy; (2) build heterostructure interfaces between two slabs; (3) estimate hydrogen adsorption energies on a chosen surface; and (4) compute ionization potentials and electron affinities via vacuum-referenced band alignment.

# Functionality

**Bulk relaxation and slab generation.** Starting from a single bulk structure file, `SurfAIcing` relaxes the bulk cell — either with the MLIAP (cell and atomic-position relaxation via an `ASE` `ExpCellFilter`/BFGS optimizer) or by accepting a pre-relaxed DFT structure — and then uses `pymatgen`'s `generate_all_slabs` to enumerate every symmetrically distinct slab up to a user-defined maximum Miller index, for user-set minimum slab and vacuum thicknesses, optionally repairing terminations that would otherwise expose broken bonds. For every resulting slab, `SurfAIcing` records its Miller index, termination shift, surface area, atom count, and polarity/symmetry flags, writes it out as a VASP structure file, and generates a matching k-point mesh at a user-specified reciprocal-space density, so each candidate surface arrives ready for either DFT or MLIAP evaluation.

**Hybrid DFT/MLIAP surface-energy ranking.** For each generated slab, `SurfAIcing` first looks for a completed DFT calculation; if none is present, it transparently falls back to relaxing the structure with the same MLIAP used for the bulk and taking a single-point energy, so partially and fully DFT-evaluated slabs can be mixed in the same ranking without any change to the analysis code. The surface energy of formation is then computed from the slab energy, the bulk energy per formula unit, and the surface area, following the standard thermodynamic definition (energy cost per unit area of exposing two free surfaces). All slabs generated from a given bulk structure are ranked from most to least stable and visualized together, so that the most likely surface terminations for a given material can be identified before any further, more expensive property is computed on them.

**Hydrogen adsorption energetics.** For a chosen surface, `SurfAIcing` compares several candidate adsorption configurations — different sites and orientations for one or more hydrogen atoms — matching each one to the bare reference surface by lattice vectors so that only configurations built on a consistent supercell are compared, and retaining the lowest-energy configuration. The atomic and molecular adsorption energies are then obtained by subtracting the (replica-scaled) bare-surface energy and the appropriate reference energy — a free hydrogen atom or half of a free H$_2$ molecule — from the adsorbed-configuration energy, using the same DFT/MLIAP-agnostic energy-lookup logic as the surface-ranking step.

**Vacuum-referenced band alignment.** `SurfAIcing` implements the planar/macroscopic-average electrostatic potential method for aligning electronic energy levels to the vacuum level. For the bulk structure, it reads the local electrostatic potential, computes its planar average along the surface normal, and reduces it to a macroscopic reference value, while the valence-band maximum (VBM) and band gap are extracted directly from the bulk band structure. For each slab, the planar-averaged potential is split into a bulk-like region and a vacuum region (discarding a user-set fraction near each region's edges to avoid boundary artifacts), and the mean potential in each region is computed. The VBM is then referenced to that slab's own vacuum level by combining the bulk VBM with the bulk-like and vacuum potential offsets, and adding the band gap gives the conduction-band minimum (CBM). This yields ionization potentials and electron affinities that can be directly compared across compositions, facets, and materials — the quantity used to assess photocatalytic redox-potential straddling and photovoltaic contact alignment in a recent application of the software [@lopez2026bandedge].

**Heterostructure and interface generation.** *(This section is a placeholder pending finalization of the workflow, at the author's request — the core interface-building step below is implemented; the automated selection/optimization procedure is still being developed.)* `SurfAIcing` builds coherent heterostructure interfaces between two independently generated slabs, for user-specified Miller indices and terminations on each side, using `pymatgen`'s lattice-matching algorithm [@zur1984lattice] and an optional in-plane twist rotation between the two components. [Further automated selection of interface termination, twist angle, and interlayer spacing — minimizing an MLIAP-computed interface energy — is in development and will be described here once finalized.]

`SurfAIcing` has already been used to generate the surfaces and compute the vacuum-referenced band alignments underlying a systematic study of band-edge engineering in pnictogen chalcohalide (MChX) solid solutions for photocatalytic and photovoltaic applications [@lopez2026bandedge], which itself builds on earlier machine-learning-aided first-principles predictions of MChX optoelectronic properties [@lopez2024mlaided].

# Acknowledgements

The authors thank Edgardo Saucedo for helpful discussions during the development of this software.

# References

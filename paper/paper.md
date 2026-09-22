---
title: 'SurfAIcing: A Python Package for Accelerated Surface and Interface Screening'
tags:
  - Python
  - Materials science
  - Condensed matter physics
  - Photocatalysis
  - Photovoltaics
authors:
  - name: Cibrán López
    orcid: 0000-0003-3949-5058
    corresponding: true
    affiliation: "1, 2"
  - name: Claudio Cazorla
    orcid: 0000-0002-6501-4513
    affiliation: "1, 2, 3"
affiliations:
 - name: Departament de Física, Universitat Politècnica de Catalunya, 08034 Barcelona, Spain
   index: 1
 - name: Research Center in Multiscale Science and Engineering, Universitat Politècnica de Catalunya, Campus Diagonal-Besòs, Av. Eduard Maristany 10-14, 08019 Barcelona, Spain
   index: 2
 - name: Institució Catalana de Recerca i Estudis Avançats (ICREA), Passeig Lluís Companys 23, 08010 Barcelona, Spain
   index: 3
date: 11 September 2026
bibliography: paper.bib
---

# Summary

Surfaces and interfaces determine many of the properties that make materials useful in devices: how a battery electrode degrades, how a solar absorber transfers electrons to a contact layer, or how a catalyst activates a reactant molecule. Predicting these properties from first principles often begins by identifying the most energetically stable surface orientations and terminations of a bulk crystal. `SurfAIcing` provides an automated workflow for generating and screening candidate surfaces and interfaces, as well as calculating properties such as hydrogen adsorption energies and vacuum-referenced band edges. For computationally demanding workflows involving hundreds of candidate slabs, terminations, or interfaces, the software can automatically use fast machine-learning interatomic potentials (ML-IAPs), reserving expensive first-principles calculations for the most promising candidates. The repository provides the tools to prepare, execute, and analyze these calculations within a single workflow.

# Statement of need

Surface-energy screening is combinatorially expensive. A single bulk structure can yield tens to hundreds of distinct slabs when multiple Miller indices and terminations are considered [@lopez2024mlaided]. Traditionally, each candidate requires an independent first-principles relaxation before the thermodynamically favored surfaces can be identified. Existing tools such as Surfaxe [@brlec2021surfaxe], built on pymatgen [@ong2013pymatgen], streamline this process by automating slab generation, first-principles input preparation, and post-processing. These tools, however, assume that every calculation will ultimately be evaluated at first-principles level, which is often the actual bottleneck in practice.

`SurfAIcing` targets this bottleneck directly [@lopez2026bandedge]. It combines pymatgen-based slab and coherent-interface generation [@zur1984lattice] with machine-learning interatomic potentials (ML-IAPs), such as the MACE-MP-0 foundation model [@batatia2024foundation], accessed through ASE [@larsen2017ase]. Surface energies, relaxed geometries, and single-point energies for molecular adsorption or band-alignment calculations can therefore be obtained using ML-IAPs when a fast estimate is sufficient. These calculations can subsequently be replaced by first-principles results, parsed from VASP's [@kresse1996] vasprun.xml and LOCPOT files, without changing the analysis workflow.

This combination is particularly useful in application areas where the relevant surface or interface is rarely known in advance. In heterogeneous catalysis and photocatalysis, activity and selectivity depend on the exposed facet, the strength of reactant adsorption, and whether the band edges straddle the redox potentials of the target reaction (e.g., proton or CO$_2$ reduction and water oxidation). Screening these properties across many candidate terminations at _ab initio_ cost alone is often prohibitive. In photovoltaics, device efficiency depends on aligning an absorber's band edges with those of its selective contacts. It also increasingly requires engineering interfaces between different absorber compositions or phases to promote charge extraction and suppress recombination.

`SurfAIcing` was developed to make both types of screening, surface and adsorption energetics, and interfaces and band alignments, tractable across the large compositional and orientational spaces involved. The resulting pipeline can be used to (1) rank candidate surfaces by formation energy; (2) construct heterostructure interfaces between two slabs; (3) estimate hydrogen adsorption energies on a selected surface; and (4) compute ionization potentials and electron affinities through vacuum-referenced band alignment.

# SurfAIcing

![**Overview of the SurfAIcing workflow.** Starting from a bulk crystal structure, SurfAIcing generates candidate slabs and evaluates their surface energies using either machine-learning interatomic potentials (ML-IAPs) or density functional theory (DFT). The resulting surfaces can then be screened for hydrogen adsorption, vacuum-referenced band alignment, or coherent heterostructure formation. \label{fig:flow-chart}](figure.pdf){width=100%}

**Bulk relaxation and slab generation.** Starting from a bulk structure file (\autoref{fig:flow-chart}), `SurfAIcing` either relaxes the bulk structure using an ML-IAP or accepts a pre-relaxed density functional theory (DFT) structure. For ML-IAP relaxation, cell and atomic positions are optimized using `ASE`'s `ExpCellFilter` and BFGS optimizer. The relaxed structure is then passed to pymatgen's generate_all_slabs to enumerate symmetrically distinct slabs up to a user-defined maximum Miller index and subject to user-defined minimum slab and vacuum thicknesses.

Terminations can optionally be repaired when the generated slabs would otherwise expose broken bonds. For each slab, `SurfAIcing` records the Miller index, termination shift, surface area, atom count, and polarity and symmetry flags. It also writes a VASP structure file and generates a matching k-point mesh at a user-specified reciprocal-space density, making each candidate ready for ML-IAP or DFT evaluation.

**Hybrid ML-IAP/DFT surface-energy ranking.**

For each generated slab, `SurfAIcing` first checks whether a completed DFT calculation is available. If not, it relaxes the structure using the same ML-IAP used for the bulk and evaluates a single-point energy. ML-IAP- and DFT-evaluated slabs can therefore be combined in the same ranking without modifying the analysis code. The surface formation energy is calculated from the slab energy, the bulk energy per formula unit, and the surface area, following the standard thermodynamic definition (the energy cost per unit area of exposing two free surfaces) [@boettger1994]:

$$
\gamma_{\text{surf}} = \frac{E_{\text{slab}} - N E_{\text{bulk}}}{2A},
$$

where $E_{\text{slab}}$ is the total energy of the relaxed slab, $E_{\text{bulk}}$ is the bulk energy per formula unit, $N$ is the number of bulk formula units contained in the slab, and $A$ is the slab's surface area; the factor of 2 accounts for the two free surfaces exposed by each slab. All slabs generated from a given bulk structure are ranked from most to least stable and visualized together, so that the most likely surface terminations for a given material can be identified before any further, more expensive property is computed on them.

**Hydrogen adsorption energetics.** For a selected surface, `SurfAIcing` evaluates multiple candidate hydrogen adsorption configurations, including different adsorption sites and orientations. Each configuration is matched to the bare reference surface through its lattice vectors, ensuring that all candidates use a consistent supercell. The lowest-energy configuration is retained.

Adsorption energies are calculated by subtracting the replica-scaled bare-surface energy and the appropriate reference energy. a free H atom or half the energy of a free H$_2$ molecule, from the energy of the adsorbed configuration [@norskov2004]:

$$
\Delta E_{\text{ads}} = E_{\text{conf}} - \left(n\, E_{\text{surf}} + E_{\text{ref}}\right),
$$

where $E_{\text{conf}}$ is the energy of the retained (lowest-energy) adsorption configuration, $E_{\text{surf}}$ is the energy of the bare reference surface, $n$ is the number of surface replicas contained in the adsorption supercell (obtained from the ratio of the two structures' unit-cell volumes), and $E_{\text{ref}}$ is either the energy of an isolated H atom or half the energy of an isolated H$_2$ molecule, depending on whether atomic or molecular adsorption is being evaluated. The same ML-IAP/DFT-agnostic energy-lookup mechanism used for surface ranking is applied here.

**Vacuum-referenced band alignment.** `SurfAIcing` aligns electronic energy levels to the vacuum level using the planar- and macroscopic-averaged electrostatic potential.

For the bulk structure, the local electrostatic potential is read and averaged along the surface normal to obtain a macroscopic reference value. The valence-band maximum (VBM) and band gap are extracted directly from the bulk band structure.

For each slab, the planar-averaged potential is divided into bulk-like and vacuum regions. A user-defined fraction near the boundaries of each region is discarded to avoid edge artifacts, and the mean potential is calculated in both regions. The slab VBM relative to vacuum is then obtained as [@liu2022piezo]:

$$
\text{VBM}_{\text{vac}} = \text{VBM}_{\text{DFT}} + \left(\bar{V}_{\text{bulk-like}} - \bar{V}_{\text{bulk}}\right) - \bar{V}_{\text{vac}}, \qquad \text{CBM}_{\text{vac}} = \text{VBM}_{\text{vac}} + E_{\text{gap}},
$$

where $\text{VBM}_{\text{DFT}}$ and $E_{\text{gap}}$ are the valence-band maximum and band gap read from the bulk band structure, $\bar{V}_{\text{bulk}}$ is the macroscopic-average potential obtained from the separate bulk calculation, and $\bar{V}_{\text{bulk-like}}$ and $\bar{V}_{\text{vac}}$ are the mean planar-averaged potentials in the bulk-like and vacuum regions of the slab, respectively. The parenthesised term realigns the slab's internal electrostatic reference to that of the bulk calculation, and subtracting $\bar{V}_{\text{vac}}$ then re-references the result to that slab's own vacuum level.

This yields ionization potentials and electron affinities that can be compared across compositions, facets, and materials, enabling the assessment of photocatalytic redox-potential alignment and photovoltaic contact alignment [@lopez2026bandedge].

**Heterostructure and interface generation.** `SurfAIcing` constructs coherent heterostructure interfaces directly from two independently generated slabs, such as those produced by the slab-generation workflow, without requiring the original bulk structures. The Zur–McGill lattice-matching algorithm [@zur1984lattice] is applied to the in-plane lattices of the two slabs to identify coincidence supercells.


The film can optionally be rotated about the interface normal before lattice matching to explore different twist angles. Because different rotations generally produce distinct sets of coincidence matches, this procedure also enables the generation of moiré or twisted-bilayer structures. For each angle, the lowest-strain matches are retained and ranked according to the von Mises strain of the coincidence-lattice deformation.

For each match, symmetry-distinct in-plane registries (lateral stacking offsets) are enumerated and combined with a small grid of interlayer gaps. The resulting candidate set is capped and evenly subsampled to a user-defined total when necessary.

Relaxation and ranking with an ML-IAP are optional. When enabled, `SurfAIcing` calculates the adhesion energy of formation [@bjorkman2012]:

$$
E_{\text{adhesion}} = \frac{E_{\text{interface}} - E_{\text{film}} - E_{\text{substrate}}}{A},
$$

where $E_{\text{interface}}$, $E_{\text{film}}$, and $E_{\text{substrate}}$ are, respectively, the total energies of the relaxed interface and of the isolated film and substrate slabs sharing the same (strained) interface cell, and $A$ is the interface area; unlike the surface formation energy above, no factor of 2 appears here, since each interface exposes a single film–substrate contact rather than two free surfaces. This uses the same ML-IAP/DFT-agnostic energy-lookup mechanism as the other modules. Interfaces can therefore either be ranked using ML-IAPs or passed directly to DFT.

# Acknowledgements

C.C. acknowledges support by MICIN/AEI/10.13039/501100011033 and ERDF/EU under the grants CNS2025-165467, PID2023-146623NB-I00 and PID2023-147469NB-C21 and by the Generalitat de Catalunya under the grants 2021SGR-00343, 2021SGR-01519 and 2021SGR-01411. Computational support was provided by the Red Española de Supercomputación under the grants FI-2024-1-0005, FI-2024-2-0003, FI-2024-3-0004, FI-2024-1-0025, FI-2024-2-0006, and FI-2025-1-0015. This work is part of the Maria de Maeztu Units of Excellence Programme CEX2023-001300-M funded by MCIN/AEI (10.13039/501100011033). C.L. acknowledges support from the Renew-PV European COST action (CA21148).

# References

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

Surfaces and interfaces determine how a battery electrode degrades, how a solar absorber passes electrons to a contact layer, or how a catalyst activates a molecule. Predicting these properties from first principles often begins by identifying the most stable surface orientations and terminations of a bulk crystal. `SurfAIcing` provides an automated workflow for generating and screening candidate surfaces and interfaces, as well as calculating properties such as hydrogen adsorption energies and vacuum-referenced band edges. For workflows involving hundreds of candidate slabs, terminations, or interfaces, the software can use fast machine-learning interatomic potentials (ML-IAPs), reserving expensive first-principles calculations for the most promising candidates. It is a Python library driven by Jupyter notebooks that prepares, executes, and analyzes these calculations within a single workflow.

# Statement of need

Surface-energy screening is combinatorially expensive. A single bulk structure can yield tens to hundreds of distinct slabs when several Miller indices and terminations are considered, and traditionally each candidate requires its own first-principles relaxation before the thermodynamically favored surfaces can be identified. `SurfAIcing` targets this bottleneck directly [@lopez2026bandedge; @lopez2026surfacephysicslowdimensionalpnictogen; @wurtzite2026screening]. Surface energies, relaxed geometries, and adsorption and adhesion energies are first obtained with an ML-IAP, such as the MACE-MPA-0 (medium) foundation model [@batatia2024foundation] accessed through ASE [@larsen2017ase], and can later be replaced by first-principles results parsed from VASP [@kresse1996] files without changing the analysis. Vacuum-referenced band alignment additionally needs first-principles (`LOCPOT`, `vasprun.xml`) data on the selected candidates.

This matters where the relevant surface or interface is rarely known in advance. In photocatalysis [@maggi2026q1D], activity depends on the exposed facet, on reactant adsorption, and on whether the band edges straddle the redox potentials of the target reaction. In photovoltaics [@lopez2024mlaided], efficiency depends on aligning the absorber's band edges with those of its contacts. The target audience is computational materials scientists who must survey many facets, terminations, and interfaces before committing first-principles resources. The pipeline (1) ranks candidate surfaces by formation energy, at 0 K or, optionally, by free energy at a chosen temperature; (2) constructs heterostructure interfaces between two slabs and ranks them by adhesion energy or free energy; (3) estimates atomic and molecular hydrogen adsorption energies or free energies; and (4) computes ionization potentials and electron affinities through vacuum-referenced band alignment.

# State of the field

Several packages cover parts of this workflow. pymatgen [@ong2013pymatgen] provides slab enumeration, adsorption-site construction (the basis of the adsorption workflow of @montoya2017), and Zur–McGill lattice matching [@zur1984lattice]. Surfaxe [@brlec2021surfaxe] automates slab generation, DFT inputs, slab/vacuum convergence tests, and post-processing, including planar-averaged potentials for ionization potentials and electron affinities. SurfFlow [@yalcin2023surfflow] automates DFT surface energies and Wulff shapes, handles asymmetric and non-stoichiometric slabs, and lowers the cost by pre-screening slabs with a bond-valence heuristic. MPInterfaces [@mathew2016mpinterfaces] automates the screening of interfaces, including heterostructures. All of them evaluate the surviving candidates with first-principles calculations (MPInterfaces also supports empirical energy models).

ML-IAPs offer a more direct pre-screen, but not as a drop-in replacement. Universal ML-IAPs are trained mostly on bulk data, and early models were found insufficiently accurate for surface energies without fine-tuning [@focassio2024]. A recent benchmark of 19 models on 36,718 slabs found that training data matters more than architecture, and that MACE-MPA reproduces cleavage energies with a mean absolute percentage error of 9.7% and identifies the most stable termination in 79% of materials [@mehdizadeh2025]. That benchmark covers only metallic compounds, Miller indices up to 1, and fixed DFT geometries, and its best model was least accurate for halogen-containing surfaces (errors of 39–80% for Cl, Br, and I). Since semiconductors, halides, and relaxed slabs are the cases of interest here, an ML-IAP ranking must be a screening layer that DFT can confirm. `SurfAIcing` provides the infrastructure for this: an energy layer shared by all modules that prefers DFT results whenever they exist and warns when a calculation mixes levels of theory, so that ML-IAP estimates select candidates and are then overwritten by DFT.

We build on these libraries rather than replace them: slab enumeration and lattice matching are delegated to pymatgen, and the potential alignment follows the planar-average approach also used by Surfaxe. We release a separate package because the new functionality (ML-IAP evaluation through ASE, one folder convention across the surface, adsorption, alignment, and interface stages, and level-of-theory bookkeeping) is a workflow layer outside the scope of those libraries.

# Software design

![**Overview of the `SurfAIcing` workflow.** Starting from a bulk crystal structure, `SurfAIcing` generates candidate slabs and evaluates their surface energies using either machine-learning interatomic potentials (ML-IAPs) or density functional theory (DFT), at 0 K or, optionally, as free energies from NPT molecular dynamics (dashed). The resulting surfaces can then be screened for hydrogen adsorption, vacuum-referenced band alignment, or coherent heterostructure formation. \label{fig:flow-chart}](figure.pdf){width=100%}

`SurfAIcing` is a Python library driven by five Jupyter notebooks, one per stage of \autoref{fig:flow-chart}. Screening many candidates cheaply and then refining a few led to four design decisions.

**Folders and files instead of a database.** Every slab, adsorption configuration, or interface is a folder of VASP-format files with the same layout: the inputs (`POSCAR`, `KPOINTS`) and the slab metadata in the root, the 0 K calculation in `relaxation/`, the NPT runs in `md/<T>K/`, and the band-alignment data in `alignment/`. Its energy is read, in order of priority, from `relaxation/vasprun.xml` (DFT), a precomputed single-point energy file, or an ML-IAP single point on `relaxation/CONTCAR` (a bare `POSCAR` is first relaxed at fixed cell). Replacing an ML-IAP estimate by DFT only requires copying the DFT output into `relaxation/`. The trade-off is the absence of a workflow manager or scheduler, which we accepted for hundreds to a few thousand candidates; it keeps the workflow transparent on HPC clusters where users already manage VASP jobs.

**Level-of-theory consistency is checked, not assumed.** Energy differences are meaningful only if every term comes from the same method. Each ML-IAP relaxation writes a provenance file (model, convergence, residual forces and stress); a bulk `POSCAR` is relaxed with the ML-IAP (cell and ions), so that the bulk energy is the minimum of the potential evaluating the slabs; and every ranking, adsorption, or adhesion calculation warns if its inputs mix DFT and ML-IAP energies or different models.

**Reuse first, new algorithms where needed.** Slab generation, Zur–McGill matching, and ASE optimizers are reused. We implemented a periodic Voronoi site finder for adsorption, automatic location of the bulk-like and vacuum regions of the planar potential (valid wherever the slab sits in the cell), and enumeration of interface registries with the twist angle of each match.

**Notebooks as interface, library for logic.** Parameters sit in notebook cells next to their output, which suits exploratory screening, while all logic lives in scriptable library functions.

**Limitations.** Asymmetric slabs are ranked by the average of their two terminations and non-stoichiometric slabs lack chemical potentials, so individual polar terminations (e.g., wurtzite (0001)) are not resolved; convergence with slab and vacuum thickness is not automated; ML-IAP evaluation uses MACE models; and band alignment requires DFT. The finite-temperature free energy uses an effective-harmonic description of the vibrations; it neglects electronic, configurational, and magnetic contributions (and any pressure–volume term, which vanishes at the zero pressure of the NPT runs), and it has not been validated against thermodynamic integration for surfaces. Unless the user fixes it, the cell of the slab is not constrained during the NPT run, so the vacuum can thin or the cell shear; the vacuum thickness and the length of $c$ are checked and flagged rather than corrected, whereas shear is not monitored. The hydrogen adsorption free energy references the gas at its vibrational level only (no translation, rotation, or pressure, which contribute about $-0.4$ eV at 300 K and 1 bar), and a fixed cell is at constant volume rather than at zero pressure.

**Slabs and surface-energy ranking.** Slabs are generated up to a maximum Miller index with user-defined slab and vacuum thicknesses, optionally repaired or restricted to symmetric terminations. Each is stored with its Miller index, shift, area, polarity, symmetry, and stoichiometry flags, and a k-point mesh. The surface energy is [@boettger1994]

$$
\gamma_{\text{surf}} = \frac{G_{\text{slab}} - N g_{\text{bulk}}}{2A},
$$

where $G_{\text{slab}}$ is the free energy of the relaxed slab, $g_{\text{bulk}}$ the bulk free energy per formula unit, $N$ the number of formula units in the slab, and $A$ the surface area; the factor of 2 accounts for the two free surfaces. The expression is the same at every level of the free energy. At 0 K, $G$ is the total energy $E$ of the relaxed structure (ML-IAP or DFT) and $\gamma_{\text{surf}}$ reduces to $(E_{\text{slab}} - N E_{\text{bulk}})/2A$; at finite temperature, $G$ includes the vibrational contribution described below. The adsorption and adhesion free energies below are built in the same way, at the 0 K level ($G = E$) or at finite temperature. Energies are reported in J/m$^2$, and the flags above are reported with the ranking.

**Finite-temperature surface free energy.** Optionally, the ranking is repeated with free energies. The bulk and every slab get one folder per MD temperature, containing the unrelaxed structure repeated until its lattice vectors reach a minimum length (15 Å by default; in-plane vectors only for slabs, so that the thickness and vacuum are unchanged) and the plan of the run. The structures are propagated in the NPT ensemble with a cell that is fully flexible by default (Nosé–Hoover thermostat and Parrinello–Rahman barostat in the scheme of Melchionna *et al.* [@melchionna1993], as implemented in ASE) with the ML-IAP, or with VASP, whose `vasprun.xml` is read instead. The free energy per atom follows the effective-harmonic scheme [@hellman2013],

$$
G(T) = U_0(T) + F_{\text{vib}}(T), \qquad U_0 = \frac{\langle U_{\text{MD}} \rangle}{N_{\text{at}}} - \frac{(3N_{\text{at}}-3)\,k_B T}{2N_{\text{at}}}, \\
F_{\text{vib}} = \int g(\omega)\left[\frac{\hbar\omega}{2} + k_B T \ln\left(1 - e^{-\hbar\omega/k_B T}\right)\right] d\omega,
$$

where $N_{\text{at}}$ is the number of atoms of the run, $\langle U_{\text{MD}} \rangle$ is the mean potential energy after discarding a thermalization time chosen by the user (2 ps by default), and the vibrational density of states $g(\omega)$, normalized to three modes per atom, is the Fourier transform of the mass-weighted velocity autocorrelation of the run, so that it contains the renormalization of the frequencies at that temperature. $G$ is built as a sum of terms (here $U_0$ and $F_{\text{vib}}$), so that further contributions can be added without changing the rest of the workflow; the 0 K energy does not enter it. The surface free energy is then

$$
\gamma(T) = \frac{G_{\text{slab}}(T) - N g_{\text{bulk}}(T)}{2A(T)},
$$

with $g_{\text{bulk}}(T)$ the bulk free energy per formula unit from the NPT run of the bulk and $A(T)$ the mean area of the slab cell in the run. The ranking can be evaluated at any operating temperature, not only at the MD temperatures: $F_{\text{vib}}$ is evaluated at the operating temperature with the vibrational spectrum of the MD run closest in temperature, which also provides $U_0$ and $A$. This assumes that the phonons, $U_0$ and $A$ do not change between the temperature of the run and the operating temperature, and a warning is attached when the two differ by more than 1 K. Every run is analyzed for equilibration, kinetic temperature, vacuum thickness, and change of the cell, and a warning is attached to each slab whose result may be affected. The 0 K and finite-temperature rankings are reported side by side, with their rank correlation.

**Hydrogen adsorption.** The surface normal is obtained from the lattice vectors, and the top layer of a chosen termination is decomposed with a periodic two-dimensional Voronoi tessellation: atoms define atop sites, pairs sharing a Voronoi ridge define bridge sites, and Voronoi vertices define hollow sites. Sites with equivalent surroundings are merged, and each H$_2$ site is generated perpendicular and parallel to the surface. All configurations and an isolated H$_2$ reference are relaxed at fixed cell, and the adsorption free energy of each configuration, per H atom or per H$_2$ molecule, is [@norskov2004]

$$
\Delta G_{\text{ads}} = G_{\text{conf}} - \left(n\, G_{\text{surf}} + \tfrac{n_{\text{H}}}{2}\, G_{\text{H}_2}\right),
$$

with $n$ the number of surface replicas in the configuration and $n_{\text{H}}$ its number of H atoms ($n_{\text{H}} = 1$ for dissociative adsorption, per H atom, and $n_{\text{H}} = 2$ for molecular adsorption, per H$_2$ molecule). At 0 K, $G = E$ and the lowest-energy configuration per adsorbate is reported. Optionally, $\Delta G_{\text{ads}}(T)$ is evaluated for all configurations with $G(T) = U_0 + F_{\text{vib}}$ from NPT runs of the surface and of every configuration, which start from the relaxed configuration and take the settings of the runs of the surface (including the thermalization time and the treatment of the cell). The contribution of every term of $G$ to $\Delta G_{\text{ads}}$ is reported, so that the change of the vibrational free energy on adsorption can be read directly. The molecular reference is the free energy of the stretching mode alone, $G_{\text{H}_2} = E + F_{\text{vib}}$, with the harmonic frequency of the relaxed molecule from the ML-IAP (or supplied by the user); its translation and rotation, and with them the pressure of the gas, are not included, although they can be added as further terms. The adsorption mode is verified from the H–H distance, since H$_2$ may dissociate, and every run is checked for H atoms that desorb, H$_2$ molecules that dissociate, and H atoms that hop between sites, for which the harmonic spectrum is not a good description.

**Vacuum-referenced band alignment.** From the first-principles planar-averaged electrostatic potential, the mean potential is computed in the bulk-like and vacuum regions of each slab (discarding a fraction at their edges) and in the bulk calculation. The band edges relative to vacuum are [@liu2022piezo]

$$
\text{VBM}_{\text{vac}} = \text{VBM}_{\text{DFT}} + \left(\bar{V}_{\text{bulk-like}} - \bar{V}_{\text{bulk}}\right) - \bar{V}_{\text{vac}}, \qquad \text{CBM}_{\text{vac}} = \text{VBM}_{\text{vac}} + E_{\text{gap}},
$$

where $\text{VBM}_{\text{DFT}}$ and $E_{\text{gap}}$ come from the bulk band structure. The ionization potential and electron affinity follow as $\text{IP} = -\text{VBM}_{\text{vac}}$ and $\text{EA} = -\text{CBM}_{\text{vac}}$ [@lopez2026bandedge].

**Heterostructures.** Coherent interfaces are built from two independently generated slabs, without the original bulk structures. The Zur–McGill algorithm [@zur1984lattice] identifies coincidence supercells, ranked by von Mises strain; each match fixes the relative in-plane orientation of the crystals, and its effective twist angle is stored. Symmetry-distinct lateral registries are combined with a small grid of interlayer gaps. If no match exists within the configured area and tolerances, no interface is generated and the lattice mismatch is reported. Optional ML-IAP relaxation and ranking use the adhesion free energy [@bjorkman2012]

$$
\Delta G_{\text{adhesion}} = \frac{G_{\text{interface}} - G_{\text{film}} - G_{\text{substrate}}}{A},
$$

where $G_{\text{film}}$ and $G_{\text{substrate}}$ are the free energies of the film and substrate atoms alone in the same strained interface cell, so that strain energy does not contribute, and $A$ is the interface area (no factor of 2: one film–substrate contact). At 0 K, $G = E$; optionally, $\Delta G_{\text{adhesion}}(T)$ is evaluated with $G(T) = U_0 + F_{\text{vib}}$ from NPT runs of the interface, the film, and the substrate, with $A$ the mean area of the interface cell in its run. The cell of these runs is fully flexible by default, with the option of fixing any of its directions, up to every lattice vector. With a flexible cell the film and the substrate on their own relax the strain that the interface cell imposes on them, so the strain energy of the film is then part of $\Delta G_{\text{adhesion}}(T)$, and a warning reports a reference whose area differs from that of the interface; fixing the in-plane lattice of the three runs removes the effect, at the price of no thermal expansion in the plane.

# Research impact statement

`SurfAIcing` was developed for, and is used in, our research on semiconductors for photovoltaic and photocatalytic applications. In the study of Bi/Sb chalcohalide solid solutions published in the *Journal of Physics: Energy* [@lopez2026bandedge], the vacuum-referenced band edges computed on the two most stable surfaces, (010) and (011), shift by up to 0.6 eV between terminations with nearly degenerate formation energies, which makes facet selection a design parameter comparable to chemical substitution. A companion study of the surface physics of these materials [@lopez2026surfacephysicslowdimensionalpnictogen] and a high-throughput surface screening of wurtzite semiconductors [@wurtzite2026screening] extend its use beyond a single materials family.
<!-- TODO(author): confirm that SurfAIcing produced the surface/band-edge results of each work (in particular @wurtzite2026screening, whose arXiv page could not be read when the citation was added, so check the sentence above against its content) and add specifics (number of slabs/interfaces screened, wall-clock saving vs DFT). -->

Each stage is reproducible from a notebook with documented inputs and outputs, and every ML-IAP result is accompanied by a provenance file recording the model and relaxation settings, so that published rankings can be regenerated and audited. <!-- TODO(author): add the repository URL, the tagged release (v1.0.0 + Zenodo DOI) and CONTRIBUTING.md once they exist. -->
<!-- TODO(author): strongest missing evidence is a benchmark on your own materials (e.g. BiSI, ZnO, GaN): ML-IAP vs DFT surface energies, rank correlation and whether the most stable termination is recovered, plus band edges. Add 2-3 sentences with the numbers here; it is the semiconductor/halide case the Mehdizadeh & Schindler benchmark does not cover. -->

# AI usage disclosure

Generative AI (Claude, Anthropic) was used to revise this manuscript for the current JOSS paper requirements: condensing the methods description and drafting the State of the field, Software design, and Research impact sections and this disclosure. The authors reviewed and edited all AI-assisted text, checked it against the code, and verified the added references against the publishers' records; they take full responsibility for the content. **[AUTHOR TO COMPLETE: state whether and how AI tools were used in writing the software or its documentation (tool, version, tasks) and how the output was verified, or state that none was used.]**

# Acknowledgements

C.C. acknowledges support by MICIN/AEI/10.13039/501100011033 and ERDF/EU under the grants CNS2025-165467, PID2023-146623NB-I00 and PID2023-147469NB-C21 and by the Generalitat de Catalunya under the grants 2021SGR-00343, 2021SGR-01519 and 2021SGR-01411. Computational support was provided by the Red Española de Supercomputación under the grants FI-2024-1-0005, FI-2024-2-0003, FI-2024-3-0004, FI-2024-1-0025, FI-2024-2-0006, and FI-2025-1-0015. This work is part of the Maria de Maeztu Units of Excellence Programme CEX2023-001300-M funded by MCIN/AEI (10.13039/501100011033). The authors acknowledge support from the Renew-PV European COST action (CA21148).

# References

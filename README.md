# Microparticle traction force microscopy
This repository provides implementations of traction reconstructions by the volume and surface method workflows in microparticle traction force microscopy (MP-TFM), together with example simulations, analysis scripts, and experimental workflows.

It accompanies our manuscript that describes the methodologies:

**Volume and surface methods for microparticle traction force microscopy: a computational and experimental comparison**

Simon Brauburger, Bastian K. Kraus, Tobias Walther, Tobias Abele, Kerstin Göpfrich and Ulrich S. Schwarz

---

We provide a runner script for an exemplary simulated Hertzian contact scenario (applying both the volume and surface methods), together with the Matlab files for the FIDVC, and example data and results.

We also include a JupyterLab notebook applying the surface method to our experimental data from DNA hydrogel microparticles (DNA-HMPs) compressed from the top.

---

## References

**Volume method**:

E. Mohagheghian, J. Luo, J. Chen, G. Chaudhary, J. Chen, J. Sun, R. H. Ewoldt and N. Wang, Nature Communications, 2018, 9, 1878.

**Surface method**:

D. Vorselen, Y. Wang, M. M. de Jesus, P. K. Shah, M. J. Footer, M. Huse, W. Cai and J. A. Theriot, Nature Communications, 2020, 11, 20.

**FIDVC algorithm** (slightly adapted for our analyses, therefore the Matlab files are included in the repo):

E. Bar-Kochba, J. Toyjanova, E. Andrews, K.-S. Kim and C. Franck, Experimental Mechanics, 2015, 55, 261–274.

F. Lab, FranckLab/FIDVC, 2015, https://github.com/FranckLab/FIDVC.

**ShElastic package** (functions for spherical harmonics, required for our scripts):

Y. Wang, X. Zhang and W. Cai, Journal of the Mechanics and Physics of Solids, 2019, 126, 151–167.

Y. Wang, ShElastic: Case06, 2021, https://github.com/yfwang09/ShElastic/blob/fc64cce8ea9a955d98fc3184e821235dff931f3d/examples/Case06-Hydrogel_deformation_test_case_with_penalty.ipynb.

---

For analyses of experimental data, we used custom Python scripts for preprocessing in the volume method to recover suitable nanoparticle image pairs as explained in our manuscript. For experimental evaluations with the surface method, we used GeoV on DNA-HMP outlines generated in Fiji (ImageJ) to reconstruct deformed surfaces as .ply files as described in our manuscript.

**GeoV**:

Y. Dreher, J. Niessner, A. Fink and K. Göpfrich, Advanced Intelligent Systems, 2023, 5, 2300170.

GeoV, Jakob Niessner, Yannik Dreher and Kerstin Göpfrich, 2022, https://github.com/jabrucohee/GeoV/edit/main/README.md.

**Fiji (ImageJ)**:

C. A. Schneider, W. S. Rasband and K. W. Eliceiri, Nature Methods, 2012, 9, 671-675.

Fiji, https://fiji.sc/.

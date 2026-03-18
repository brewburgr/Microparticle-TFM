# Matlab Files – FIDVC Displacement Field Analysis

This directory contains all files needed to run the FIDVC (Fast Iterative Digital Volume Correlation) algorithm in Matlab to generate a 3D displacement field from a pair of volumetric images (undeformed and deformed). The results are saved as `u_profile.mat` and `mesh_profile.mat` for use in the main Python runner script.

Image pairs are expected to be labeled `00*.mat` (reference/undeformed) and `01*.mat` (deformed). A synthetic image pair for a standardized Hertzian contact scenario is included, but it can also be created by the Hertzian contact runner script.

## Contents

- `exampleRunFile.m` – Main script to run the FIDVC on an image pair
- `funIDVC.m`, `DVC.m`, `IDVC.m` – Core FIDVC algorithm functions
- `addDisplacements.m`, `checkConvergenceSSD.m`, `filterDisplacements.m`, `removeOutliers.m`, `volumeMapping.m` – Supporting FIDVC functions
- `Supplemental MFiles/` – Supplementary functions required by the FIDVC:
  - `inpaint_nans3.m` – 3D NaN inpainting function
  - `mirt3D_mexinterp.m` / `mirt3D_mexinterp.cpp` – Fast 3D linear interpolation (MEX-based); a pre-compiled Windows binary (`mirt3D_mexinterp.mexw64`) is included
- `00Referencepicture*.mat` / `01Deformedpicture*.mat` – Synthetic Hertzian contact image pair

## Requirements

We used the following Matlab version on Linux:

23.2.0.2409890 (R2023b) Update 3

A compatible C compiler is required for the MEX compilation of `mirt3D_mexinterp.cpp`. See [Mathworks supported compilers](https://mathworks.com/support/requirements/supported-compilers.html) for your Matlab version and OS.

On **Windows**, a pre-compiled MEX binary (`mirt3D_mexinterp.mexw64`) is already included and no manual compilation is needed.

On **Linux** and **Mac**, `mirt3D_mexinterp.m` will automatically compile `mirt3D_mexinterp.cpp` the first time the function is called, provided a supported C compiler is installed and configured (e.g. via `mex -setup`).

## Setting up the Supplemental MFiles

Add the `Supplemental MFiles` directory to your Matlab path so that `inpaint_nans3` and `mirt3D_mexinterp` are accessible:

```matlab
addpath('path/to/Matlab files/Supplemental MFiles')
```

Or add it permanently via **Home → Set Path** in the Matlab GUI.

## Running the FIDVC

1. Open `exampleRunFile.m` in Matlab.
2. Update the `baseDir` variable to the absolute path of this `Matlab files` directory on your system:
   ```matlab
   baseDir = '/path/to/Matlab files';
   ```
3. Ensure the `Supplemental MFiles` directory is on the Matlab path (see above).
4. Run `exampleRunFile.m`. Matlab will automatically set the working directory if prompted.

The script will produce two output files in the directory specified by `baseDir`:

- `u_profile.mat` – 3D displacement field
- `mesh_profile.mat` – Corresponding mesh grid points

These files are used by `Hertzian_example_runner.py` for further analysis with the volume method.

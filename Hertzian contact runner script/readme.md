To run the main script `Hertzian_example_runner.py` (volume and surface method for simulated Hertzian contact), you need to download the following files, keeping relative paths between the files intact:

- `Hertzian_example_runner.py`
- `environment_master.yml`
- `ShElastic/`

## Setting up the conda environment

First, install the conda environment from the `environment_master.yml` file. The environment name will be `master`.

**Linux:**
```bash
conda env create -f environment_master.yml
conda activate master
```

**Windows** (run in Anaconda Prompt):
```
conda env create -f environment_master.yml
conda activate master
```

**Mac:**
```bash
conda env create -f environment_master.yml
conda activate master
```

> **Note:** The `environment_master.yml` file was generated on Linux. On Windows and Mac, some platform-specific packages may cause the environment creation to fail. If this happens, you may need to remove incompatible packages from the file or install the core dependencies manually.

## Running the script

Open and run `Hertzian_example_runner.py`. The analysis will run from the following relative path:

```
ShElastic/examples/Data_Hertzian_contact
```

This directory already contains results from a standardized run for both methods, including 3D traction plots, error evaluations, and profile visualizations.

Both `u_profile.mat` and `mesh_profile.mat` files for volume method evaluations are also present. These can alternatively be obtained by analyzing the synthetic image pair (`00Referencepicture*.mat`, `01Deformedpicture*.mat`) with the Matlab FIDVC algorithm.

If `u_profile.mat` and `mesh_profile.mat` are not detected, the volume method branch of the script will instead re-generate the synthetic image pair for analysis by the FIDVC. In that case, run the volume method branch again once the resulting `u_profile.mat` and `mesh_profile.mat` files are available.

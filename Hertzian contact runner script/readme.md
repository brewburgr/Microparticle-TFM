To run the main script Hertzian_contact_runner.py (volume and surface method for simulated Hertzian contact), you need to download the following files, keeping relative paths between the files intact:

Hertzian_contact_runner.py
master.yml
ShElastic

First install the conda environment from the master.yml file:

Linux:

Windows:

Mac:

Then, open and run the Hertzian_contact_runner.py script. The analysis will run from this relative path:

ShElastic/examples/Data_Hertzian_contact

It already contains results from a standardized run for both methods, including 3D traction plots, error evaluations, and profile visualizations.

Also, both u_profile.mat and mesh_profile.mat files for volume method evaluations are present, which you could also obtain by analyzing the synthetic image pair (00Referencepicture*.mat, 01Deformedpicture*.mat) with the Matlab FIDVC algorithm.

If u_profile.mat and mesh_profile.mat are not detected, the volume method branch of the script will instead re-generate the synthetic image pair for analysis by the FIDVC, and you need to run the volume method branch again with the resulting u_profile.mat and mesh_profile.mat files.

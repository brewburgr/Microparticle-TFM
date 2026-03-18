To run the FIDVC algorithm in Matlab to generate a displacement field by image correlation of undeformed and deformed images (simulated or experimental, labeled 00*.mat and 01*.mat for reference/undeformed or deformed respectively), first download the directory.

It contains all .m scripts for the FIDVC, a synthetic image pair for a standardized Hertzian contact, as well as the supplementary .m files with functions inpaint_nans3 and mirt3D_mexinterp (inpainting nans and faster interpolation).

We used the following Matlab version on Linux:

23.2.0.2409890 (R2023b) Update 3

To run the script, open exampleRunFile.m and run it with the image pair in the same directory. Set the working directory in Matlab if it does not ask you to automatically. The FIDVC will run and output the u_profile.mat and mesh_profile.mat files in the Matlab files directory for further analysis with the main runner script.

%% Example run file for the IDVC - LDTFM

% This script has to be run with the .mat files created by the volume method
% workflow in Python (synthetic nanoparticle distributions in deformed and reference
% states) saved in the directory specified below, and it creates the displacement field
% profile u_profile.mat and according mesh mesh_profile.mat in the same
% directory by FIDVC. These can then be used for further analysis in Python.

% Matlab changelog for MP-TFM workflow:
% Changed filename = 'vol*.mat'; to filename = '*.mat' and moved all the
% files to the same folder so Matlab can find everything.
% There was an error between data types from mex and mwSize. Fixed it.
% There was an error because of incompatible array sizes in
% checkConvergenceSSD.m, fixed it by introducing an if-error-clause
% Implemented a fix with clearvars except (in refloop)
% Tuned the standard value for WindowSize (8 -> 4)

% VARIABLES OPTIONS
% -------------------------------------------------------------------------
%   filename: string for the filename prefix for the volumetric images in
%             the current directory.
%             Input options:
%             --- If image is not within a cell) ---
%             1) 'filename*.mat' or 'filename*'
%
%             --- If image is within a cell that contains multichannels ---
%             2) filename{1} = 'filename*.mat' or 'filename*' and
%                filename{2} = channel number containing images you want to
%                              run IDVC on.
%                (if the channel is not provided, i.e. length(filename) = 1
%                , then channel = 1
%
%   sSize: interrogation window (subset) size for the first iterations.
%          Must be 32,64,96, or 128 voxels and a three column
%          array (one for each dimension) or scalar (equal for all
%          dimensions).
%   incORcum: string that defines the method of running IDVC. Options:
%             cumulative (time0 -> time1, time0 -> time2, ...)
%             (Allowable inputs: 'c','cum','cumulative')
%             or
%             incremental (time0 -> time1, time1 -> time2, ...)
%             (Allowable inputs: 'i','inc','incremental')
%
% OUTPUTS
% -------------------------------------------------------------------------
%   u:  displacement field vector calculated from FIDVC. Format: cell array,
%      which is a 3D vector (components in x,y,z)  per each time point
%      (units are in voxels)
%         u{time}{1} = displacement in x-direction at t=time of size MxNxP
%         u{time}{2} = displacement in y-direction at t=time of size MxNxP
%         u{time}{3} = displacement in z-direction at t=time of size MxNxP
%   cc: peak values of the cross-correlation for each interrogation
%   dm: meshgrid spacing (8 by default)
%   m:  The grid points at which displacements are computed. The grid 
%        points locations are in the same format as 'u'.
% 
% NOTES
% -------------------------------------------------------------------------
% To run you need a compatible C compiler. Please see
% (http://www.mathworks.com/support/compilers/R2014a/index.html)
% 
% If used please cite:
% Bar-Kochba E., Toyjanova J., Andrews E., Kim K., Franck C. (2014) A fast 
% iterative digital volume correlation algorithm for large deformations. 
% Experimental Mechanics. doi: 10.1007/s11340-014-9874-2




clear; close all
sSize = [128 128 64];
incORcum = 'incremental';

% ---- Define directory with .mat input images, also used as output directory ----
baseDir = '/home/bq_bkraus/Desktop/Matlab/Matlab';

% ---- Define the images, starting with 00 (reference) or 01 (deformed) ----
file00 = dir(fullfile(baseDir, '00*.mat'));
file01 = dir(fullfile(baseDir, '01*.mat'));

% ---- Safety checks ----
if isempty(file00) || isempty(file01)
    error('Required image files not found in %s', baseDir);
end

% ---- Build full paths ----
filename = { ...
    fullfile(baseDir, file00(1).name), ...
    fullfile(baseDir, file01(1).name) };

% ---- Clean workspace ----
close all
clear u cc dm m

fprintf('Processing image pair in: %s\n', baseDir);

% ---- Run DVC ----
[u, cc, dm, m] = funIDVC(filename, sSize, incORcum);

% ---- Save results ----
save(fullfile(baseDir,'u_profile.mat'), 'u');
save(fullfile(baseDir,'mesh_profile.mat'), 'm');


% Ignoring the plotting functions in Matlab to use the outputs in Python.
disp('Saved resulting u_profile.mat and mesh_profile.mat, stopping.')
return


%% PLOTTING 
load('vol01.mat');
sizeI = size(vol{1});
plotIdx = cell(1,3);
for i = 1:3, plotIdx{i} = 1:dm:sizeI(i)+1; end

close all;
% plot IDVC results
figure;
for i = 1:3
    subplot(2,3,i);
    [~,h] = contourf(plotIdx{1}, plotIdx{2}, u{1}{i}(:,:,ceil(size(u{1}{i},3)/2)),25); colorbar
    set(h,'linestyle','none'); axis image
    title(['displacement component u_',num2str(i)])
    xlabel('X_1'); ylabel('X_2');
    
    subplot(2,3,i+3);
    [~,h] = contourf(plotIdx{1}, plotIdx{3}, squeeze(u{1}{i}(:,48,:))',25); colorbar
    set(h,'linestyle','none'); axis image
    xlabel('X_2'); ylabel('X_3');
    
end




function I = volumeMapping(varargin)
%VOLUMEMAPPING images for 3-D images
% I = volumeMapping(I0,m,u0) symmetrically warps undeformed
% and deformed 3-D images by the displacement field from previous iteration
% using trilinear interpolation.
%
% INPUTS
% -------------------------------------------------------------------------
%   I0: cell containing the undeformed, I0{1}, and deformed, I0{2} 3-D
%       images
%   m: meshgrid of the displacement field
%   u0: displacement field vector defined at every meshgrid point with 
%      spacing dm. Format: cell array, each containing a 3D matrix 
%         (components in x,y,z)
%         u0{1} = displacement in x-direction
%         u0{2} = displacement in y-direction
%         u0{3} = displacement in z-direction
%         u0{4} = magnitude
%
% OUTPUTS
% -------------------------------------------------------------------------
%   I: cell containing the symmetrically warped images of I0{1} and I0{2}
%
% NOTES
% -------------------------------------------------------------------------
% interp3FastMex and mirt3D-mexinterp are used to perform the interpolation
% since they are faster and less memory demanding than MATLAB's interp3
% function. To run you need a compatible C compiler. Please see
% (http://www.mathworks.com/support/compilers/R2014a/index.html)
%
% For more information please see section 2.2.
% If used please cite:
% Bar-Kochba E., Toyjanova J., Andrews E., Kim K., Franck C. (2014) A fast 
% iterative digital volume correlation algorithm for large deformations. 
% Experimental Mechanics. doi: 10.1007/s11340-014-9874-2

[I0,m0,u0] = parseInputs(varargin{:});


idx = cell(1,3); idx_u0 = cell(1,3);

for i = 1:3
    %disp(m0{i}(1))
    idx{i} = m0{i}(1):(m0{i}(end) - 1); % get index for valid indices of image
    %disp(idx{i})
    idx_u0{i} = linspace(1,size(u0{1},i),length(idx{i})+1); % get index for displacement meshgrid
    idx_u0{i} = idx_u0{i}(1:length(idx{i}));
    u0{i} = double(u0{i}*0.5);
end
%return

[m_u0{2}, m_u0{1}, m_u0{3}] = ndgrid(idx_u0{:});
[m{2}, m{1}, m{3}] = ndgrid(idx{:});

u = cell(1,3);


disp('Starting mirt3D_mexinterp')
for i = 1:3, u{i} = mirt3D_mexinterp(u0{i}, m_u0{1}, m_u0{2}, m_u0{3}); end

fprintf('\n Calculated u{i} for i=1,2,3')

% disp('Stopping after mirt3D_mexinterp')
% return

% disp(u{1}(1:10))
% disp(u{2}(1:10))
% disp(u{3}(1:10))
% disp(u{2})

% Check whether the x, y and z entries are equal everywhere:
%all(u{1}(:) == u{2}(:))
%all(u{1}(:) == u{3}(:))

%all(u{1}(:) == 0) % Check if all x-components are zero
%all(u{2}(:) == 0) % Check if all y-components are zero
%all(u{3}(:) == 0) % Check if all z-components are zero


%% Warp images (see eq. 8)
%disp('Stopping before cellfun...')
%return
disp('Using cellfun to calculate mForward, mBackward')

% The following code applies the anonymous function x+y to the cell array
% inputs m and u, giving m{i}+u{i} for each element as a cell array output
mForward = cellfun(@(x,y) x + y, m, u, 'UniformOutput',false);
disp('Calculated mForward')

%disp('Stopping before mBackward')
%return

mBackward = cellfun(@(x,y) x - y, m, u, 'UniformOutput',false);
disp('Calculated mBackward')

%disp('Stopping before mirt3D_mexinterp')
%return

% interpolate volume based on the deformation
I{1} = mirt3D_mexinterp(I0{1},  mForward{1}, mForward{2}, mForward{3});
%disp('Calculated I{1}')
%disp('Stopping')
%return
I{2} = mirt3D_mexinterp(I0{2},  mBackward{1}, mBackward{2}, mBackward{3});



end

%% ========================================================================
function varargout = parseInputs(varargin)
I0 = varargin{1};
m  = varargin{2};
u0 = varargin{3};

% convert I0 to double.  Other datatypes will produce rounding errors
I0 = cellfun(@double, I0, 'UniformOutput', false);

varargout{      1} = I0;
varargout{end + 1} = m;
varargout{end + 1} = u0;

end

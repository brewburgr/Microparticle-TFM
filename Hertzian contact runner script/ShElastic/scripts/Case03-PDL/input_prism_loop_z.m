addpath('dd3d')

% a circular prismatic loop

R_void = 1;                        % void radius


for loop = [0, 1]

    if loop == 0
        % smaller loop
        Sname = 's';
        zlen  = 41;                         % discretization of loop position
        zlist = linspace(0.67, 4.0, zlen);   % nodes on the loop
        rho0 = 0.75*R_void;                 % radius of the loop
        a_b_list = [40, 200, 400, 4000, 40000];
    else
        % larger loop
        Sname = 'l';
        zlist = [-4:0.2:-0.4, -0.2:0.04:0.2, 0.4:0.2:4]; 
        zlen = length(zlist);
        rho0 = 1.20*R_void;                 % radius of the loop
        a_b_list = [40, ];
    end

    Ngrid = 40;                         % gridding on the void surface (2NxN)
    ndis  = 50;                         % discretize the loop

    for a_b = a_b_list
        bmag = R_void/a_b;              % magnitude of burger's vector
        disp(bmag)

        MU = 1;
        NU = 1./3;

        % figure; hold on;
        for i = 1:zlen
            % disp(i)
            z0 = R_void*zlist(i);           % height of the loop
            disp(z0)

            % construct the loop
            phi0 = [0:ndis-1]'*2*pi/ndis;   % [=] ndis x 1
            rn = [rho0*cos(phi0), ...       % [=] ndis x 4 (x,y,z,nodetype)
                  rho0*sin(phi0), ones(ndis,1)*z0, zeros(ndis,1)];
            link_id = [1:ndis; [2:ndis,1]]';        % [=] ndis x 2 (n1,n2)
            burg = repmat([0, 0, 1], ndis, 1)*bmag; % [=] ndis x 3 (bx,by,bz)
            dl = circshift(rn(:, 1:end-1), -1, 1) ...
               - rn(:, 1:end-1);                    % r2 - r1 [=] ndis x 3
            dlnorm = sqrt(sum(dl.^2, 2));           % [=] ndis x 1
            xi = dl./repmat(dlnorm, [1, 3]);        % [=] ndis x 3
            %disp(xi)
            % slip = vecnorm(cross(burg, xi, 2), 2, 2);
            nvec = cross(burg, xi, 2);              % b x xi [=] ndis x 3
            %disp(nvec)
            nnorm= sqrt(sum(nvec.^2, 2));           % [=] ndis x 1
            %disp(nnorm)
            slip = nvec./repmat(nnorm, [1, 3]);     % [=] ndis x 3 (nx,ny,nz)
            %disp(slip)
            links= [link_id, burg, slip];   % [=] ndis x 8

            % dislocation core (a=0.1, Ec=0)
            a = 0.1;
            appliedstress = 0.0*MU*eye(3);

            % save the data for ShElastic
            Susrfile = ['../../testdata/data_PDL/Susr',Sname,'_z', num2str(z0, '%.5f'), '_ab', num2str(a_b), '.mat'];
            [ Tusr, Xgrid ] = tractionBC(MU, NU, a, rn, links, 0, appliedstress, R_void, Ngrid, 1);
            save(Susrfile, 'Tusr', 'Xgrid', 'burg', 'rn')

        end

    end

end

function [ Tusr, Xgrid ] = tractionBC(MU, NU, a, rn, links, linkid, appliedstress, R, Ngrid, gridtype)
% R = radius of the void
% Ngrid = gridding of the void surface

    if gridtype == 0
    % regular grid
        phi   = ([0:Ngrid-1]+0.0)/Ngrid*(pi);
        theta = ([0:2*Ngrid-1]+0.0)/(2*Ngrid)*(2*pi);
    elseif gridtype == 1
    % GLQ grid
        GLQgrid = py.pyshtools.expand.GLQGridCoord(Ngrid); 
        lat = double(py.array.array('d', py.numpy.nditer(GLQgrid{1})));
        lon = double(py.array.array('d', py.numpy.nditer(GLQgrid{2})));
        phi = deg2rad(90-lat); theta = deg2rad(lon);
    else
        disp('unknown gridtype (0 for regular grid, 1 for GLQgrid)')
        return
    end

    [THETA,PHI] = meshgrid(theta,phi);
    Z = R*cos(PHI); X = R*sin(PHI).*cos(THETA); Y = R*sin(PHI).*sin(THETA);
    sizeZ = size(Z);
    XX = zeros([sizeZ, 3]);
    XX(:,:,1)=X; XX(:,:,2)=Y; XX(:,:,3)=Z;
    N = -XX / R;

    % construct segment list
    segments = constructsegmentlist(rn, links);
    if linkid ~= 0
        segments = segments(linkid,:);
    end
    x = reshape(XX, [sizeZ(1)*sizeZ(2), 3]);
    b1 = segments(:, 3:5);
    x1 = segments(:, 6:8);
    x2 = segments(:, 9:11);
    
    % evaluate stress on the surface points
    svec = reshape(FieldPointStress(x, x1, x2, b1, a, MU, NU), [sizeZ, 6]);
    smat = zeros([sizeZ, 3, 3]);
    smat(:, :, 1, 1) = svec(:, :, 1); smat(:, :, 1, 2) = svec(:, :, 4); smat(:, :, 1, 3) = svec(:, :, 6);
    smat(:, :, 2, 1) = svec(:, :, 4); smat(:, :, 2, 2) = svec(:, :, 2); smat(:, :, 2, 3) = svec(:, :, 5);
    smat(:, :, 3, 1) = svec(:, :, 6); smat(:, :, 3, 2) = svec(:, :, 5); smat(:, :, 3, 3) = svec(:, :, 3);
    Sext = repmat(reshape(appliedstress, [1,1,3,3]), [sizeZ, 1, 1]);
    % disp(Sext(1,1,:,:))
    S = smat + Sext;

    % calculate traction boundary condition on the surface
    Tusr = S; %squeeze(sum(S .* repmat(N, [1,1,1,3]), 3));
    Xgrid= XX;
end

function segments = constructsegmentlist(rn, links)
    [LINKMAX, ~] = size(links);

    segments = zeros(LINKMAX,14);
    nseg = 0;
    for i = 1:LINKMAX
        n0 = links(i,1);
        n1 = links(i,2);
        if ((n0~=0) && (n1~=0))
            nseg = nseg+1;
            segments(nseg,:) = [links(i,1:5), rn(n0,1:3),...
                                rn(n1,1:3), links(i,6:8)];
        end
    end
    segments = segments(1:nseg,:);
end

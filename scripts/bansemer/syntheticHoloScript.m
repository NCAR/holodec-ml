function x = syntheticHoloScript(op, fn, dist)
    %% Generate holograms of random particles

    %Get the default settings, mostly consistent with HOLODEC
    if ~exist('op')
        op = Fraunhofer();

        %Make any changes to defaults here
        op.nHolograms = 100;  %Number of holograms to make
        op.NParticles = 3;  %Particles per hologram
        op.Nx = 600;        %Image dimensions
        op.Ny = 400;
        op.Dpmin = 20e-6;   %Particle size min/max
        op.Dpmax = 70e-6;
    end
    if length(op.NParticles) == 2  %Range of NParticles given
        RandomizeNParticles = 1;
        %Randomize the number of particles in the range
        RandomParticleCount = randi(op.NParticles,1,op.nHolograms);
        TotalNParticles = sum(RandomParticleCount);
    else
        RandomizeNParticles = 0;
        TotalNParticles = op.nHolograms * op.NParticles;
    end
    
    if exist('dist')
        havedistributions = 1;
        if RandomizeNParticles == 1
            disp('Incompatible options'); 
            return
        end
    else
        havedistributions = 0;
    end

    %% Set up the netCDF file
    if ~exist('fn')
        fn='synthetic_holograms.nc'
    end
    cmode = netcdf.getConstant('NETCDF4');
    if exist(fn)  %Clobber not supported, use system
       delete(fn)
    end
    ncid = netcdf.create(fn, cmode);

    %Dimensions
    hologram_dimid = netcdf.defDim(ncid, 'hologram_number', op.nHolograms);
    xsize_dimid = netcdf.defDim(ncid, 'xsize', op.Nx);
    ysize_dimid = netcdf.defDim(ncid, 'ysize', op.Ny);
    particle_dimid = netcdf.defDim(ncid, 'particle', TotalNParticles);

    %Variables and attributes
    ncdfprops = {'hid', 'Hologram index (1-based, first hologram index = 1)', 'unitless';
        'd', 'Particle Diameter', 'microns';
        'x', 'Particle x-position (origin at center)', 'microns';
        'y', 'Particle y-position (origin at center)', 'microns';
        'z', 'Particle z-position (origin at sensor)', 'microns'};
    if havedistributions
        numprops = length(ncdfprops);
        ncdfprops(numprops+1:numprops+4, :) = {'n0', 'Gamma distribution N0', 'm^-4';
            'mu', 'Gamma distribution mu', 'unitless';
            'lam', 'Gamma distribution lambda', '1/m';
            'distid', 'Gamma distribution index', 'unitless'};
    end
    for i = 1:length(ncdfprops)
        varid = netcdf.defVar(ncid, ncdfprops{i,1}, 'float', particle_dimid);
        netcdf.putAtt(ncid, varid, 'longname', ncdfprops{i,2});
        netcdf.putAtt(ncid, varid, 'units', ncdfprops{i,3});
    end
    varid = netcdf.defVar(ncid, 'image', 'NC_UBYTE', [ysize_dimid, xsize_dimid, hologram_dimid]);
    netcdf.putAtt(ncid, varid, 'longname', 'Hologram image');
    %Turn on compression for the image, saves about 50% file size
    netcdf.defVarDeflate(ncid, varid, true, true, 5); 
    %Specify chunking in 2D only, speeds up the process
    netcdf.defVarChunking(ncid, varid, 'CHUNKED', [op.Ny/10, op.Nx/10, 1]);

    %Write options to global attributes
    tags = fieldnames(op);
    varid = netcdf.getConstant('NC_GLOBAL');  %Need to set for global atts
    netcdf.putAtt(ncid, varid, 'DateCreated', date);
    for i = 1:length(tags)
       attdata = getfield(op, tags{i});
       if islogical(attdata) attdata = uint8(attdata); end  %netCDF req
       if length(attdata) == 1
           netcdf.putAtt(ncid, varid, tags{i}, attdata);
       end
    end

    netcdf.endDef(ncid);  %Enter data mode

    %Save variable ids for use when writing
    xvarid = netcdf.inqVarID(ncid, 'x');
    yvarid = netcdf.inqVarID(ncid, 'y');
    zvarid = netcdf.inqVarID(ncid, 'z');
    dvarid = netcdf.inqVarID(ncid, 'd');
    hvarid = netcdf.inqVarID(ncid, 'hid');
    ivarid = netcdf.inqVarID(ncid, 'image');
    if havedistributions
        n0varid = netcdf.inqVarID(ncid, 'n0');
        lamvarid = netcdf.inqVarID(ncid, 'lam');
        muvarid = netcdf.inqVarID(ncid, 'mu');
        distidvarid = netcdf.inqVarID(ncid, 'distid');
    end
    %% Make the holograms
    for i = 1:op.nHolograms
        if RandomizeNParticles == 1
            op.NParticles = RandomParticleCount(i);  %Set particles per hologram
        end

        % Generate random particle sizes and positions
        op.particles = randomParticles(op);
        if havedistributions
            %this may crash for NParticles > 1, untested
            clear n0 lam mu distid  %Will rebuild these arrays below, need to be right size for nCDF write
            for j = 1:op.NParticles
                op.particles(j).Dp = dist.Dp((i-1)*op.NParticles+j);
                n0(j) = dist.n0((i-1)*op.NParticles+j);
                lam(j) = dist.lam((i-1)*op.NParticles+j);
                mu(j) = dist.mu((i-1)*op.NParticles+j);
                distid(j) = dist.id((i-1)*op.NParticles+j);
            end
        end

        % Make the hologram
        holoField = perfectHolo(op);

        % Make the hologram image as seen by the camera
        img = syntheticHolo(op);
        img2write = uint8(img);  %reshape for netCDF
        clearCache(op);    %Need to clear, otherwise Fraunhofer.m keeps

        %ifn = 'last_hologram.png'
        %imwrite(uint8(img),ifn);

        %Write data to netCDF.  This breaks in parallel mode, fixable?
        if RandomizeNParticles == 1
            offset = sum(RandomParticleCount(1:(i-1))); %This works even when i-1 = 0
        else
            offset = (i-1)*op.NParticles;
        end
        netcdf.putVar(ncid, xvarid, offset, op.NParticles, [op.particles.x]*1e6) 
        netcdf.putVar(ncid, yvarid, offset, op.NParticles, [op.particles.y]*1e6) 
        netcdf.putVar(ncid, zvarid, offset, op.NParticles, [op.particles.z]*1e6) 
        netcdf.putVar(ncid, dvarid, offset, op.NParticles, [op.particles.Dp]*1e6) 
        netcdf.putVar(ncid, hvarid, offset, op.NParticles, zeros(1,op.NParticles)+i) 
        netcdf.putVar(ncid, ivarid, [0, 0, (i-1)], [op.Ny, op.Nx, 1], img2write) 
        if havedistributions
            netcdf.putVar(ncid, n0varid, offset, op.NParticles, n0)
            netcdf.putVar(ncid, lamvarid, offset, op.NParticles, lam)
            netcdf.putVar(ncid, muvarid, offset, op.NParticles, mu)
            netcdf.putVar(ncid, distidvarid, offset, op.NParticles, distid)
        end
        if mod(i,10) == 0
           disp([i,op.nHolograms])
        end
    end  

    netcdf.close(ncid);
    x=i;  %Output number of holograms written
end

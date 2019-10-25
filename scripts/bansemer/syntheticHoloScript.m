%% Generate holograms of random particles

%Get the default settings, mostly consistent with HOLODEC
op = Fraunhofer();

%Make any changes to defaults here
op.nHolograms = 100;  %Number of holograms to make
op.NParticles = 1;  %Particles per hologram
op.Nx = 600;        %Image dimensions
op.Ny = 400;
op.Dpmin = 20e-6;   %Particle size min/max
op.Dpmax = 70e-6;

%% Set up the netCDF file
cmode = netcdf.getConstant('NETCDF4');
cmode = bitor(cmode,netcdf.getConstant('CLOBBER'));
ncid = netcdf.create('temp.nc', cmode);

%Dimensions
hologram_dimid = netcdf.defDim(ncid, 'hologram_number', op.nHolograms);
xsize_dimid = netcdf.defDim(ncid, 'xsize', op.Nx);
ysize_dimid = netcdf.defDim(ncid, 'ysize', op.Ny);
particle_dimid = netcdf.defDim(ncid, 'particle', op.nHolograms * op.NParticles);

%Variables and attributes
ncdfprops = {'hid', 'Hologram index (1-based, first hologram index = 1)', 'unitless';
    'd', 'Particle Diameter', 'microns';
    'x', 'Particle x-position (origin at center)', 'microns';
    'y', 'Particle y-position (origin at center)', 'microns';
    'z', 'Particle z-position (origin at sensor)', 'microns'};
for i = 1:length(ncdfprops)
    varid = netcdf.defVar(ncid, ncdfprops{i,1}, 'float', particle_dimid);
    netcdf.putAtt(ncid, varid, 'longname', ncdfprops{i,2});
    netcdf.putAtt(ncid, varid, 'units', ncdfprops{i,3});
end
varid = netcdf.defVar(ncid, 'image', 'NC_UBYTE', [hologram_dimid, xsize_dimid, ysize_dimid]);
netcdf.putAtt(ncid, varid, 'longname', 'Hologram image');

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

%% Make the holograms
for i = 1:op.nHolograms
    % Generate random particles
    op.particles = randomParticles(op);
    %particleData = op.particles;

    % Make the hologram
    holoField = perfectHolo(op);

    % Make the hologram image as seen by the camera
    img = syntheticHolo(op);
    img2write = uint8(transpose(img));  %reshape for netCDF
    clearCache(op);    %Need to clear, otherwise Fraunhofer.m keeps
    
    %ifn = 'last_hologram.png'
    %imwrite(uint8(img),ifn);
    varid = netcdf.inqVarID(ncid, 'x');
    netcdf.putVar(ncid, varid, (i-1)*op.NParticles, op.NParticles, op.particles.x*1e6) 
    varid = netcdf.inqVarID(ncid, 'y');
    netcdf.putVar(ncid, varid, (i-1)*op.NParticles, op.NParticles, op.particles.y*1e6) 
    varid = netcdf.inqVarID(ncid, 'z');
    netcdf.putVar(ncid, varid, (i-1)*op.NParticles, op.NParticles, op.particles.z*1e6) 
    varid = netcdf.inqVarID(ncid, 'd');
    netcdf.putVar(ncid, varid, (i-1)*op.NParticles, op.NParticles, op.particles.Dp*1e6) 
    varid = netcdf.inqVarID(ncid, 'hid');
    netcdf.putVar(ncid, varid, (i-1)*op.NParticles, op.NParticles, i) 
    varid = netcdf.inqVarID(ncid, 'image');
    netcdf.putVar(ncid, varid, [(i-1), 0, 0], [1, op.Nx, op.Ny], img2write) 

    if mod(i,10) == 0
        disp([i,op.nHolograms])
    end
end  

netcdf.close(ncid);

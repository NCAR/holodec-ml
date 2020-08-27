%Script to make holograms using gamma distributions
%One each for single-particle, and for three-particle.
%Each set will contain training, test, validation, and private files.

%Set up parameters for the datasets
NParticles=[20,20,20,20]; %, 3,3,3,3, 3,3,3,3];
NHolograms=[5000,1000,1000,1000]; %, 50000,10000,10000,10000, 50000,10000,10000,10000];
fn=["synthetic_holograms_20particle_gamma_training.nc"
    "synthetic_holograms_20particle_gamma_test.nc"
    "synthetic_holograms_20particle_gamma_validation.nc"
    "synthetic_holograms_20particle_gamma_private.nc"];

rng('shuffle');   %Set random seed

%Create each dataset
for i = 1:length(NHolograms)
    op = Fraunhofer();    %This needs to be re-initialized each time
    op.nHolograms = NHolograms(i);  %Number of holograms to make
    op.NParticles = NParticles(i);  %Particles per hologram
    op.Nx = 600;          %Image dimensions
    op.Ny = 400;
    op.Dpmin = 0e-6;     %Particle size min/max
    op.Dpmax = 10000e-6;

    clear Dp mu lam id n0
    xvalues = [0:0.1:1000]./1e6;   %Size array up to 1 mm
    
    %Get preset diameters
    n0range = 1.0e8;  %Essentially arbitrary for this 
    lamrange = 100000; %50000 gives relativley sharp peak, replicator fit=28161.0.  100000 sharper, peak at ~10*mu;
    murange = [1,5];   %Range between 1 and 5 gives peak < 100um, replicator = 3.95
    nperdist = op.NParticles;  %Number of particles to generate for each distribution, set to NParticles if each hologram unique
    
    for j = 1:op.nHolograms*op.NParticles/nperdist
       istart =  (j-1)*nperdist+1;
       istop = j*nperdist;
       mu(istart:istop) = (murange(2)-murange(1)).*rand(1) + murange(1);
       n0(istart:istop) = n0range(1);
       lam(istart:istop) = lamrange(1);
       id(istart:istop) = j;
       %Might eventually move Dp determination into syntheticHoloScript.m 
       %for more flexibility
       yvalues = n0(istart).*exp(-lam(istart).*xvalues).*xvalues.^mu(istart);
       Dp(istart:istop) = randomf(xvalues, yvalues, nperdist);
    end
    
    dist = struct('Dp',Dp,'lam',lam,'n0',n0,'mu',mu,'id',id);

    x = syntheticHoloScript(op, fn(i), dist);
end


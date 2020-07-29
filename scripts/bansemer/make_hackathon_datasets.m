%Script to make holograms for the 2020 AI4ESS Hackathon
%One each for single-particle, and for three-particle.
%Each set will contain training, test, validation, and private files.

%Set up parameters for the datasets
NParticles=[1,1,1,1, 3,3,3,3];
NHolograms=[50000,10000,10000,10000, 50000,10000,10000,10000];
fn=["synthetic_holograms_1particle_training.nc"
    "synthetic_holograms_1particle_test.nc"
    "synthetic_holograms_1particle_validation.nc"
    "synthetic_holograms_1particle_private.nc"
    "synthetic_holograms_3particle_training.nc"
    "synthetic_holograms_3particle_test.nc"
    "synthetic_holograms_3particle_validation.nc"
    "synthetic_holograms_3particle_private.nc"];

rng('shuffle');   %Set random seed

%Create each dataset
for i = 6:length(NParticles)
    op = Fraunhofer();    %This needs to be re-initialized each time
    op.nHolograms = NHolograms(i);  %Number of holograms to make
    op.NParticles = NParticles(i);  %Particles per hologram
    op.Nx = 600;          %Image dimensions
    op.Ny = 400;
    op.Dpmin = 20e-6;     %Particle size min/max
    op.Dpmax = 70e-6;

    x = syntheticHoloScript(op, fn(i));
end

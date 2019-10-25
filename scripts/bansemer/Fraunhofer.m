% Class intended to make a Fraunhofer hologram

%    Copyright (C) 2011 Matt Beals and Jacob Fugal and Oliver Schlenczek
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    When using this program as part of research that results in
%    publication, acknowledgement is appreciated. The following citation is
%    appropriate to include: 
%
%    Fugal, J. P., T. J. Schulz, and R. A. Shaw, 2009: Practical methods
%    for automated reconstruction and characterization of particles in
%    digital inline holograms, Meas. Sci. Technol., 20, 075501,
%    doi:10.1088/0957-0233/20/7/075501.  
%
%    Funding for development of holoViewer at Michigan Tech (Houghton,
%    Michigan, USA) provided by the US National Science Foundation (US-NSF)
%    Graduate Research Fellowship Program, and NASA's Earth Science
%    Fellowship Program. Funding for development at the National Center for
%    Atmospheric Research (NCAR, Boulder, Colorado, USA) provided by the
%    US-NSF. Funding for development at the Max Planck Institute for
%    Chemistry (MPI-Chemistry, Mainz, Germany) and the Johannes Gutenberg
%    University of Mainz (Uni-Mainz, Mainz, Germany), provided by
%    MPI-Chemistry, the US-NSF, and the Deutsche Forschungsgesellschaft
%    (DFG, the German Science Foundation). 
%
%    Please address questions or bug reports to Jacob Fugal at
%    fugalscientific (at) gmail (dot) com 

%    Previous update: 2012/07/11 by Oliver
%    Property added to allow artificial darkening of the particle field as
%    a simulation for dark edges in the entire hologram (this parameter is
%    called relPartInt (added in makeHologram in line 188))

%    Last update: 2012/07/26 by Oliver
%    Now it is allowed to generate polydisperse holograms using a vector
%    for particle number, Dpmin and Dpmax. If not commented out, the
%    numbers are Poisson random numbers. 


classdef Fraunhofer < handle

properties    
   
    %prtclCon      = 10;
    Nx            = 1200;       %Image dimensions
    Ny            = 800;
    dx            = 2.96e-6;    %Resolution
    dy            = 2.96e-6;
    lambda        = 0.355e-6;   %Wavelength
    Dpmin         = 30e-6;      %Particle diameter min/max
    Dpmax         = 30e-6;
    zMin          = 14e-3;      %Window locations
    zMax           = 158e-3;
    NParticles      = 10;
    Nbits           = 8;        % Number of bits in the hologram
    ccdelevel       = 2e4;      % The average number of electrons for the CCD (Kodak KAI 29050 sensor)
    gainFactor      = 24;       % Camera gain in dB (amplification)
    
  
    readNoiseSTD    = 12;       % This value is also taken from Kodak
    gaussCornCon    = 1;        % The contrast at the corner of the gaussian 
                                % beam as a fraction of maximum intensity
                                % About 50% is very realistic compared to
                                % our real holograms in the lab!
    lowPassFiltCutoff = 2500;    % The spatial frequency (in cycles per unit length) below which
                                % noise is preserved.
    fracBeamDistort = 0.02;      % The fraction of the circular Gaussian beam profile that the
                                % the noise is multiplied into. 0 is no
                                % noise, 1 is 100% noise multiplied in.
                                % Now we use 2% low-pass noise of the laser
                                % beam which is common for most
                                % experimental setups
    
    should_beam    = true;     % Should we apply a circular Gaussian beam profile
    should_noise   = true;     % Should we apply shot and read noise as the camera would
    should_lnoise  = false;    % Should we add some noise to the Gaussian beam profile so
                               % one profile pulse is not the same as the
                               % next
                               
    partRelInt     = 0.5;      % How much should the particle field be darkened to simulate low light conditions?    
    nHolograms     = 1;        % Number of holograms in a series (default is 1)
    
    savePath       = '';       % The path where the holograms and data should be saved
    fileNameBase   = '';       % First part of the file name
    
    makeRandParts  = true;     % Should we generate random particles? If not, we load an existing particle data file.
    loadFnBase     = '';       % First part of the particle files to load
    loadPath       = '';       % The path to load the existing particle data from
    should_random  = true;     % Should the particle numbers be randomized via Poisson statistics?
    should_extend_domain = false; % Should the domain be extended by factor 2 in both dimensions?
    
    superSamplingRate = 1;
    
    config_handle
end 

properties (Dependent)
    xs, ys;
end    

properties
    particles;          % The structure array of particle positions and sizes
    should_cache=true; % Should we cache the hologram field that is made
    cached_hologram_field; % The cached field   
    signalStrength; % The signal strength
    noiseLevel; % The noise level as sqrt(sum(difference_image.^2))
end

methods
    function this = Fraunhofer( configpath )
        if exist('configpath','var')
            if ischar(configpath)
                this.config_handle = config(configpath);
            elseif isa(configpath, 'config')
                this.config_handle = configpath;
            end
            this.config_handle.dynamic     
            this.prtclCon = this.config_handle.dynamic.prtclCon;
            this.dx = this.config_handle.dx;
            this.dy = this.config_handle.dy;
            this.lambda = this.config_handle.lambda;
            this.zMin = this.config_handle.zMin;
            this.zMax = this.config_handle.zMax;
            this.Nbits = this.config_handle.dynamic.Nbits;
            this.NParticles = round(str2double(this.prtclCon)*1e6*(this.zMax-this.zMin)*str2double(this.Nx)*this.dx*str2double(this.Ny)*this.dy);
            this.Nx = this.config_handle.dynamic.Nx;
            this.Ny = this.config_handle.dynamic.Ny;
            this.Dpmin = this.config_handle.dynamic.Dpmin;
            this.Dpmax = this.config_handle.dynamic.Dpmax;
            this.should_lnoise = this.config_handle.dynamic.should_lnoise;
            this.should_beam = eval(this.config_handle.dynamic.should_beam);
            this.partRelInt = this.config_handle.dynamic.partRelInt;
            this.nHolograms = this.config_handle.dynamic.nHolograms;
            this.gaussCornCon = this.config_handle.dynamic.gaussCornCon;
            this.fileNameBase = this.config_handle.dynamic.fileNameBase;
            this.savePath = this.config_handle.dynamic.savePath;
            this.loadPath = this.config_handle.dynamic.loadPath;
            this.loadFnBase = this.config_handle.dynamic.loadFnBase;
            this.makeRandParts = this.config_handle.dynamic.makeRandParts;
            this.superSamplingRate = eval(this.config_handle.dynamic.superSamplingRate);
            this.gainFactor = eval(this.config_handle.dynamic.gainFactor);
            % Gain factor is given in dB, so convert the ccd e- level to the
            % desired value (to account for heavy amplification)
            this.ccdelevel = this.ccdelevel/10^(this.gainFactor/10);
        else
            this.config_handle = [];
        end

    end
            
    function particle_field = randomParticles(this,NParticles)
            if nargin == 1, NParticles = this.NParticles; end
            % There has to be a distinction if NParticles is a vector or a
            % scalar
            %NParticles
            if length(NParticles)<2
                if abs(this.Dpmax-this.Dpmin)<eps
                    diameters = ones(NParticles,1)*this.Dpmin;
                else
            diameters = (this.Dpmax-this.Dpmin)*rand(NParticles,1) + this.Dpmin;
                end
                
                            %Generate a random grid of positions
            particleX       = (max(this.xs)-min(this.xs))*rand(NParticles,1)+min(this.xs);  % Particle position
            particleY       = (max(this.ys)-min(this.ys))*rand(NParticles,1)+min(this.ys);
            particleZ       = (this.zMax-this.zMin) * rand(NParticles,1) + this.zMin;

            for cnt = 1:NParticles
                particle_field(cnt).Dp = diameters(cnt);
                [particle_field(cnt).x, particle_field(cnt).y, particle_field(cnt).z] = ...
                    deal(particleX(cnt), particleY(cnt), particleZ(cnt));
            end
            clear diameters;
                      
            else
                % NParticles and also Dpmax and Dpmin are vectors. Apply
                % Poisson statistics only if they are real vectors
                if this.should_random
                NParticles = poissrnd(NParticles);
                idx = find(NParticles>0);
                this.Dpmin = this.Dpmin(idx);
                this.Dpmax = this.Dpmax(idx);
                NParticles = NParticles(idx);
                end
                
                for cnt = 1:length(NParticles)
            if abs(this.Dpmax(cnt)-this.Dpmin(cnt))<1e-7
                    diameters{cnt} = ones(NParticles(cnt),1)*this.Dpmin(cnt);
            else
            diameters{cnt} = (this.Dpmax(cnt)-this.Dpmin(cnt))*rand(NParticles(cnt),1) + this.Dpmin(cnt);
            end
                end
            diams = [];
                for cnt = 1:length(NParticles)
                    diams = [diams;diameters{cnt}];
                end

            
            
            %Generate a random grid of positions
            particleX       = (max(this.xs)-min(this.xs))*rand(sum(NParticles),1)+min(this.xs);  % Particle position
            particleY       = (max(this.ys)-min(this.ys))*rand(sum(NParticles),1)+min(this.ys);
            particleZ       = (this.zMax-this.zMin) * rand(sum(NParticles),1) + this.zMin;
            
            for cnt = 1:sum(NParticles)
                [particle_field(cnt).Dp, particle_field(cnt).x, particle_field(cnt).y, particle_field(cnt).z] = ...
                    deal(diams(cnt),particleX(cnt), particleY(cnt), particleZ(cnt));
            end    

            clear diams diameters;
            
            end
            
            
            
    end
    
    function this = set.particles(this,particles)
        this.particles = particles;
        this = this.updateNParticles;
    end
    
    function this = updateNParticles(this)
        this.NParticles = length(this.particles);
    end
    
    function image = makeEmptyNoisyHologram(this)
%      scenario = 'gauss';

        % Function to generate empty holograms with a defined beam profile
        % and random noise to get the noise level via reconstruction
        if this.should_beam
            beamProfile = this.laserDistortion;
        else
            beamProfile = ones([this.Ny this.Nx]);
        end

        hologramField = beamProfile;
        hologramIntensity = abs(hologramField).^2;
        
        if this.should_noise
            image = this.cameraDistort(hologramIntensity); 
        end
        image = uint8(image);
        
    end
    
    function hologramField = makeHologram(this,particles) % Makes the hologram from the particles structure
        
        if this.should_beam
            beamProfile = this.laserDistortion;
        else
            beamProfile = ones([this.Ny this.Nx]);
        end
        
        % Allow supersampling to get the amplitude and phase in the near
        % field as accurate as possible. Be aware of the fact that the
        % supersampled grid is four times as large as the original domain
        % to avoid interpolation artifacts for particles close to the edges
%         if this.should_extend_domain
%         xs = -this.Nx*this.superSamplingRate:1:(this.Nx)*this.superSamplingRate-1;
%         ys = -this.Ny*this.superSamplingRate:1:(this.Ny)*this.superSamplingRate-1;
%         else
        xs = -this.Nx/2*this.superSamplingRate:1:(this.Nx/2)*this.superSamplingRate-1;
        ys = -this.Ny/2*this.superSamplingRate:1:(this.Ny/2)*this.superSamplingRate-1;            
%         end
        xs = (xs./this.superSamplingRate).*this.dx;
        ys = (ys./this.superSamplingRate).*this.dy;        
        [xx,yy] = meshgrid(xs,ys);
        % For very big particles, the background has to be so huge to avoid
        % a flip of the sign (which might lead to white particles)
        hologramField = 100*complex(ones(size(beamProfile)));
        
        if isempty(particles)
            return;
        end
        
        SignalStrength = zeros(size(particles));
        shadingFactor = ones(size(particles));
        
        nxs = (-this.Nx/2:1:(this.Nx/2)-1).*this.dx;
        nys = (-this.Ny/2:1:(this.Ny/2)-1).*this.dy;
        [xxn,yyn] = meshgrid(nxs,nys);
        
%         if this.should_extend_domain
%             xsi = find(xs<=nxs(1),1,'last');
%             ysi = find(ys<=nys(1),1,'last');
%             if ~isempty(intersect(this.superSamplingRate,[1 2 4 8]))
%             [s1,s2] = size(xx);
%             xse = xsi + (s2/2) - 1;
%             yse = ysi + (s1/2) - 1;    
%             else
%                 warning('This case is not yet analyzed.'); 
%                 keyboard;
%             end
%         end
        
        % Sort the particle data by their z position to treat shading in a
        % correct way! 
        z = [particles(:).z];
        x = [particles(:).x];
        y = [particles(:).y];
        Dp = [particles(:).Dp];
        
        [z,ind] = sort(z,'descend');
        x = x(ind);
        y = y(ind);
        Dp = Dp(ind);
         
        % Set the Propagator for the reconstructions
        pEngine = Propagator();  %AB was (0)
        pEngine.config_handle = this.config_handle;        
        
        %Add a particles one at a time to the hologram
        uS = etd(clock, 1, length(particles),60);

        for cnt = 1:length(particles);
            FHCheck = 2*Dp(cnt)^2/(this.lambda*z(cnt));
            if FHCheck > 100
                 disp(['Fraunhofer condition violated for particle ',num2str(cnt),', now using another diffraction formula...']);
                 [afield,~,~] = this.makeCircleField(x(cnt),y(cnt),z(cnt),Dp(cnt),xs,ys);
                 afield(isnan(afield)) = 0;
                 if this.should_extend_domain
                 afield = afield(ysi:yse,xsi:xse);
                 end
            if this.superSamplingRate > 1
            afield = filtfilt(ones(1,this.superSamplingRate)./this.superSamplingRate,1,afield);
            afield = filtfilt(ones(1,this.superSamplingRate)./this.superSamplingRate,1,afield');
            afield = afield(round(this.superSamplingRate/4):...
                this.superSamplingRate:end,round(this.superSamplingRate/4)...
                :this.superSamplingRate:end)';
            end
            % Get the position subscripts for the particle center...
            rr = sqrt((xxn-x(cnt)).^2 + (yyn-y(cnt)).^2);
            [~,rowind] = min(rr,[],1);
            [~,colind] = min(rr,[],2);
            ri = unique(rowind);
            ci = unique(colind);
            % Check the sign of the imaginary part near the particle
            % center. If it is positive, then the particle field has to be
            % added. Otherwise, subtract it to get a realistic (dark)
            % particle! Sometimes, the phase does not seem to be valid with
            % this routine, so these lines are not yet completely analyzed.
            ai = imag(afield);
            [l,m] = size(ai);
            profile3 = ai(ri,max([1,ci-10]):min([m,ci+10]));
            profile4 = ai(max([1,ri-10]):min([l,ri+10]),ci);
            check2 = mean([profile3,profile4']);                 
 
             if check2>0
             hologramField = hologramField + afield;
             else
             hologramField = hologramField - afield;
             end
            signalIntensity = abs(afield).^2;
            SignalStrength(ind(cnt)) = sum(sum(signalIntensity));  
            else
            % Here is the new method for laser distortion
            bc = interp2(this.xs,this.ys,beamProfile,x(cnt),y(cnt));
            rr  = sqrt((xx-x(cnt)).^2 + (yy-y(cnt)).^2);
            %C = sqrt(partRelInt) Pi D_p^2 /4 lambda z
            C = pi * Dp(cnt)^2 ./(4 *this.lambda * z(cnt));
            %Q(r) = 2 J1((pi r d)/(lambda z))/((pi r d)/(lambda z))
            temp = (pi.*rr.*Dp(cnt))./(this.lambda.*z(cnt));
            Q = 2*besselj(1,temp)./(eps + temp);
            clear temp
            %Phi(r)=pi r^2/lambda z. This assumption returns correct
            %results
            Phi = pi.*rr.^2./(this.lambda*z(cnt)); 
            clear rr;
            % Field = 1 - C/j exp(j Phi) Q
            % Calculate the signal intensity
            particleField = double(1j*bc*C*exp(1j*Phi).*Q);
            clear Phi Q 
            % Correct negative amplitudes
%             if this.should_extend_domain
%             particleField = particleField(ysi:yse,xsi:xse);
%             end
            if this.superSamplingRate > 1
            particleField = filtfilt(ones(1,this.superSamplingRate)./this.superSamplingRate,1,particleField);
            particleField = filtfilt(ones(1,this.superSamplingRate)./this.superSamplingRate,1,particleField');
            particleField = particleField(round(this.superSamplingRate/4):...
                this.superSamplingRate:end,round(this.superSamplingRate/4)...
                :this.superSamplingRate:end)';
            end
            
            [l1,n1] = size(particleField);
            [l2,n2] = size(hologramField);
            
            if cnt == 1 && (l1 ~= l2 || n1 ~= n2)
                warning('Dimensions of particle and hologram field do not match. Check for inconsistencies and clear them!');
                keyboard;
            end
            % Check the sign of the imaginary part of the field to avoid
            % white particles
            t2 = imag(particleField);
            rrn = sqrt((xxn-x(cnt)).^2+(yyn-y(cnt)).^2);
            [~,rowind] = min(rrn,[],1);
            [~,colind] = min(rrn,[],2);
            ri = unique(rowind);
            ci = unique(colind);            
            [l,m] = size(t2);
            profile3 = t2(ri,max([1,ci-10]):min([m,ci+10]));
            profile4 = t2(max([1,ri-10]):min([l,ri+10]),ci); 
            
            % Reconstruction to account for shading (for the particles
            % which appear behind the one which is closest to the laser
            % beam)
            if cnt > 1
            % Temporary hologram
            tmp = (abs(hologramField)-99).^2;
            % Reconstruct at distance z of this particle to see how much
            % other particles reduce the available light for this one
            pEngine.preconstruct(tmp);
            rec = pEngine.slice(z(cnt));
            % Get the intensity at the center position of the particle
            I0 = mean(abs(gather(rec(rrn<=Dp(cnt)/2))));
            % Catch errors if the particles are very small
            if isnan(I0)
               I0 = gather(abs(rec(ri,ci))); 
            end  
            if isnan(I0)
                I0 = 1;
            end
            % Multiply with the light amplitude at the particle center
            particleField = particleField.*I0;
            shadingFactor(ind(cnt)) = I0;
            clear I0 rec tmp
            end            
            
            
            % We should subtract the field if the imaginary part is less
            % than zero (see explanation above for makeCircleField)
            if mean([profile3,profile4'])<0
            hologramField = hologramField - particleField;
            signalIntensity = abs(-particleField).^2;
            else
            signalIntensity = abs(particleField).^2;    
            hologramField = hologramField + particleField;
            end
            SignalStrength(ind(cnt)) = sum(sum(signalIntensity));  
            end
            uS = etd(uS, cnt);
            
        end

         hologramAmp = abs(hologramField);
         hologramAmp = hologramAmp - 99;
         hologramAmp(hologramAmp<0) = 0;
         hologramPhase = angle(hologramField);
         clear hologramField
         hologramField = beamProfile.*(hologramAmp.*exp(1j*hologramPhase));        

         % Export the signal strength and the shading factor (1 for no
         % influence, values close to zero indicate a strong shading
         % effect)
         this.signalStrength = [SignalStrength;shadingFactor];
           
        if this.should_cache
            this.cached_hologram_field = hologramField;
        else
            this.cached_hologram_field = [];
        end
        
    end
    
function [afield, rs, Ur] = makeCircleField(this,x,y,pz,Dp,xs,ys)

xs   = xs - x;
ys   = ys - y;
subsampling = 1;
subsampdx   = mean(diff(xs))/subsampling;
subsampdy   = mean(diff(ys))/subsampling;
subsampxs   = min(xs)-(subsampling-1)/2*subsampdx:subsampdx:max(xs)+(subsampling-1)/2*subsampdx;
subsampys   = min(ys)-(subsampling-1)/2*subsampdy:subsampdy:max(ys)+(subsampling-1)/2*subsampdy;
[xxsubsamp, yysubsamp]    = meshgrid(subsampxs, subsampys);
rrsubsamp   = sqrt(xxsubsamp.^2 + yysubsamp.^2);
clear xxsubsamp yysubsamp;

% Now make the radial function
circ.dr = this.dx/2;%this.lambda/2;
circ.rmin = min(rrsubsamp(:));
% Max out r at 7 besselj(1,) zeros:
[xx,yy] = meshgrid(xs,ys);
rmax = max(max(sqrt(xx.^2 + yy.^2))); % pi*pz/(2*pi/this.lambda*Dp)*fzero(@(x) besselj(1,pi*x), 7.2448);
circ.rmax = min(rmax, max(rrsubsamp(:))+2*circ.dr);
rs = circ.rmin:circ.dr:circ.rmax;


%uS = etd(clock, 0, numel(rs), 60);
% Now make an anonymous function for the integrand:
integrand = @(k, a, r, z, rho) ...
    exp(1j*k*a^2*rho.^2/(2*sqrt(r^2+z^2))) .* besselj(0, k*a*r*rho/sqrt(r^2+z^2)) .* rho;
integrated = zeros(size(rs));

for cnt = 1:numel(rs)
    integrated(cnt) = integral(@(rho) integrand((2*pi/this.lambda), Dp, rs(cnt), pz, rho), 0, 1);
end

Ur = -1j*(2*pi/this.lambda)*Dp^2*pz./(rs.^2+pz^2) .* exp(1j*(2*pi/this.lambda)* sqrt(rs.^2+pz^2)) .* integrated;
%Ur = exp(1j*(2*pi/this.lambda)*pz) - Ur;

afieldsubsampled = interp1(rs, Ur, rrsubsamp,'linear',0);
clear rrsubsamp;

afield = filter2(ones(subsampling)/subsampling^2, afieldsubsampled);
afield = interp2(subsampxs, subsampys', afield, xs, ys');

end
    
    
    
    
    function field = perfectHolo(this)
       if this.should_cache && ~isempty(this.cached_hologram_field)
           field = abs(this.cached_hologram_field).^2;
       else
           field = this.makeHologram(this.particles);
           field = abs(field).^2;
       end
    end        
    
%     function [image,snr] = perfectHolo(this)
%         image = abs(this.perfectHoloField).^2;
%     end
    
    function image = syntheticHolo(this)
        image = this.perfectHolo;
        %if this.should_beam, image = this.laserDistort(image);end
        if this.should_noise, image = this.cameraDistort(image); end
%         image = image - min(image(:));
        image = uint16(image);
    end
    
    function beamProfile = laserDistortion(this)
        [xx, yy] = meshgrid(this.xs,this.ys);
        
        rhorho = sqrt(xx.^2 + yy.^2);
        clear xx yy;
        % Add some noise to the Gaussian beam?
        %should_lnoise = true;
        
        % Add a Gaussian laser beam profile 
        distToCorner = sqrt((this.Nx*this.dx).^2 + (this.Ny*this.dy).^2)/2;
        beamGaussianDia = sqrt( distToCorner^2/(-2 * log(this.gaussCornCon)));


        %% Generate some random noise (addition by Oliver on May 23, 2012)
        % This option is used to simulate deviations of the center beam
        % position which is common for a real laser beam. However, we
        % cannot simulate interference fringes from optical components
        % (which are found in almost every hologram).
        if this.should_lnoise
            % Create random noise with large amplification factor
            ldistort = rand(this.Ny,this.Nx);
            fp.cutoffFreq = this.lowPassFiltCutoff;
            fp.dx = this.dx;
            fp.dy = this.dy;
            % Low-pass filter the noise
            ldistort = lowPass(ldistort,fp);
            % Normalize the noise between 0 and 1;
            ldistort = ldistort - min(ldistort(:));
            ldistort = ldistort./max(ldistort(:));
            % Put the noise level within [0.9,1]
            ldistort = ones(size(ldistort))-this.fracBeamDistort*ldistort;
        else
            ldistort = ones(this.Ny,this.Nx);
        end
        % Gaussian beam profile
        if this.gaussCornCon<1
        beamProfile = 1/(2*pi*beamGaussianDia.^2) * exp(-rhorho.^2/(2*beamGaussianDia.^2));
        else
            beamProfile = ones(size(rhorho));
        end
        beamProfile = ldistort.*beamProfile;
        % Normalize beam profile
        beamProfile = beamProfile./max(beamProfile(:));

    end
    
%     function image = laserDistort(this,image)
%          % Now distort the hologram:
%           image = image.*this.laserDistortion;
%     end
    
    function im = cameraDistort(this,field)
        tmpim = round(field*this.ccdelevel*this.ccdGain);
        im = round(field*this.ccdelevel);    % Set im to some level of electrons in the ccd's pixels
        im = poissrnd(double(im));                   % Add the poisson noise
        im = im + randn(size(im))*this.readNoiseSTD;     % Add read noise
        im = im*this.ccdGain;            % Convert to the digital numbers that the camera reads out
        im = round(im);  % Discretize to digital numbers
        diffim = (im-tmpim)./(2^(this.Nbits-1));
        im = max(0,min(im,2^this.Nbits));    % Saturate
        this.noiseLevel = sqrt(sum(diffim(:).^2)); % This is the noise level corresponding to the signal level (same units)
        %         im = im/2^this.Nbits; % Return to numbers that are comparable to the hologram for now
    end
    
    function gain = ccdGain(this)
        % Calculate the gain such that the hologram
        % results in an image at the half gray level, or half saturation.
        % (supposedly the best resolved signal.
         gain  = (2^this.Nbits*0.5)/(this.ccdelevel); 
    end
    
    %% New routine for generating hologram series
        
    function this = generateHolo(this,hologramNumber)
        % Get file names and directory names to load and save the files
        directory = this.savePath; % Directory to load and save
        if ~exist(directory,'dir')
            mkdir(directory);
        end
        fn = this.fileNameBase; % File name base
        addno = 10^(ceil(log10(this.nHolograms+1)));
        addno = num2str(addno+hologramNumber);
        
        % particleData should be the name of the structure array for the
        % particles! Otherwise seePartBlock won't work. 
        
        partFn = [directory,'/',fn,addno(2:end),'.mat'];
        imageFn = [directory,'/',fn,addno(2:end),'.png'];
        
        
        if this.makeRandParts

        % Generate random particles 
        particleData = this.randomParticles;
        save(partFn,'particleData');       
        else

        % Get the particle positions and diameters from an already generated file
        loadPartFn = [this.loadPath,'/',this.loadFnBase,addno(2:end),'.mat'];
        tmp = load(loadPartFn);
        particleData = tmp.particleData;

       

       % Save the particle positions and diameters to a file
       if ~strcmp(this.loadPath,this.savePath)
       save(partFn,'particleData');
       else
           disp('This file already exists');
       end
       
       end       
       % Make the hologram 
       this.particles = particleData;
       im = this.syntheticHolo; 
       imwrite(uint8(im),imageFn); 
    end
    
    % Function to create a particle file without making a hologram
    function this = generateParticleFile(this,hologramNumber)
        % Get file names and directory names to load and save the files
        directory = this.savePath; % Directory to load and save
        if ~exist(directory,'dir')
            mkdir(directory);
        end
        fn = this.fileNameBase; % File name base
        addno = 10^(ceil(log10(this.nHolograms+1)));
        addno = num2str(addno+hologramNumber);
        
        % particleData should be the name of the structure array for the
        % particles! Otherwise seePartBlock won't work. 
        
        partFn = [directory,'/',fn,addno(2:end),'.mat'];
       
        
        if this.makeRandParts

        % Generate random particles 
        particleData = this.randomParticles;
        save(partFn,'particleData');       
        else

        % Get the particle positions and diameters from an already generated file
        loadPartFn = [this.loadPath,'/',this.loadFnBase,addno(2:end),'.mat'];
        tmp = load(loadPartFn);
        particleData = tmp.particleData;

       % Save the particle positions and diameters to a file
       if ~strcmp(this.loadPath,this.savePath)
       save(partFn,'particleData');
       else
           disp('This file already exists');
       end
       
       end       

    end
    
    
    
    
    
    
    function this = generateEmptyHolo(this,hologramNumber)
        % Get file names and directory names to load and save the files
        directory = this.savePath; % Directory to load and save
        if ~exist(directory,'dir')
            mkdir(directory);
        end
        fn = this.fileNameBase; % File name base
        addno = 10^(ceil(log10(this.nHolograms+1)));
        addno = num2str(addno+hologramNumber);
        
        % particleData should be the name of the structure array for the
        % particles! Otherwise seePartBlock won't work. 
        
        imageFn = [directory,'/',fn,addno(2:end),'.png'];

       % Make the hologram 
       im = this.makeEmptyNoisyHologram;
       imwrite(im,imageFn); 
    end
    
    %% Routine to get SNR
    function this = getSNR(this,hologramNumber)
        % Get file names and directory names to load and save the files
        directory = this.savePath; % Directory to load and save
        if ~exist(directory,'dir')
            mkdir(directory);
        end
        fn = this.fileNameBase; % File name base
        addno = 10^(ceil(log10(this.nHolograms+1)));
        addno = num2str(addno+hologramNumber);
        
        % particleData should be the name of the structure array for the
        % particles! Otherwise seePartBlock won't work. 
        
        partFn = [directory,filesep,fn,addno(2:end),'.mat'];
        snrFn = [directory,filesep,fn,addno(2:end),'-snr.mat'];
        imageFn = [directory,filesep,fn,addno(2:end),'.png'];
        
        if this.makeRandParts&&isempty(dir([this.savePath,filesep,fn,addno(2:end),'.mat']))

        % Generate random particles 
        particleData = this.randomParticles;
        save(partFn,'particleData');       
        else

        % Get the particle positions and diameters from an already generated file
        loadPartFn = [this.loadPath,filesep,this.loadFnBase,addno(2:end),'.mat'];
        tmp = load(loadPartFn);
        particleData = tmp.particleData;

       

       % Save the particle positions and diameters to a file
       if ~strcmp(this.loadPath,this.savePath)&&~exist(partFn,'file')
       save(partFn,'particleData');
       else
           disp('This file already exists');
       end
       
       end       
       % Make the hologram 
       this.particles = particleData;
       
       tmpfn = fullfile(this.savePath,[this.loadFnBase,addno(2:end),'_hf.mat']);
       
       if isempty(dir([this.savePath,filesep,fn,addno(2:end),'_hf.mat']))
       hologram = this.perfectHolo; 
       signalStrength = this.signalStrength;
       save(tmpfn,'hologram','signalStrength');
       if this.should_noise, image = this.cameraDistort(hologram); end
       im = uint16(image);       
       else
           s7 = load(tmpfn);
           image = s7.hologram;
           signalStrength = s7.signalStrength;
       if this.should_noise, image = this.cameraDistort(image); end
        im = uint16(image);
       end
       noiseLevel = this.noiseLevel;
       save(snrFn,'signalStrength','noiseLevel');
       if ~exist(imageFn,'file')
           imwrite(uint8(im),imageFn);
       end
       
    end
    
    
    %% Set or get the values
    
    function this = set.should_cache(this,value) 
        this.should_cache = logical(this.unit_Convert(value)); 
        if ~this.should_cache 
            this.clearCache;
        end 
    end
    function clearCache(this)
        this.cached_hologram_field = [];
    end
    
    function xs = get.xs(this)
        xs  = (-this.Nx/2:this.Nx/2-1)*this.dx;
    end
    function ys = get.ys(this)
        ys  = (-this.Ny/2:this.Ny/2-1)*this.dy;
    end
    
    function value = get.dx(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dx;
        else
            value = this.dx;
        end
    end
    function this = set.dx(this,value)
        this.dx = this.unit_Convert(value); 
        this.pushToConfig('dx',this.dx); 
    end

    
    function value = get.dy(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dy;
        else
            value = this.dy;
        end
    end   
    function this = set.dy(this,value)
        this.dy = this.unit_Convert(value); 
    this.pushToConfig('dy',this.dy);
    end
    
    
    function value = get.zMin(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.zMin;
        else
            value = this.zMin;
        end
    end      
    function this = set.zMin(this,value)
        this.zMin = this.unit_Convert(value); 
        this.pushToConfig('zMin',this.zMin);
    end
    
    
    function value = get.zMax(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.zMax;
        else
            value = this.zMax;
        end
    end      
    function this = set.zMax(this,value)
        this.zMax = this.unit_Convert(value); 
        this.pushToConfig('zMax',this.zMax); 
    end    
    
    
    function value = get.lambda(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.lambda;
        else
            value = this.lambda;
        end
    end     
    function this = set.lambda(this,value)
        this.lambda = this.unit_Convert(value);
        this.pushToConfig('lambda',this.lambda);
    end
    
    function value = get.Nbits(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.Nbits;
        else
            value = this.Nbits;
        end
    end        
    function this = set.Nbits(this,value)
        value = this.unit_Convert(value);
        this.Nbits = value;
        this.pushToConfig('Nbits',value,true);
    end    
    
    
    function value = get.Nx(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.Nx;
        else
            value = this.Nx;
        end
    end
    function this = set.Nx(this,value) 
        value = this.unit_Convert(value); 
        this.Nx = value;
        this.pushToConfig('Nx',value,true); 
    end

    function value = get.Ny(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.Ny;
        else
            value = this.Ny;
        end
    end
    function this = set.Ny(this,value) 
        value = this.unit_Convert(value); 
        this.Ny = value;
        this.pushToConfig('Ny',value,true); 
    end
    
    
    
    function value = get.NParticles(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.NParticles;
        else
            value = this.NParticles;
        end
    end
    function this = set.NParticles(this,value) 
        value = this.unit_Convert(value);
        this.NParticles = value;
        this.pushToConfig('NParticles',value,true); 
    end
   
    function value = get.Dpmin(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.Dpmin;
        else
            value = this.Dpmin;
        end
    end    
    function this = set.Dpmin(this,value)
        value = this.unit_Convert(value); 
        this.Dpmin = value;
        this.pushToConfig('Dpmin',value,true); 
    end

    function value = get.Dpmax(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.Dpmax;
        else
            value = this.Dpmax;
        end
    end
    function this = set.Dpmax(this,value)
        value= this.unit_Convert(value); 
        this.Dpmax = value;
        this.pushToConfig('Dpmax',value,true); 
    end  
    
    function value = get.should_lnoise(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.should_lnoise;
        else
            value = this.should_lnoise;
        end
    end
    function this = set.should_lnoise(this,value) 
        value = this.unit_Convert(value);
        this.should_lnoise = value;
        this.pushToConfig('should_lnoise',value,true); 
    end    
    
    function value = get.partRelInt(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.partRelInt;
        else
            value = this.partRelInt;
        end
    end
    function this = set.partRelInt(this,value) 
        value = this.unit_Convert(value);
        this.partRelInt = value;
        this.pushToConfig('partRelInt',value,true); 
    end      

    function value = get.nHolograms(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.nHolograms;
        else
            value = this.nHolograms;
        end
    end
    function this = set.nHolograms(this,value) 
        value = this.unit_Convert(value);
        this.nHolograms = value;
        this.pushToConfig('nHolograms',value,true); 
    end      
    
    function value = get.gaussCornCon(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.gaussCornCon;
        else
            value = this.gaussCornCon;
        end
    end
    function this = set.gaussCornCon(this,value) 
        value = this.unit_Convert(value);
        this.gaussCornCon = value;
        this.pushToConfig('gaussCornCon',value,true); 
    end        
    
    function value = get.savePath(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.savePath;
        else
            value = this.savePath;
        end
    end
    function this = set.savePath(this,value) 
        this.savePath = value;
        this.pushToConfig('savePath',value,true); 
    end   
    
    function value = get.fileNameBase(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.fileNameBase;
        else
            value = this.fileNameBase;
        end
    end
    function this = set.fileNameBase(this,value) 
        this.fileNameBase = value;
        this.pushToConfig('fileNameBase',value,true); 
    end      
    
   function value = get.loadPath(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.loadPath;
        else
            value = this.loadPath;
        end
    end
    function this = set.loadPath(this,value) 
        this.loadPath = value;
        this.pushToConfig('loadPath',value,true); 
    end  

   function value = get.loadFnBase(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.loadFnBase;
        else
            value = this.loadFnBase;
        end
    end
    function this = set.loadFnBase(this,value) 
        this.loadFnBase = value;
        this.pushToConfig('loadFnBase',value,true); 
    end  

   function value = get.makeRandParts(this)
        if ~isempty(this.config_handle)
            value = this.config_handle.dynamic.makeRandParts;
        else
            value = this.makeRandParts;
        end
    end
    function this = set.makeRandParts(this,value) 
        value = this.unit_Convert(value); % To make the char a logical
        this.makeRandParts = value;
        this.pushToConfig('makeRandParts',value,true); 
    end  
  
    function pushToConfig(this,parameter,value,isDynamic)
        if ~isempty(this.config_handle)
            if exist('isDynamic','var') && isDynamic
                this.config_handle.dynamic.(parameter) = value;
            else
                this.config_handle.(parameter) = value;
            end
        end
    end
end

methods (Static)
    function value = unit_Convert(value)
           %This function converts string based values to floating point.  
           %It allows for units to be used in the config files, making them
           %more user friendly.
           
           m = 1;
           mm = 1e-3;
           um = 1e-6;
           nm = 1e-9;
           
           if ischar(value)
               value = eval(value);
           end
    end
end
    
end
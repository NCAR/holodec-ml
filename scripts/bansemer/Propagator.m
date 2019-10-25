classdef Propagator < dynamicprops
   %Propagator class to propagate the amplitude and phase of a plane wave field to some distance.
   %    A field (or hologram) propagation class.
   %
   %    There is a feature to do double or single reconstruction to save
   %    memory. Even when should_double is false, some of the reconstruction
   %    runs via double to get adequate precision.
   %
   %    Methods:
   %
   %    preconstruct(Field)
   %               This  initializes the reconstruction.  Reconstruction
   %               involves taking the FFT of the field and propagating that
   %               field to different distances.  Since the initial FFT is a
   %               constant (for a given input field), we cache it and some
   %               other useful properties.
   %
   %               Calling this function stores the updated FPrepped structure
   %               inside the parent object, treating it as an internal cache.
   %
   %    fieldOut = slice(z,[field])
   %               This method performs the propagation and generates the
   %               output field (single or double complex) at each z distance
   %               specified in the vector zs.
   %
   %    Inline frequency domain spatial filtering can be performed through the
   %    freq_filter parameter.  Setting this to a fuction handle will pass the
   %    spatial frequency field to the function for filtering before the final
   %    IFFT.  Setting the property to a [m,n,p] array will multiply elements
   %    of the field by the filter array along the p dimension, resulting in
   %    an [m,n,p] output field.
   %
   %    Custom filters should have a single [m,n,p] output where [m,n] must
   %    match the input field dimensions.  The function must take 2 inputs,
   %    consisting of the [m,n] input field and a copy of the config file
   %    object.
   
   %    Copyright (C) 2007-2015 Matthew Beals, Jacob Fugal
   
   %    Version History
   %    08.08.2015 Added comments to and docmunheled the code better.
   %
   %    06/2015 -- Code changed to have no GPU components and only amplitude
   %    thresholds as determined by the tresh object.
   %
   %    05/2011 -- Code developed by Matthew Beals at Michigan Tech University
   %    whie visiting MPI-Chemistry, Mainz, Germany in association with Jacob
   %    Fugal (MPI-Chemistry) as an extension of the HoloViewer package
   %    developed by Jacob Fugal.
   %
   %    Propagation algorithms originally developed by Jacob Fugal for the
   %    HoloViewer project.  Please see:
   %
   %    Fugal, J. P., T. J. Schulz, and R. A. Shaw, 2009: Practical methods
   %    for automated reconstruction and characterization of particles in
   %    digital inline holograms, Meas. Sci. Technol., 20, 075501,
   %    doi:10.1088/0957-0233/20/7/075501.
   %
   %    For more information regarding the exact nature of the hologram
   %    reconstruction technique.
   
   
   properties  (SetObservable, AbortSet)
      dx =0;
      dy =0;
      lambda = 0;
      k;
      zMaxForRes = 0;
      thresholdTuning = 0.5;
      should_cache  = true;
      should_double = false;
      should_normalize = false;
      force_recache = false;
      config_handle;
      
      pad = 0;
      recrop = 1;
      
      freq_filter;
      FPrepped_FieldFFT;
      cached_root;
      cached_filter;
      
      ampthreshs = [];
   end
   
   properties (Dependent)
      Nx, Ny;
      FPrepped_root;
      FPrepped_filter;
   end
   
   properties (GetAccess = public)
      listeners = cell(0);
   end
   
   events
      FieldFFT_update;
      Kernel_update;
      Base_update;
      newSlice_generated;
      UpdateValue;
   end
   
   %%%Constructor Method%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   methods
      function this = Propagator(propstruct)
         if ~exist('propstruct','var') || isempty(propstruct)
            this.registerListeners;
            return;
         end
         this.should_cache       = propstruct.should_cache;
         this.should_double      = propstruct.should_double;
         this.should_normalize   = propstruct.should_normalize;
         this.config_handle      = propstruct.config_handle;
         this.FPrepped_FieldFFT  = propstruct.FPrepped_FieldFFT;
         this.cached_root        = propstruct.cached_root;
         this.cached_filter      = propstruct.cached_filter;
      end
      
      function propstruct = dump(this)
         propstruct.should_cache    = this.should_cache;
         propstruct.should_double   = this.should_double;
         propstruct.should_normalize= this.should_normalize;
         propstruct.config_handle   = this.config_handle;
         propstruct.FPrepped_FieldFFT= this.FPrepped_FieldFFT;
         propstruct.cached_root     = this.cached_root;
         propstruct.cached_filter   = this.cached_filter;
      end
      
      function propcfg = dump_cfg(this)
         propcfg.dx           = this.dx;
         propcfg.dy           = this.dy;
         propcfg.lambda       = this.lambda;
         propcfg.zMaxForRes   = this.zMaxForRes;
         propcfg.ampthreshs   = this.ampthreshs;
         propcfg.zMin         = this.config_handle.zMin;
         propcfg.zMax         = this.config_handle.zMax;
         propcfg.thresholdTuning = this.thresholdTuning;
      end
      
      function loadobj(this)
         this.registerListeners;
      end
      
      function saveobj(this)
         this.unregisterListeners(this);
      end
      
      function registerListeners(this)
         this.listeners.dx             = this.addlistener('dx','PostSet',@this.setPropEvent);
         this.listeners.dy             = this.addlistener('dy','PostSet',@this.setPropEvent);
         this.listeners.lambda         = this.addlistener('lambda','PostSet',@this.setPropEvent);
         this.listeners.zMaxForRes     = this.addlistener('zMaxForRes','PostSet',@this.setPropEvent);
         this.listeners.config_handle  = this.addlistener('config_handle','PostSet',@this.setPropEvent);
      end
      
      function unregisterListeners(this)
         names = fieldnames(this.listeners);
         for cnt= 1:numel(names)
            this.listeners.(names{cnt}).delete;
            this.listeners = rmfield(this.listeners, names{cnt});
         end
      end
      
      function setPropEvent(this,~,evnt)
         notify(this,'UpdateValue',evnt);
         if ~isempty(this.FPrepped_FieldFFT)
            this.updateKernel;
         end
      end
      
   end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   methods
      
      function phi = phaseFactor(this,zs)
         phi = angle(exp(1j*this.k*zs));
      end
      %%%%%%%% method to reconstruct to a specified z position
      
      
      function fieldOut = slice(this, zs, field)
         %if we don't have a cached fft, calculate it on the fly
         if isempty(this.FPrepped_FieldFFT) || this.force_recache
            if nargin < 3, error('No cached fft.  Need a field to propagate');end
            this.preconstruct(field);
         elseif exist('field','var') && ~isempty(field);
            error('A cached field is already in this Propagator Object');
         end
         
         fieldOut = this.HFFFTPropagate(zs);
         
         if this.should_normalize
            fieldOut = this.phasenormalize(fieldOut, zs);
         end
         
         if ~this.should_cache, this.clearCache; end
         
         if this.recrop
            fieldOut = fieldOut(this.pad+1:size(fieldOut,1)-this.pad, ...
               this.pad+1:size(fieldOut,2)-this.pad, : );
         end
         
         notify(this, 'newSlice_generated');
      end
      
      function fieldOut = phasenormalize(this, fieldIn, zs)
         phaseZero = this.phaseFactor(zs);
         fieldOut = fieldIn;
         for cnt=1:size(fieldIn,3)
            fieldOut(:,:,cnt) = fieldIn(:,:,cnt) .* exp(-1j*phaseZero(cnt));
         end
      end
      
      %%%%%%%% Method to generate pre-reconstruted field for faster
      %%%%%%%% processing
      function preconstruct(this, Field)
         
         if this.pad, Field = padarray(Field, [this.pad this.pad], 'replicate' ); end
         
         % Make the FFT for this field
         this.updateFFT(Field);
         % Notify of the new field
         notify(this,'Base_update');
      end
      
      % Method to determine or estimate the thresholds for the cached
      % field, at zs from config by default, otherwise at zs specified. It
      % reconstructs, by default to 10 evenly space slices between
      % config.zMin and config.zMax.
      function [myampthreshs, mythreshstruct] = determineThresholds(this, zs)
         % We require an image to already to have been give to Propagator.
         if isempty(this.FPrepped_root)
            warning('Propagator:NoGivenField','To determine Thresholds, an image must be given to Propagator');
            myampthreshs = []; return;
         end
         if ~exist('zs','var') || istempty(zs)
            threshzs = linspace(this.config_handle.zMin, this.config_handle.zMax, 10);
         else
            threshzs = zs;
         end
         
         % Prepare ampbins using the cached FFT of the field:
         field = ifft2(this.FPrepped_FieldFFT);
         minamp = min(abs(field(:)));
         maxamp = max(abs(field(:)));
         minamp = minamp - (maxamp - minamp)/2;
         maxamp = maxamp + (maxamp - minamp)/2;
         ampbins  = linspace(minamp, maxamp, 2000)'; %2000 bins around the maximum and minimum
         % around the cached field maximum and
         % minimum plus some extra room on the
         % edges.
         
         % preallocate ampthreshs
         tampthreshs    = nan(numel(threshzs), 2);
         amphists = nan(numel(ampbins),numel(threshzs));
         fitcoeffs = nan(numel(threshzs),6);
         
         % Determine if we have a parallel pool
         mypp = gcp('nocreate');
         if ~isempty(mypp)
            propcfg     = this.dump_cfg;
            propstruct  = this.dump;
            propstruct.config_handle   = propcfg;
            parfor cnt = 1:numel(threshzs)
               pfprop  = Propagator(propstruct);
               slice   = pfprop.slice(threshzs(cnt));
               % Calculate the histogram of the amplitudes in this slice
               amphists(:,cnt)    = histc(abs(slice(:)),ampbins);
               % And estimate the threshold and get the lower amplitude threshold
               [tampthreshs(cnt,:),~,fitcoeffs(cnt,:)] = ...
                  thresh.estThisThreshAboutModeHybrid(ampbins, amphists(:,cnt), 'thresholdTuning', pfprop.thresholdTuning);
            end
        else
            for cnt = 1:numel(threshzs)
               slice   = this.slice(threshzs(cnt)); % reconstructed to a particular z
               % Calculate the histogram of the amplitudes in this slice
               amphists(:,cnt)    = histc(abs(slice(:)),ampbins)';
               % And estimate the threshold and get the lower amplitude threshold
               [tampthreshs(cnt,:),~,fitcoeffs(cnt,:)] = ...
                  thresh.estThisThreshAboutModeHybrid(ampbins, amphists(:,cnt), 'threshldTuning', this.thresholdTuning);
            end
         end
         threshstruct = struct('ampthreshs',tampthreshs,'fitcoeffs',fitcoeffs,...
            'hists',amphists,'bins',ampbins,'threshTuning',this.thresholdTuning);
         tampthreshs = [max(tampthreshs(:,1)) min(tampthreshs(:,2))]; % Use the most generous threshold from all the slices
         threshstruct.ampthresh = tampthreshs(1);
         this.ampthreshs = tampthreshs;
         
         if nargout > 0
            myampthreshs = tampthreshs;
            mythreshstruct = threshstruct;
         end
      end
   end
   
   methods
      %%%%%%%% Clear the FPrepped struct
      function clearCache(this)
         this.FPrepped_FieldFFT = [];
         this.cached_root       = [];
         this.cached_filter     = [];
         this.ampthreshs        = [];
      end
      
      %%%%%%%% Clear the force recache flag
      function clearTrigger(this)
         this.force_recache = false;
      end
      
      %%%%%%%% handle syncing dx and dy between this and the config file
      function set.dx(this,value)
         this.dx = value;
         this.pushToConfig('dx',value);
      end
      
      function set.dy(this,value)
         this.dy = value;
         this.pushToConfig('dy',value);
      end
      
      function set.lambda(this,value)
         this.lambda = value;
         this.pushToConfig('lambda',value);
      end
      
      function set.zMaxForRes(this,value)
         this.zMaxForRes = value;
         this.pushToConfig('zMaxForRes',value);
      end
      
      function set.ampthreshs(this, value)
         this.ampthreshs = value;
         this.pushToConfig('ampthreshs',value);
      end
      
      function set.thresholdTuning(this, value)
         this.thresholdTuning = value;
         this.pushToConfig('thresholdTuning',value);
      end
      
      %%%%%%%% Function to copy the value set to any parameter to the copy
      %%%%%%%% stored in config (if a config_handle is present).  This
      %%%%%%%% keeps the config object in sync and lets it broadcast the
      %%%%%%%% notifier that the value has changed
      function pushToConfig(this,parameter,value)
         if ~isempty(this.config_handle)
            this.config_handle.(parameter) = value;
         end
      end
      
      function value = get.ampthreshs(this)
         if ~isempty(this.config_handle)
            value = this.config_handle.ampthreshs;
         else
            value = this.ampthreshs;
         end
      end
      
      function value = get.thresholdTuning(this)
         if ~isempty(this.config_handle)
            value = this.config_handle.thresholdTuning;
         else
            value = this.thresholdTuning;
         end
      end
      
      %%%%%%%% Dx and Dy get methods.  These determine whether they should
      %%%%%%%% return the local copy or the config copy of the variable
      function value = get.dx(this)
         if ~isempty(this.config_handle)
            value = this.config_handle.dx;
         else
            value = this.dx;
         end
      end
      
      function value = get.dy(this)
         if ~isempty(this.config_handle)
            value = this.config_handle.dy;
         else
            value = this.dy;
         end
      end
      
      function val = get.Nx(this)
         val = size(this.FPrepped_FieldFFT,2);
      end
      function val = get.Ny(this)
         val = size(this.FPrepped_FieldFFT,1);
      end
      
      function value = get.lambda(this)
         if ~isempty(this.config_handle)
            value = this.config_handle.lambda;
         else
            value = this.lambda;
         end
      end
      
      function value = get.zMaxForRes(this)
         if ~isempty(this.config_handle)
            value = this.config_handle.zMaxForRes;
         else
            value = this.zMaxForRes;
         end
      end
      
      function value = get.k(this)
         value = 2*pi/this.lambda;
      end
      
      %%%%%%%% Update the should_cache variable and clear the cache if we
      %%%%%%%% are disabling caching.
      function set.should_cache(this, value)
         this.should_cache = value;
         if ~value, this.clearCache; end
      end
      
   end
   
   
   methods (Static)
      function bool = toBool(value)
         % This converts the floats to bool if
         %needed.  Note: 0 = false and any other number = 1
         if islogical(value)
            bool = value;
         elseif isnumeric(value)
            bool = value ~=0;
         else
            error('Invalid value. Must be logical');
         end
      end
   end
   methods
      function set.should_double(this,value)
         this.should_double = value;
      end
      
      function set.freq_filter(this,value)
         if isa(value,'function_handle')
            this.freq_filter = value;
         elseif isfloat(value)
            this.freq_filter = value;
         end
      end
   end
   
   %%%%%% Propogation Methods.... these do the work %%%%%%%%%%%%%%%%%%%%%%%%
   methods
      
      %%%%%%%% This method returns FPrepped_root or the variable as in
      %   exp(1j * lambda * z * FPrepped_root)
      %   FPrepped_root = sqrt(1 - lambda^2 * ( (x * dnux)^2 + (y * dnuy)^2 ) )
      function root = get.FPrepped_root(this)
         if ~isempty(this.cached_root)  && ...
               all(size(this.cached_root) == size(this.FPrepped_FieldFFT))
            root = this.cached_root;
            return;
         else
            this.cached_root = [];
         end
         
         %verify we have a field to work with
         if isempty(this.FPrepped_FieldFFT)
            error('No Field Loaded');
         end
         
         % Find the point of nux and nuy at which the propagator becomes
         % undersampled at distance maxz. A little work will show this to be when
         % nuy = 0, nux = +/- (lambda * sqrt((2 maxz dnux)^2 + 1 ))^-1
         [myNy, myNx] = size(this.FPrepped_FieldFFT);      % The field size
         
         
         dnux  = 1/(this.dx*myNx);   % Frequency 'pixel width'
         dnuy  = 1/(this.dy*myNy);
         
         x = [0:myNx/2-1 -myNx/2:-1];  % xs and ys
         y = [0:myNy/2-1 -myNy/2:-1];
         
         %%%
         [xx,yy] = meshgrid(x,y);        % root in phase multiplier
         
         %%%
         clear('x', 'y');
         
         root = abs(sqrt(complex(1 - this.lambda^2 ...
            *( (xx.*dnux).^2 + (yy.*dnuy).^2))));
         
         this.cached_root = root;
      end
      
      %%%%%%% This method is to make the supergaussian filter that
      %       maintains resolution along the resolution axis.
      function filter = get.FPrepped_filter(this)
         if ~isempty(this.cached_filter) && ...
               all(size(this.cached_filter) == size(this.FPrepped_FieldFFT))
            filter = this.cached_filter;
            return;
         else
            this.cached_filter = [];
         end
         
         % Find the point of nux and nuy at which the propagator becomes
         % undersampled at distance maxz. A little work will show this to be when
         % nuy = 0, nux = +/- (lambda * sqrt((2 maxz dnux)^2 + 1 ))^-1
         [myNy, myNx] = size(this.FPrepped_FieldFFT);      % The field size
         
         
         dnux  = 1/(this.dx*myNx);   % Frequency 'pixel width'
         dnuy  = 1/(this.dy*myNy);
         
         nuxwidth = 1/(this.lambda*sqrt(1 + (2*this.zMaxForRes*dnux)^2));
         nuywidth = 1/(this.lambda*sqrt(1 + (2*this.zMaxForRes*dnuy)^2));
         
         x = [0:myNx/2-1 -myNx/2:-1];  % xs and ys
         y = [0:myNy/2-1 -myNy/2:-1];
         
         %%%
         [xx,yy] = meshgrid(x,y);        % root in phase multiplier
         %Supergaussian cutoff filter
         % SG(x,y) = exp(-1/2*((x/sigmax)^2 + (y/sigmay)^2)^n )
         
         f        = .5;
         n        = 3;
         sigmax   = nuxwidth * log(1/f^2)^(-1/(2*n));
         sigmay   = nuywidth * log(1/f^2)^(-1/(2*n));
         
         if ~this.should_double
            xx = single(xx);
            yy = single(yy);
         end
         
         if this.zMaxForRes <= 0
            filter = ones(size(xx),'like',xx);
         else
            filter = exp(-1/2*((xx*dnux/sigmax).^2+(yy*dnuy/sigmay).^2).^n);
         end
         
         this.cached_filter = filter;
      end
      
      %%%%%%%% Method to update the field FFT
      function updateFFT(this, Field)
         
         %resize the image to make it FFT friendly
         if mod(size(Field,2),2), Field = Field(:,1:end-1); end
         if mod(size(Field,1),2), Field = Field(1:end-1,:); end
         
         if this.should_double
            %if we are using double precision, then use it
            Field = double(Field);
         else
            Field = single(Field);
         end
         
         this.FPrepped_FieldFFT = Field;
         clear Field;
         
         this.FPrepped_FieldFFT = fft2(this.FPrepped_FieldFFT);
         notify(this,'FieldFFT_update')
      end
      
      %%%%
      function updateKernel(this)
         this.clearCache;
         notify(this,'Base_update');
      end
      
      %%% Propagation methods %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      function fieldOut = HFFFTPropagate(this, thisz)
         
         %Calculate the slice fft
         root = this.FPrepped_root;
         filter = this.FPrepped_filter;
         phaseFilter = exp((1j*this.k * thisz) * root);
         
         FieldFFT = this.FPrepped_FieldFFT .* phaseFilter;
         
         FieldFFT = FieldFFT .* filter;
         
         
         %Apply spatial freqency filtering if specified
         if isa(this.freq_filter,'function_handle')
            FieldFFT = this.freq_filter(FieldFFT,this.config_handle);
         elseif ~isempty(this.freq_filter) && isfloat(this.freq_filter)
            FieldFFT = this.freq_filter .* FieldFFT;
         end
         
         % Take the ifft2 which completes the transform
         fieldOut = ifft2(FieldFFT);
         
         if ~this.should_double && isa(fieldOut,'double')
            fieldOut = single(fieldOut);
         end
      end
   end
end

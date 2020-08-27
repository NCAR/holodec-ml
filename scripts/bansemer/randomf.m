function out = randomf(x, y, n)
   %Return n random numbers based on the distribution described by x and y.  
   %AB 7/2020
   
   if min(y) < 0
      disp('y values must all be positive numbers');
      out = 0;
      return
   end

   %Numerically integrate the function to find cumulative probability
   nx = length(x);
   r = rand(1,n);
   cp = zeros(1,nx);
  
   %Need to do some normalization in case of uneven x
   dx = [0,x(2:end)-x(1:end-1)];    
   
   %Integrate at each interval
   for i = 2:nx
      cp(i) = cp(i-1) + y(i)*dx(i);  
   end
   
   %Normalize to 1
   cp = cp./max(cp);
   
   %Matlab gives an interpolation error if delta(cp) is too small
   %so truncate after maximum needed value
   top = min(find(cp > max(r))); 
   
   %Find interpolates of the inverse function at random spots
   out = interp1(cp(1:top), x(1:top), r);
end

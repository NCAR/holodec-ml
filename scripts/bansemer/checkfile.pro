PRO checkfile, fn
   ;Use IDL to check the netCDF files
   restorenc, fn, /lite
   loadct,0   
   v=''
   window,0,xsize=data.global.nx, ysize=data.global.ny
   FOR i=0,data.global.nholograms-1 DO BEGIN
      print,i,format='(i6, $)'
      restorenc, fn, 'd2', var='image', offset=[0,0,i], count=[data.global.ny, data.global.nx, 1]
      image=transpose(d2.image)  ;netCDF convention is different than IDL
      ;Find particles in this hologram
      w=where(data.hid eq i+1, nw)  ;+1 is due to numbering convention (Matlab)
      ;Mark centers with a white dot
      x=data.x[w]/data.global.dx/1e6+data.global.nx/2
      y=data.y[w]/data.global.dy/1e6+data.global.ny/2
      image[x,y]=255
      tv,image
      FOR j=0, nw-1 DO xyouts, x[j], y[j]+10, string('D=',data.d[w[j]], 'z=', data.z[w[j]]/1e4,format='(a2,f4.1,a3,f4.1)'), /device 
      read,v,prompt='  [Ret] for next, q to quit, s to save PNG, or #'
      IF string(v) eq 'q' THEN return
      IF string(v) eq 's' THEN BEGIN & write_png, 'hologram_'+string(i,format='(i05)')+'.png',transpose(d2.image) & v='' & ENDIF 
      IF long(v) ne 0 THEN i=long(v)-1
   ENDFOR
END


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
      
      ;Mark centers with a white cross
      x=data.x[w]/data.global.dx/1e6+data.global.nx/2
      y=data.y[w]/data.global.dy/1e6+data.global.ny/2
      minx=(x-2)>0
      miny=(y-2)>0
      maxx=(x+2)<(data.global.nx-1)
      maxy=(y+2)<(data.global.ny-1)
      FOR j=0, nw-1 DO BEGIN
         image[minx[j]:maxx[j],y[j]]=255
         image[x[j],miny[j]:maxy[j]]=255
      ENDFOR
      tv,image
      
      ;Write properties to screen
      FOR j=0, nw-1 DO BEGIN
         ;xyouts, x[j], y[j]+10, string('D=',data.d[w[j]],'um  z=', data.z[w[j]]/1e4,'cm',format='(a2,f4.1,a6,f4.1,a2)'), /device, charsize=1.5, charthick=2
         cgtext, x[j], y[j]+10, string('D=',data.d[w[j]],'um  z=', data.z[w[j]]/1e4,'cm',format='(a2,f4.1,a6,f4.1,a2)'), /device, color='white', charsize=2.5, charthick=2, /font, tt_font='Helvetica'
      ENDFOR
      
      ;Move to next image
      read,v,prompt='  [Ret] for next, q to quit, s to save PNG, r to save clean PNG, or #'
      IF string(v) eq 'q' THEN return
      IF string(v) eq 'r' THEN BEGIN & write_png, 'hologram_'+string(i,format='(i05)')+'.png',transpose(d2.image) & v='' & ENDIF 
      IF string(v) eq 's' THEN BEGIN & write_png, 'hologram_'+string(i,format='(i05)')+'.png',tvrd(/true) & v='' & ENDIF 
      IF long(v) ne 0 THEN i=long(v)-1
   ENDFOR
END


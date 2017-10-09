vec4 tex2DBiLinear( sampler2D textureSampler_i, vec2 texCoord_i )
{
    vec4 p0q0 = texture2D(textureSampler_i, texCoord_i);
    vec4 p1q0 = texture2D(textureSampler_i, texCoord_i + vec2(texelSizeX, 0));

    vec4 p0q1 = texture2D(textureSampler_i, texCoord_i + vec2(0, texelSizeY));
    vec4 p1q1 = texture2D(textureSampler_i, texCoord_i + vec2(texelSizeX , texelSizeY));

    float a = fract( texCoord_i.x * fWidth ); // Get Interpolation factor for X direction.
					// Fraction near to valid data.

    vec4 pInterp_q0 = mix( p0q0, p1q0, a ); // Interpolates top row in X direction.
    vec4 pInterp_q1 = mix( p0q1, p1q1, a ); // Interpolates bottom row in X direction.

    float b = fract( texCoord_i.y * fHeight );// Get Interpolation factor for Y direction.
    return mix( pInterp_q0, pInterp_q1, b ); // Interpolate in Y direction.
}

    index = floor(vTexCoord.s * float(uTexSize.x)) + floor(vTexCoord.t * float(uTexSize.y)) * uTexSize.x;
    
interpolate = gpu.createKernel(function(A) { 
    var texCoord_i = vec2(vTexCoord.s,vTexCoord.t);
    var p0q0 = texture2D(A, texCoord_i);
    var p1q0 = texture2D(A, texCoord_i + vec2(uTexSize.x, 0));

    var p0q1 = texture2D(A, texCoord_i + vec2(0, uTexSize.y));
    var p1q1 = texture2D(A, texCoord_i + vec2(uTexSize.x , uTexSize.y));

    var a = fract( texCoord_i.x / float(uTexSize.x) ); // Get Interpolation factor for X direction.
					// Fraction near to valid data.

    var pInterp_q0 = mix( p0q0, p1q0, a ); // Interpolates top row in X direction.
    var pInterp_q1 = mix( p0q1, p1q1, a ); // Interpolates bottom row in X direction.

    var b = fract( texCoord_i.y / float(uTexSize.y) );// Get Interpolation factor for Y direction.
    return decode32(mix( pInterp_q0, pInterp_q1, b ));
    //return decode32(texture2D(a, vec2(vTexCoord.s,vTexCoord.t)));
}).setOutput([8,8])

interpolate_nearest = gpu.createKernel(function(a) { 
    return decode32(texture2D(a, vec2(vTexCoord.s,vTexCoord.t)));
}).setOutput([8,8])

gpu.addNativeFunction('tex2DBiLinear', `vec4 tex2DBiLinear( sampler2D textureSampler_i, vec2 texCoord_i, vec2 texSize)
{
    vec4 p0q0 = texture2D(textureSampler_i, texCoord_i);
    vec4 p1q0 = texture2D(textureSampler_i, texCoord_i + vec2(texSize.x, 0));

    vec4 p0q1 = texture2D(textureSampler_i, texCoord_i + vec2(0, texSize.y));
    vec4 p1q1 = texture2D(textureSampler_i, texCoord_i + vec2(texSize.x , texSize.y));

    float a = fract( texCoord_i.x / texSize.x ); // Get Interpolation factor for X direction.
					// Fraction near to valid data.

    vec4 pInterp_q0 = mix( p0q0, p1q0, a ); // Interpolates top row in X direction.
    vec4 pInterp_q1 = mix( p0q1, p1q1, a ); // Interpolates bottom row in X direction.

    float b = fract( texCoord_i.y * texSize.y );// Get Interpolation factor for Y direction.
    return mix( pInterp_q0, pInterp_q1, b ); // Interpolate in Y direction.
}`);


interpolate = gpu.createKernel(function(a) {
    return decode32(tex2DBiLinear(a, vec2(vTexCoord.s,vTexCoord.t), vec2(uTexSize.x, uTexSize.y)));
}).setOutput([8,8])


decimate = gpu.createKernel(function(a) {
    var ix = this.thread.x*2;
    var iy = this.thread.y*2;
    return 0.25* (a[ix][iy] + a[ix+1][iy] + a[ix][iy+1] + a[ix+1][iy+1])
}).setOutput([4,4])

interpolate = gpu.createKernel(function(f){
     const tsize = this.constants.size;
     var x = this.thread.x*0.5;
     var y = this.thread.y*0.5;
     var ix = floor(x*0.5 - 0.5);
     var iy = floor(y*0.5 - 0.5);
     var a = x + 0.25 - (ix + 0.5);
     var b = y + 0.25 - (iy + 0.5);
     var ix0 = mod(ix  , tsize);
     var ix1 = mod(ix+1, tsize);
     var iy0 = mod(iy  , tsize);
     var iy1 = mod(iy+1, tsize);

     const p0q0 = f[ix0][iy0];
     const p1q0 = f[ix1][iy0];
     const p0q1 = f[ix0][iy1];
     const p1q1 = f[ix1][iy1];

     const pInterp_q0 = mix( p0q0, p1q0, a ); // Interpolates top row in X direction.
     const pInterp_q1 = mix( p0q1, p1q1, a ); // Interpolates bottom row in X direction.
     return mix( pInterp_q0, pInterp_q1, b ); // Interpolate in Y direction.
})
.setConstants({size: 4})
.setOutput([8,8]);


interpolate = gpu.createKernel(function(f){
     const tsize = this.constants.size;
     var x = this.thread.x;
     var y = this.thread.y;
     var ix = floor(this.thread.x*0.5 - 0.5);
     var iy = floor(this.thread.y*0.5 - 0.5);
     var a = x + 0.25 - (ix+0.5);
     var b = y + 0.25 - (iy+0.5);
     var ix0 = mod(ix  , tsize);
     var ix1 = mod(ix+1, tsize);
     var iy0 = mod(iy  , tsize);
     var iy1 = mod(iy+1, tsize);
     return a; // Interpolate in Y direction.
})
.setConstants({size: 4})
.setOutput([8,8]);


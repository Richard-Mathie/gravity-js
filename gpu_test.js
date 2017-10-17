(function() {
    var matrixSize =  512;
    var a = new Array(matrixSize*matrixSize);
    var b = new Array(matrixSize*matrixSize);
    var c = new Array(matrixSize*matrixSize);

    a = splitArray(fillArrayZero(a), matrixSize);
    b = splitArray(fillArrayZero(b), matrixSize);
    b[128-64][128-64] = 16.9;
    b[128-64][128+64] = -16.9;
    c = splitArray(fillArrayRandom(c), matrixSize);
    var d = c
    var phi = a;
    var rho = b;
    //console.log(a);

    const gpu = new GPU({ mode: 'gpu' });
    const cpu = new GPU({ mode: 'cpu' });

    function mod(n, m) {
        return ((n % m) + m) % m;
    }

    cpu.addFunction(mod)

    // Laplace Transform
    // aij+1 + aij-1 + ai+1j + ai-1j - 4 aij
    const cross = gpu.createKernel(function(a, b) {
        var sum = 0;
	    for (var i=0; i<2; i++) {
		    sum += B[this.thread.y][i] * C[i][this.thread.x];
	    }
	    return sum;
    })
    .setOutput([2, 2])
    .setOutputToTexture(true);

    var s = 1;
    var dh = 1;
    var dh2 = dh ** 2;

    function mat_scale(A, a) {
        return A.map(function(row){return row.map(function (e){return e*a;});});
    }

    function mat_mult(A, b) {
        var result = new Array(A.length);
        var row, sum;
        for (var i=0, m=A.length; i< m; i++){
            row = A[i];
            sum = 0;
            for (var j=0, n=row.length; j< n; j++){
                sum += row[j] * b[j];
            };
            result[i] = sum;
        };
        return result;
    };

    // inverse of the laplace oporator for a 2x2 grid with
    var Apinv = [[-5,  1,  1,  3],
                 [ 1, -5,  3,  1],
                 [ 1,  3, -5,  1],
                 [ 3,  1,  1, -5]]; // / 32
    Apinv = mat_scale(Apinv, 1/32 * (dh*matrixSize/2));

    /* =========== rendering functions =========== */
    const render = gpu.createKernel(function(a) {
        var p = a[this.thread.y][this.thread.x];
        var r = p % 1;
        var g = 1 - r;
        this.color(r, g, 0, 1);
    })
    .setOutput([matrixSize, matrixSize])
    .setGraphical(true);

    const canvas = render.getCanvas();

    document.getElementsByTagName('body')[0].appendChild(canvas);

    var ctx = canvas.getContext('2d');
    //ctx.scale(10, 3);

    const make_kernel = function (level) {
        var gridSize = level;
        var dhs = dh*matrixSize/gridSize;
        var dhs2 = dhs ** 2;
        var phi = new Array(gridSize*gridSize);
        phi = splitArray(fillArrayZero(phi), gridSize);

        const decimate = gpu.createKernel(function(a) {
            var ix = this.thread.x*2;
            var iy = this.thread.y*2;
            return 0.25* (a[iy][ix] + a[iy+1][ix] + a[iy][ix+1] + a[iy+1][ix+1]);
        })
        .setOutput([gridSize/2, gridSize/2])
        .setOutputToTexture(true);

        const interpolate2 = gpu.createKernel(function(f){
             const tsize = this.constants.size;
             var ix = floor(this.thread.x*0.5);
             var iy = floor(this.thread.y*0.5);
             return f[iy][ix]
        })
        .setConstants({size: gridSize/2})
        .setOutput([gridSize, gridSize])
        .setOutputToTexture(true);

        const interpolate = gpu.createKernel(function(f){
             const tsize = this.constants.size;
             var x = this.thread.x*0.5;
             var y = this.thread.y*0.5;
             var ix = floor(x - 0.25);
             var iy = floor(y - 0.25);
             var a = x - ix - 0.25;
             var b = y - iy - 0.25;
             var ix0 = mod(ix  , tsize);
             var ix1 = mod(ix+1, tsize);
             var iy0 = mod(iy  , tsize);
             var iy1 = mod(iy+1, tsize);

             const p0q0 = f[iy0][ix0];
             const p1q0 = f[iy0][ix1];
             const p0q1 = f[iy1][ix0];
             const p1q1 = f[iy1][ix1];

             const pInterp_q0 = mix( p0q0, p1q0, a ); // Interpolates top row in X direction.
             const pInterp_q1 = mix( p0q1, p1q1, a ); // Interpolates bottom row in X direction.
             return mix( pInterp_q0, pInterp_q1, b ); // Interpolate in Y direction.
        })
        .setConstants({size: gridSize/2})
        .setOutput([gridSize, gridSize])
        .setOutputToTexture(true);

        const row_sum = gpu.createKernel(function(a) {
            var sum =0;
            for (var i=0; i<this.constants.size; i++){
                sum += a[i][this.thread.x];
            }
            return sum;
        })
        .setConstants({size: gridSize})
        .setOutput([gridSize]);

        function sum2(t){
            return row_sum(t).reduce(function(a,b){return a+b;});
        };

        return {
        gridSize: gridSize,
        dh: dhs,
        dh2: dhs2,
        phi: phi,

        decimate: decimate,

        interpolate2: interpolate2,
        interpolate: interpolate,
        row_sum: row_sum,
        sum2: sum2,

        laplace_relaxation: gpu.createKernel(function(a, rho) {
            var sum = - rho[this.thread.y][this.thread.x] * this.constants.dh2;
            var ip;
            var s = this.constants.s;
            sum += a[this.thread.y][mod(this.thread.x - 1, this.constants.size)];
            sum += a[this.thread.y][mod(this.thread.x + 1, this.constants.size)];
            sum += a[mod(this.thread.y - 1, this.constants.size)][this.thread.x];
            sum += a[mod(this.thread.y + 1, this.constants.size)][this.thread.x];
            return ((1-s) * a[this.thread.y][this.thread.x]) + ( s*0.25*sum );
        })
        .setConstants({size: gridSize, s: s, dh2: dhs2})
        .setOutput([gridSize, gridSize])
        .setOutputToTexture(true),

        laplace_residual: gpu.createKernel(function(a, rho) {
            var sum = - 4*a[this.thread.y][this.thread.x];
            sum += a[this.thread.y][mod(this.thread.x - 1, this.constants.size)];
            sum += a[this.thread.y][mod(this.thread.x + 1, this.constants.size)];
            sum += a[mod(this.thread.y - 1, this.constants.size)][this.thread.x];
            sum += a[mod(this.thread.y + 1, this.constants.size)][this.thread.x];
            return rho[this.thread.y][this.thread.x] - sum * this.constants.dhm2;
        })
        .setConstants({size: matrixSize, dhm2: 1/dhs2})
        .setOutput([matrixSize, matrixSize])
        .setOutputToTexture(true),

        add: gpu.createKernel(function(a, b){
            return a[this.thread.y][this.thread.x] + b[this.thread.y][this.thread.x];
        })
        .setOutput([matrixSize, matrixSize])
        .setOutputToTexture(true),


        offset: gpu.createKernel(function(a, b){
            return a[this.thread.y][this.thread.x] + b[0];
        })
        .setOutput([matrixSize, matrixSize])
        .setOutputToTexture(true)
        }
    };

    /* ========== Make Levels ============ */

    var levels = Math.log2(matrixSize);
    levels = Array.apply(null, Array(levels)).map(function (_, i) {return 2 ** (levels - i);});
    levels = levels.map(function(l){return make_kernel(l);});
    levels.forEach(function(l){})
    levels[0].rho = rho

    /* ======== Multi-Grid Loop ========== */

    function MultiGrid(){
        // Coarsen
        for (var i=0, m=levels.length - 1; i < m; i++){
            var level = levels[i]
            for (var step=10;step--;){
                var phi = level.laplace_relaxation(level.phi, level.rho);
            }
            var phi_mean = level.sum2(phi) / level.gridSize ** 2;
            phi = level.offset(phi, [-phi_mean]);
            var r = level.laplace_residual(phi, level.rho);
            levels[i+1].rho = level.decimate(r);
            level.r = r;
            level.phi = phi;
        }

        // Solve Top level 2x2 grid
        var level = levels[levels.length-1];
        var phi = mat_mult(Apinv, [].concat.apply([], level.rho.toArray(gpu)));
        // Correct zero
        var phi_sum=0;
        for (var i=phi.length; i--;) {
          phi_sum+=phi[i];
        }
        phi_sum = phi_sum/phi.length
        for (var i=phi.length; i--;) {
          phi[i] -= phi_sum;
        }
        level.phi = splitArray(phi, 2)

        // Smooth
        for (var i=levels.length-2; i >= 0; i--){
            var level = levels[i]
            var error = level.interpolate(levels[i+1].phi);  // interpolation
            var phi = level.add(error, level.phi);             // correction
            for (var step=10;step--;){
                phi = level.laplace_relaxation(phi, level.rho);
            };
            var phi_mean = level.sum2(phi) / level.gridSize ** 2;
            level.phi = level.offset(phi, [-phi_mean]);
        }
    }

    /* ======== Multi-Grid Loop ========== */

    function MultiGrid2(){
        // Coarsen
        for (var i=0, m=levels.length - 1; i < m; i++){
            levels[i+1].rho = levels[i].decimate(levels[i].rho);
        }

        // Solve Top level 2x2 grid
        var level = levels[levels.length-1];
        var phi = mat_mult(Apinv, [].concat.apply([], level.rho.toArray(gpu)));
        // Correct zero
        var phi_sum=0;
        for (var i=phi.length; i--;) {
          phi_sum+=phi[i];
        }
        phi_sum = phi_sum/phi.length
        for (var i=phi.length; i--;) {
          phi[i] -= phi_sum;
        }
        level.phi = splitArray(phi, 2)

        // Smooth
        for (var i=levels.length-2; i >= 0; i--){
            var level = levels[i];
            var phi = level.interpolate(levels[i+1].phi);  // interpolation
            for (var step=10;step--;){
                phi = level.laplace_relaxation(phi, level.rho);
            };
            level.phi_sum = level.sum2(phi);
            level.phi = level.offset(phi, [-level.phi_sum / level.gridSize ** 2]);
            level.r = level.laplace_residual(level.phi, level.rho);
        }
    }

    level = levels[0]
    // render(level.phi)
    // render(level.laplace_residual(level.phi, level.rho))
    const to_texture = gpu.createKernel(function(a){
      return a[this.thread.y][this.thread.x]}
    )
    .setOutput([512,512])
    .setOutputToTexture(true);

    var old_time = +new Date();
    var time;
    var start;

    //MultiGrid2()

    function animate() {
        start = + new Date();

        level.phi = level.laplace_relaxation(level.phi, level.rho);
        var phi_sum = level.sum2(level.phi);
        level.phi = level.offset(level.phi, [-phi_sum / level.gridSize ** 2]);
        console.log(level.phi_sum);
        render(level.phi);
        time = + new Date();
        console.log( parseFloat(1000/(time - old_time)).toFixed(1) + " fps");
        console.log( parseFloat(1000/(time - start)).toFixed(1) + " anim");
        old_time = time;
        window.requestAnimationFrame(animate);
    }

    window.requestAnimationFrame(animate)


    function fillArrayRandom(array) {
        for(var i = 0; i < array.length; i++) {
          array[i] = Math.random()/2 -0.25;
        }
        return array;
    }

    function fillArrayZero(array) {
        for(var i = 0; i < array.length; i++) {
          array[i] = 0
        }
        return array;
    }

    function splitArray(array, part) {
        var result = [];
        for(var i = 0; i < array.length; i += part) {
          result.push(array.slice(i, i + part));
        }
        return result;
    }
})();

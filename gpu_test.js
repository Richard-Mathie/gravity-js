(function() {
    var matrixSize =  512;
    var a = new Array(matrixSize*matrixSize);
    var b = new Array(matrixSize*matrixSize);
    var c = new Array(matrixSize*matrixSize);

    a = splitArray(fillArrayZero(a), matrixSize);
    b = splitArray(fillArrayZero(b), matrixSize);
    b[128-64][128-64] = 16;
    b[128+64][128+64] = -16;
    c = splitArray(fillArrayRandom(c), matrixSize);
    var d = c
    //console.log(a);

    const gpu = new GPU({ mode: 'gpu' });
    const cpu = new GPU({ mode: 'cpu' });

    function mod(n, m) {
        return ((n % m) + m) % m;
    }

    cpu.addFunction(mod)

    var opts_1d =  {constants: {size: matrixSize},
                    output: [matrixSize] };
    var opts_2d =  {constants: {size: matrixSize},
                    output: [matrixSize, matrixSize] };

    // Laplace Transform
    // aij+1 + aij-1 + ai+1j + ai-1j - 4 aij
    const laplace = cpu.createKernel(function(a) {
        var sum = -4 * a[this.thread.y][this.thread.x];
        var ip;
        for (var i=0; i<2; i++) {
            ip = mod(this.thread.x + i*2 - 1, this.constants.size);
            sum += a[this.thread.y][ip];
        }
        for (var i=0; i<2; i++) {
            ip = mod(this.thread.y + i*2 - 1, this.constants.size);
            sum += a[ip][this.thread.x];
        }
        return sum;
    })
    .setConstants({size: matrixSize})
    .setOutput([matrixSize, matrixSize]);

    var s = 1;
    var dh = 1;
    var dh2 = dh ** 2;

    const make_kernel = function (level) {
        var gridSize = 2 ** level;
        var dhs = dh*matrixSize/gridSize;
        var dhs2 = dhs ** 2;

        return {decimate: gpu.createKernel(function(a) {
            var ix = this.thread.x*2;
            var iy = this.thread.y*2;
            return 0.25* (a[ix][iy] + a[ix+1][iy] + a[ix][iy+1] + a[ix+1][iy+1])
        }).setOutput([gridSize/2, gridSize/2]),

        interpolate: gpu.createKernel(function(f){
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
        .setConstants({size: gridSize})
        .setOutput([gridSize*2, gridSize*2])
        .setOutputToTexture(true),

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
        .setConstants({size: matrixSize, dhm2: 1/dh2})
        .setOutput([matrixSize, matrixSize])
        .setOutputToTexture(true),

        add: gpu.createKernel(function(a, b){
            return a[this.thread.y][this.thread.x] + b[this.thread.y][this.thread.x];
        })
        .setOutput([matrixSize, matrixSize])
        .setOutputToTexture(true)}
    };

    function mat_mult(A, b) {
        var result = new Array(A.length);
        var row, sum;
        for (var i=0, m=A.length; i< m; i++){
            row = A[i];
            sum = 0;
            for (var j=0, n=row.length; j< n; i++){
                sum += row[j] * b[j];
            };
            result[i] = sum;
        };
    };

    function scale(A, a) {
        return A.map(function(row){return row.map(function (e){return e*a;});});
    }

    // inverse of the laplace oporator for a 2x2 grid with 
    var Apinv = [[-5,  1,  1,  3],
                 [ 1, -5,  3,  1],
                 [ 1,  3, -5,  1],
                 [ 3,  1,  1, -5]]; // / 32
    Apinv = scale(Apinv, 1/32);

    const laptest = gpu.createKernel(function(a) {
        var sum = 0;
        sum += a[this.thread.y][mod(this.thread.x - 1, this.constants.size)];
        sum += a[this.thread.y][mod(this.thread.x + 1, this.constants.size)];
        sum += a[mod(this.thread.y - 1, this.constants.size)][this.thread.x];
        sum += a[mod(this.thread.y + 1, this.constants.size)][this.thread.x];
        //sum += a[this.thread.y][this.thread.x]
        return sum;
    })
    .setConstants({size: matrixSize})
    .setOutput([matrixSize, matrixSize])

    const laplace_relaxation = gpu.createKernel(function(a, rho) {
        var sum = - rho[this.thread.y][this.thread.x] * this.constants.dh2;
        var ip;
        var s = this.constants.s;
        sum += a[this.thread.y][mod(this.thread.x - 1, this.constants.size)];
        sum += a[this.thread.y][mod(this.thread.x + 1, this.constants.size)];
        sum += a[mod(this.thread.y - 1, this.constants.size)][this.thread.x];
        sum += a[mod(this.thread.y + 1, this.constants.size)][this.thread.x];
        return ((1-s) * a[this.thread.y][this.thread.x]) + ( s*0.25*sum );
    })
    .setConstants({size: matrixSize, s: s, dh2: dh2})
    .setOutput([matrixSize, matrixSize])
    //.setFloatTextures(true);
    .setOutputToTexture(true);

    const add = gpu.createKernel(function(a, b) {
	    return a[this.thread.y][this.thread.x] + b[this.thread.y][this.thread.x];
    }, opts_2d);
    
    const sum_row = gpu.createKernel(function(a) {
        var sum =0;
        for (var i=0; i<this.constants.size; i++){
            sum += a[i][this.thread.x];
        }
        return sum;
    }, {constants: {size: matrixSize},
        output: [matrixSize]});

    const multiply = gpu.createKernel(function(a, b) {
	    return a[this.thread.y][this.thread.x] * b[this.thread.y][this.thread.x];
    }, opts_2d);

    const scale = gpu.createKernel(function(a, b) {
	    return a * b[this.thread.y][this.thread.x];
    }, opts_2d);

    const cross = gpu.createKernel(function(a, b) {
        var sum = 0;
	    for (var i=0; i<this.constants.size; i++) {
		    sum += B[this.thread.y][i] * C[i][this.thread.x];
	    }
	    return sum;
    }, {constants: {size: matrixSize},
        output: [matrixSize, matrixSize]});

//    const superKernel = gpu.combineKernels(add, multiply, function(a, b, c) {
//	    return multiply(add(a, b), c);
//    });

    //console.log(superKernel(a, b, c));
    //console.log(c);
    console.log(laplace(a))

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

    var phi = a;
    var rho = b;

    var old_time = +new Date();
    var time;
    var start;

    function animate() {
        start = + new Date();
        for(var i=0; i<100; i++){
            phi = laplace_relaxation(phi, rho)
        };
        var sum = sum_row(phi)
        console.log(sum.reduce(function(a,c){return a+c;}));
        render(phi);
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

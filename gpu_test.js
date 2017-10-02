(function() {
    var matrixSize =  4; //512;
    var a = new Array(matrixSize*matrixSize);
    var b = new Array(matrixSize*matrixSize);
    var c = new Array(matrixSize*matrixSize);
    a = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]];
    b = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 0]];
    //a = splitArray(fillArrayZero(a), matrixSize);
    //b = splitArray(fillArrayRandom(b), matrixSize);
    //b[128-64][128-64] = 128;
    //b[128+64][128+64] = -128;
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
    var dh2 = dh * dh;

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
        for (var i=0; i<2; i++) {
            ip = mod(this.thread.x + i*2 - 1, this.constants.size);
            sum += a[this.thread.y][ip];
        }
        for (var i=0; i<2; i++) {
            ip = mod(this.thread.y + i*2 - 1, this.constants.size);
            sum += a[ip][this.thread.x];
        }
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

    //laplace_relaxation = gpu.combineKernels(laplace, scale, add, function(a) {
	//    return add(a, scale(0.1, laplace(a)));
	//});

    //laplace_relaxation.setOutputToTexture(true);
    var old_time = +new Date();
    var time;
    var start;

    function animate() {
        start = + new Date();
        //for(var i=0; i<10; i++){
            //phi = laplace_relaxation(phi);
            //phi = add(phi, scale(0.24, laplace(phi)));
            phi = laplace_relaxation(phi, rho)
            var m2 = matrixSize/2,
                m4 = matrixSize/4;
            //phi[m2-m4][m2-m4] = 300;
            //phi[m2+m4][m2+m4] = -300;
        //};
        //var sum = sum_row(phi)
        //console.log(sum.reduce(function(a,c){return a+c;}));
        render(phi.toArray(gpu));
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

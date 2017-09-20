(function() {
    var matrixSize = 128;
    var a = new Array(matrixSize*matrixSize);
    var b = new Array(matrixSize*matrixSize);
    var c = new Array(matrixSize*matrixSize);
    a = splitArray(fillArrayRandom(a), matrixSize);
    b = splitArray(fillArrayRandom(b), matrixSize);
    c = splitArray(fillArrayRandom(c), matrixSize);
    var d = c
    console.log(a);

    const gpu = new GPU({ mode: 'gpu' });

    var opts_1d =  {constants: {size: matrixSize},
                    output: [matrixSize] };
    var opts_2d =  {constants: {size: matrixSize},
                    output: [matrixSize, matrixSize] };

    // Laplace Transform
    // aij+1 + aij-1 + ai+1j + ai-1j - 4 aij
    const laplace = gpu.createKernel(function(a) {
        var sum = -4 * a[this.thread.y][this.thread.x];
        var ip;
        for (var i=0; i<2; i++) {
            ip = (this.thread.x + i*2 - 1) % this.constants.size;
            sum += a[this.thread.y][ip];
        }
        for (var i=0; i<2; i++) {
            ip = (this.thread.y + i*2 - 1) % this.constants.size;
            sum += a[ip][this.thread.x];
        }
        return sum;
    })
    .setConstants({size: matrixSize})
    .setOutput([matrixSize, matrixSize]);
    //.setFloatTextures(true);

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

    //laplace_relaxation = gpu.combineKernels(laplace, scale, add, function(a) {
	//    return add(a, scale(0.1, laplace(a)));
	//});

    //laplace_relaxation.setOutputToTexture(true);
    
    function animate() {
        //for(var i=0; i<10; i++){
            //phi = laplace_relaxation(phi);
            phi = add(phi, scale(0.24, laplace(phi)));
        //};
        var sum = sum_row(phi)
        console.log(sum.reduce(function(a,c){return a+c;}));
        render(phi);
        window.requestAnimationFrame(animate)
    }

    window.requestAnimationFrame(animate)


    function fillArrayRandom(array) {
        for(var i = 0; i < array.length; i++) {
          array[i] = Math.random();
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

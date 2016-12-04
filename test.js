
/* Utility functions to generate arbitrary input in various formats */

function inputReals(size) {
    var result = new Float32Array(size);
    for (var i = 0; i < result.length; i++)
	result[i] = (i % 2) / 4.0;
    return result;
}

function zeroReals(size) {
    var result = new Float32Array(size);
    for (var i = 0; i < result.length; i++)
	result[i] = 0.0;
    return result;
}

function inputInterleaved(size) {
    var result = new Float32Array(size*2);
    for (var i = 0; i < size; i++)
	result[i*2] = (i % 2) / 4.0;
    return result;
}

function inputReal64s(size) {
    var result = new Float64Array(size);
    for (var i = 0; i < result.length; i++)
	result[i] = (i % 2) / 4.0;
    return result;
}

function zeroReal64s(size) {
    var result = new Float64Array(size);
    for (var i = 0; i < result.length; i++)
	result[i] = 0.0;
    return result;
}

function inputComplexArray(size) {
    var result = new complex_array.ComplexArray(size);
    for (var i = 0; i < size; i++) {
	result.real[i] = (i % 2) / 4.0;
	result.imag[i] = 0.0;
    }
    return result;
}

function Trig(){
  var pi = 3.1415926535898;
  var tau = 6.2831853071796;
  var B = 1.2732395; // 4/pi
  var C = -0.40528473; // -4 / (pi²)
    
  var modRad = function(theta) {
    // to the range -pi <= theta < pi
    var remain = (theta + pi) % tau;
    //return (remain < 0 ? Math.abs(remain) + tau : remain) - pi;
    return ( ( (theta + pi) % tau) + tau) % tau - pi;
  }

  this.fastSin = function(theta) {
      // See  for graph and equations
      // https://www.desmos.com/calculator/8nkxlrmp7a
      // logic explained here : http://devmaster.net/posts/9648/fast-and-accurate-sine-cosine			
      theta = modRad(theta); // +- pi
      // 1.5707963267949; // pi/s
      var B = 1.2732395; // 4/pi
      var C = -0.40528473; // -4 / (pi²)

      return B*theta + C * theta * Math.abs(theta);
  }

  this.fastCos = function(theta) {
      return fastSin(theta - 1.5707963267949); // pi/2
  }
}

trig = new Trig()

// 
//  function testNative(){
//    var buffer;
//    for (var i = inputs.length-1; i >=0; i--) {
//      buffer = nativeSin(inputs[i]);
//    }
//  }

var iterations = 5;

function report(name, start, middle, end, total) {
    function addTo(tag, thing) {
	document.getElementById(name + "-" + tag).innerHTML += thing + "<br>";
    }
    addTo("result", total);
    addTo("1", Math.round(middle - start) + " ms");
    addTo("2", Math.round(end - middle) + " ms");
    addTo("itr", Math.round((1000.0 /
			     ((end - middle) / iterations))) + " itr/sec");
}

function testKissFFTNDR(size) {

    var fft = new KissFFTNDR([size,size]);
    
    var start = performance.now();
    var middle = start;
    var end = start;

    total = 0.0;

    // frequency weights
    // W.^i + W.^-i + W.^j + W.^-j - 4;
    // W = e^(2*pi*i/N)
    // dim[0]*dim[1]*...*dim[dims-1]/2 + 1 complex points
    weights = new Float32Array(size*(size/2+1)*2);
    W = math.exp(math.complex(0,2*math.pi/size))
    for (var i = 0; i < size; ++i) {
      x = 2*i*(size/2+1);
      for (var j = 0; j < size/2 + 1; ++j){
        y=2*j;
        weight = math.inv(math.multiply(size*size,math.sum([math.pow(W,i),math.pow(W,-i),math.pow(W,j),math.pow(W,-j),-4])))
        weights[x + y] = weight.re;
        weights[x + y + 1] = weight.im;
      }
    }

    weights[0] = 0
    weights[1] = 0

    var ri = zeroReals(size*size);
    var wre;
    var wim;
    var rre;
    var rim;

    var canvas = document.getElementById('myCanvas');
    var context = canvas.getContext('2d');
    var imageObj = new Image();

    var imageData = context.getImageData(0,0,size*2, size*2);
    var data = imageData.data;

    i = size/2;
    j = size/2;
    ri[i*size + j] = 1

    i = 3*size/4;
    j = 3*size/4;
    ri[i*size + j] = 1

    i = 2*size/4;
    j = 1*size/4;
    ri[i*size + j] = 1

// start sim
    for (var k=0; k < iterations*2; ++k){

    var R = fft.forward(ri);
    
    for (var i = 0; i < size; ++i) {
      x = 2*i*(size/2+1);
      for (var j = 0; j < size/2 + 1; ++j){
        y=2*j;
        wre = weights[x + y];
        wim = weights[x + y + 1];
        rre = R[x + y];
        rim = R[x + y + 1];
        R[x + y] = wre*rre - wim * rim;
        R[x + y + 1] = wre*rim + wim*rim;
      }
    }
    
    var r = fft.inverse(R);

    if (k == iterations) {
        middle = performance.now();
    }

	  var out = R;

    var vmax = r.reduce(function(a,b){return Math.max(a,b)},null);
    var vmin = r.reduce(function(a,b){return Math.min(a,b)},null);

    sinfunc = trig.fastSin
    for(var y=0; y<size; y++) {
      i = y*size;
      for(var j=0; j<size; j++) {
        val = (r[i + j] - vmin)/(vmax - vmin)
        frequency = math.pi*2*16;

        red =sinfunc(frequency*val + k + 0) * 127 + 128;
        green = sinfunc(frequency*val + k + 2) * 127 + 128;
        blue  = sinfunc(frequency*val + k + 4) * 127 + 128;

        p = (i << 3) + (j << 2);
        data[p] = red;
        data[p + 1] = green;
        data[p + 2] = blue;
        data[p + 3] = 255;
      }
    }
    context.putImageData(imageData, 0, 0);

    }

    //imageObj.src = 'http://www.html5canvastutorials.com/demos/assets/darth-vader.jpg';

    var end = performance.now();
    
    report("kissfftndr", start, middle, end, total);

    fft.dispose();
}

//var x =  0;
//var y = 15;
//var speed = 5;

//function animate() {

//    reqAnimFrame = window.mozRequestAnimationFrame    ||
//                window.webkitRequestAnimationFrame ||
//                window.msRequestAnimationFrame     ||
//                window.oRequestAnimationFrame
//                ;

//    reqAnimFrame(animate);

//    x += speed;

//    if(x <= 0 || x >= 475){
//        speed = -speed;
//    }

//    draw();
//}


//function draw() {
//    var canvas  = document.getElementById("ex1");
//    var context = canvas.getContext("2d");

//    context.clearRect(0, 0, 500, 170);
//    context.fillStyle = "#ff00ff";
//    context.fillRect(x, y, 25, 25);
//}

//animate();

//  var nativeSin = function(inValue) {
//    return Math.sin(inValue);
//  }
//  var fastSin = function(inValue) {
//    // See  for graph and equations
//    // https://www.desmos.com/calculator/8nkxlrmp7a
//    // logic explained here : http://devmaster.net/posts/9648/fast-and-accurate-sine-cosine			
//    var B = 1.2732395; // 4/pi
//    var C = -0.40528473; // -4 / (pi²)
//  			
//    if (inValue > 0) {
//      return B*inValue - C * inValue*inValue;
//    }
//    return B*inValue + C * inValue*inValue;
//  }
//  
//  
//  function testNative(){
//    var buffer;
//    for (var i = inputs.length-1; i >=0; i--) {
//      buffer = nativeSin(inputs[i]);
//    }
//  }

function testKissFFT(size) {

    var fft = new KissFFTR(size);
    
    var start = performance.now();
    var middle = start;
    var end = start;

    total = 0.0;

    for (var i = 0; i < 2*iterations; ++i) {
	if (i == iterations) {
	    middle = performance.now();
	}
	var ri = inputReals(size);
	var out = fft.forward(ri);
	for (var j = 0; j <= size/2; ++j) {
	    total += Math.sqrt(out[j*2] * out[j*2] + out[j*2+1] * out[j*2+1]);
	}
	// KissFFTR returns only the first half of the output (plus
	// DC/Nyquist) -- synthesise the conjugate half
	for (var j = 1; j < size/2; ++j) {
	    total += Math.sqrt(out[j*2] * out[j*2] + out[j*2+1] * out[j*2+1]);
	}
    }

    var end = performance.now();
    
    report("kissfft", start, middle, end, total);

    fft.dispose();
}

function testKissFFTCC(size) {

    var fft = new KissFFT(size);
    
    var start = performance.now();
    var middle = start;
    var end = start;

    total = 0.0;

    for (var i = 0; i < 2*iterations; ++i) {
	if (i == iterations) {
	    middle = performance.now();
	}
	var cin = inputInterleaved(size);
	var out = fft.forward(cin);
	for (var j = 0; j < size; ++j) {
	    total += Math.sqrt(out[j*2] * out[j*2] + out[j*2+1] * out[j*2+1]);
	}
    }

    var end = performance.now();
    
    report("kissfftcc", start, middle, end, total);

    fft.dispose();
}


var sizes = [256];//, 2048];
var tests = [testKissFFTNDR];
var nextTest = 0;
var nextSize = 0;
var interval;

function test() {
    clearInterval(interval);
    if (nextTest == tests.length) {
	nextSize++;
	nextTest = 0;
	if (nextSize == sizes.length) {
	    return;
	}
    }
    f = tests[nextTest];
    size = sizes[nextSize];
    nextTest++;
    f(size);
    interval = setInterval(test, 100);
}

window.onload = function() {
    document.getElementById("test-description").innerHTML =
	"Running " + 2*iterations + " iterations per implementation.<br>Timings are given separately for the first half of the run (" + iterations + " iterations) and the second half, in case the JS engine takes some warming up.<br>Each cell contains results for the following sizes: " + sizes;
    interval = setInterval(test, 100);
}


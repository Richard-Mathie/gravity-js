function poissonSolver(size) {

    var fft = new KissFFTNDR([size,size]);

    // frequency weights
    // W.^i + W.^-i + W.^j + W.^-j - 4;
    // W = e^(2*pi*i/N)
    // dim[0]*dim[1]*...*dim[dims-1]/2 + 1 complex points
    var weights = new Float32Array(size*(size/2+1)*2);

    (function calculateWeights(){
      var W = math.exp(math.complex(0,2*math.pi/size))
      for (var i = 0; i < size; ++i) {
        x = 2*i*(size/2+1);
        for (var j = 0; j < size/2 + 1; ++j){
          y=2*j;
          weight = math.inv(
                   math.multiply( size*size,
                   math.sum(
                   [math.pow(W,i),math.pow(W,-i),math.pow(W,j),math.pow(W,-j),-4]
                   )
                 )
               );
          weights[x + y] = weight.re;
          weights[x + y + 1] = weight.im;
        }
      }

      weights[0] = 0
      weights[1] = 0
    })();

    this.solve function(r){
      // r must be a Float32Array[size*size] real array 
      var R = fft.forward(ri);
      // multiply R by weights
      // C matrexes are in row first ordrer
      // frequency domain last dim size/2 + 1
      for (var i = 0; i < size; ++i) {
        x = 2*i*(size/2+1);
        for (var j = 0; j < size/2 + 1; ++j){
          y=2*j;
          // complex number multiplication
          wre = weights[x + y];
          wim = weights[x + y + 1];
          rre = R[x + y];
          rim = R[x + y + 1];
          R[x + y] = wre*rre - wim * rim;
          R[x + y + 1] = wre*rim + wim*rim;
        }
      }
      return fft.inverse(R);
    };

};



    var vmax = r.reduce(function(a,b){return Math.max(a,b)},null);
    var vmin = r.reduce(function(a,b){return Math.min(a,b)},null);

    for(var y=0; y<size; y++) {
      i = y*size;
      for(var j=0; j<size; j++) {
        val = (r[i + j] - vmin)/(vmax - vmin)
        frequency = math.pi*2*4;
        red = Math.sin(frequency*val + k + 0) * 127 + 128;
        green = Math.sin(frequency*val + k + 2) * 127 + 128;
        blue  = Math.sin(frequency*val + k + 4) * 127 + 128;

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

//var can2 = document.createElement('canvas');
//can2.width = w/2;
//can2.height = w/2;
//var ctx2 = can2.getContext('2d');

//ctx2.drawImage(img, 0, 0, w/2, h/2);
//ctx2.drawImage(can2, 0, 0, w/2, h/2, 0, 0, w/4, h/4);
//ctx2.drawImage(can2, 0, 0, w/4, h/4, 0, 0, w/6, h/6);

//ctx.drawImage(can2, 0, 0, w/6, h/6, 0, 200, w/6, h/6);

(function(){
  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');

  // resize the canvas to fill browser window dynamically
  window.addEventListener('resize', resizeCanvas, false);

  var img = new Image();
  var wmax = 512*2;
  var hmax = 512*2;
  var w
  var h

  var x =  0;
  var y = 15;
  var speedx = 5;
  var speedy = 5;


  function animate() {

//    reqAnimFrame = window.mozRequestAnimationFrame    ||
//                window.webkitRequestAnimationFrame ||
//                window.msRequestAnimationFrame     ||
//                window.oRequestAnimationFrame
//                ;

    requestAnimationFrame(animate);

    x += speedx;
    y += speedy;
    x = Math.min(Math.max(x,0),canvas.width)
    y = Math.min(Math.max(y,0),canvas.height)

    if(x <= 0 || x >= canvas.width){
        speedx = -speedx;
    }
    if(y <= 0 || y >= canvas.height){
        speedy = -speedy;
    }

    draw();
  }


  function draw() {
    //ctx.clearRect(0, 0, 500, 170);
    ctx.fillStyle = "#ff00ff";
    ctx.fillRect(x, y, 25, 25);
  }

  animate();

  function resizeCanvas() {
    canvas.width = Math.min(document.body.offsetWidth, wmax);
    canvas.height = Math.min(document.body.offsetHeight, document.body.offsetWidth, hmax);
    w = canvas.width;
    h = canvas.height;
    /**
     * Your drawings need to be inside this function otherwise they will be reset when 
     * you resize the browser window and the canvas goes will be cleared.
     */
    drawStuff(); 
  }
  //resizeCanvas();

  function drawStuff() {
    // do your drawing stuff here
    // step it down only once to 1/6 size:
    ctx.drawImage(img, 0, 0, w, h);   
    
    // Step it down several times
    var can2 = document.createElement('canvas');
    can2.width = w/2;
    can2.height = w/2;
    var ctx2 = can2.getContext('2d');
    
    // Draw it at 1/2 size 3 times (step down three times)
    
    ctx2.drawImage(img, 0, 0, w, h);
    //ctx2.drawImage(can2, 0, 0, w, h, 0, 0, w, h);
    //ctx2.drawImage(can2, 0, 0, w, h, 0, 0, w, h);
    ctx.drawImage(can2, 0, 0, w, h, 0, 200, w, h);
  }

  img.onload = resizeCanvas
  img.src = '/home/richard/work/experimental/gravityjs/poisson2dv2.png'
})();

"use strict";

var kissFFTModule = KissFFTModule({});

var kiss_fftr_alloc = kissFFTModule.cwrap(
    'kiss_fftr_alloc', 'number', ['number', 'number', 'number', 'number' ]
);

var kiss_fftr = kissFFTModule.cwrap(
    'kiss_fftr', 'void', ['number', 'number', 'number' ]
);

var kiss_fftri = kissFFTModule.cwrap(
    'kiss_fftri', 'void', ['number', 'number', 'number' ]
);

var kiss_fft_alloc = kissFFTModule.cwrap(
    'kiss_fft_alloc', 'number', ['number', 'number', 'number', 'number' ]
);

var kiss_fft = kissFFTModule.cwrap(
    'kiss_fft', 'void', ['number', 'number', 'number' ]
);

var kiss_free = kissFFTModule.cwrap(
    'free', 'void', ['number']
);


//kiss_fftndr_cfg  kiss_fftndr_alloc(const int *dims,int ndims,int inverse_fft,void*mem,size_t*lenmem);;
var kiss_fftndr_alloc = kissFFTModule.cwrap(
    'kiss_fftndr_alloc', 'number', ['number', 'number', 'number', 'number', 'number']
);
//void kiss_fftndr( kiss_fftndr_cfg cfg, const kiss_fft_scalar *timedata, kiss_fft_cpx *freqdata);
var kiss_fftndr = kissFFTModule.cwrap(
    'kiss_fftndr', 'void', ['number', 'number', 'number' ]
);
//void kiss_fftndri(
var kiss_fftndri = kissFFTModule.cwrap(
    'kiss_fftndri', 'void', ['number', 'number', 'number' ]
);


function KissFFT(size) {

    this.size = size;
    this.fcfg = kiss_fft_alloc(size, false);
    this.icfg = kiss_fft_alloc(size, true);
    
    this.inptr = kissFFTModule._malloc(size*8 + size*8);
    this.outptr = this.inptr + size*8;
    
    this.cin = new Float32Array(kissFFTModule.HEAPU8.buffer, this.inptr, size*2);
    this.cout = new Float32Array(kissFFTModule.HEAPU8.buffer, this.outptr, size*2);
    
    this.forward = function(cin) {
	this.cin.set(cin);
	kiss_fft(this.fcfg, this.inptr, this.outptr);
	return new Float32Array(kissFFTModule.HEAPU8.buffer,
				this.outptr, this.size * 2);
    }
    
    this.inverse = function(cin) {
	this.cin.set(cpx);
	kiss_fft(this.icfg, this.inptr, this.outptr);
	return new Float32Array(kissFFTModule.HEAPU8.buffer,
				this.outptr, this.size * 2);
    }
    
    this.dispose = function() {
	kissFFTModule._free(this.inptr);
	kiss_free(this.fcfg);
	kiss_free(this.icfg);
    }
}

function KissFFTR(size) {

    this.size = size;
    this.fcfg = kiss_fftr_alloc(size, false);
    this.icfg = kiss_fftr_alloc(size, true);
/*
 input timedata has nfft scalar points
 output freqdata has nfft/2+1 complex points
*/   
    this.rptr = kissFFTModule._malloc(size*4 + (size+2)*4);
    this.cptr = this.rptr + size*4;
    
    this.ri = new Float32Array(kissFFTModule.HEAPU8.buffer, this.rptr, size);
    this.ci = new Float32Array(kissFFTModule.HEAPU8.buffer, this.cptr, size+2);
    
    this.forward = function(real) {
	this.ri.set(real);
	kiss_fftr(this.fcfg, this.rptr, this.cptr);
	return new Float32Array(kissFFTModule.HEAPU8.buffer,
				this.cptr, this.size + 2);
    }
    
    this.inverse = function(cpx) {
	this.ci.set(cpx);
	kiss_fftri(this.icfg, this.cptr, this.rptr);
	return new Float32Array(kissFFTModule.HEAPU8.buffer,
				this.rptr, this.size);
    }
    
    this.dispose = function() {
	kissFFTModule._free(this.rptr);
	kiss_free(this.fcfg);
	kiss_free(this.icfg);
    }
}

function KissFFTNDR(size) {
/*
 size is dims[ndims-1]
*/
    this.size = size;

    this.dimsptr = kissFFTModule._malloc(size.length*4);
    this.dims = new Int32Array(kissFFTModule.HEAPU8.buffer, this.dimsptr, size.length);
    this.dims.set(size)

    this.fcfg = kiss_fftndr_alloc(this.dimsptr, size.length, false, 0, 0);
    this.icfg = kiss_fftndr_alloc(this.dimsptr, size.length, true, 0, 0);

    this.N_real = size.reduce(function(l,e){return l*e}, 1);
    this.N_complex = this.N_real*(1/2 + 1/size[size.length-1]); // N_complex*(2*4) = dims[0]*dims[1]*...*(dims[ndims-1]/2+1) * 4 * 2
/*
 input timedata has dims[0] X dims[1] X ... X  dims[ndims-1] scalar points
 output freqdata has dims[0] X dims[1] X ... X  dims[ndims-1]/2+1 complex points
*/
    this.rptr = kissFFTModule._malloc(this.N_real*4);
    this.cptr = kissFFTModule._malloc(this.N_complex*4*2);
   
    this.ri = new Float32Array(kissFFTModule.HEAPU8.buffer, this.rptr, this.N_real);
    this.ci = new Float32Array(kissFFTModule.HEAPU8.buffer, this.cptr, this.N_complex*2);
    
    this.forward = function(real) {
	this.ri.set(real);
	kiss_fftndr(this.fcfg, this.rptr, this.cptr);
	return new Float32Array(kissFFTModule.HEAPU8.buffer,
				this.cptr, this.N_complex*2);
    }
    
    this.inverse = function(cpx) {
	this.ci.set(cpx);
	kiss_fftndri(this.icfg, this.cptr, this.rptr);
	return new Float32Array(kissFFTModule.HEAPU8.buffer,
				this.rptr, this.N_real);
    }
    
    this.dispose = function() {
	kissFFTModule._free(this.rptr);
	kissFFTModule._free(this.cptr)
	kiss_free(this.fcfg);
	kiss_free(this.icfg);
    }
}



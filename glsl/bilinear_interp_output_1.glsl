precision highp float;
precision highp int;
precision highp sampler2D;

const float LOOP_MAX = 1000.0;
#define EPSILON 0.0000001;

uniform highp vec3 uOutputDim;
uniform highp vec2 uTexSize;

varying highp vec2 vTexCoord;

vec4 round(vec4 x) {
  return floor(x + 0.5);
}

highp float round(highp float x) {
  return floor(x + 0.5);
}

vec2 integerMod(vec2 x, float y) {
  vec2 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}

vec3 integerMod(vec3 x, float y) {
  vec3 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}

vec4 integerMod(vec4 x, vec4 y) {
  vec4 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}

highp float integerMod(highp float x, highp float y) {
  highp float res = floor(mod(x, y));
  return res * (res > floor(y) - 1.0 ? 0.0 : 1.0);
}

highp int integerMod(highp int x, highp int y) {
  return int(integerMod(float(x), float(y)));
}

// Here be dragons!
// DO NOT OPTIMIZE THIS CODE
// YOU WILL BREAK SOMETHING ON SOMEBODY'S MACHINE
// LEAVE IT AS IT IS, LEST YOU WASTE YOUR OWN TIME
const vec2 MAGIC_VEC = vec2(1.0, -256.0);
const vec4 SCALE_FACTOR = vec4(1.0, 256.0, 65536.0, 0.0);
const vec4 SCALE_FACTOR_INV = vec4(1.0, 0.00390625, 0.0000152587890625, 0.0); // 1, 1/256, 1/65536
highp float decode32(highp vec4 rgba) {
  rgba *= 255.0;
  vec2 gte128;
  gte128.x = rgba.b >= 128.0 ? 1.0 : 0.0;
  gte128.y = rgba.a >= 128.0 ? 1.0 : 0.0;
  float exponent = 2.0 * rgba.a - 127.0 + dot(gte128, MAGIC_VEC);
  float res = exp2(round(exponent));
  rgba.b = rgba.b - 128.0 * gte128.x;
  res = dot(rgba, SCALE_FACTOR) * exp2(round(exponent-23.0)) + res;
  res *= gte128.y * -2.0 + 1.0;
  return res;
}

highp vec4 encode32(highp float f) {
  highp float F = abs(f);
  highp float sign = f < 0.0 ? 1.0 : 0.0;
  highp float exponent = floor(log2(F));
  highp float mantissa = (exp2(-exponent) * F);
  // exponent += floor(log2(mantissa));
  vec4 rgba = vec4(F * exp2(23.0-exponent)) * SCALE_FACTOR_INV;
  rgba.rg = integerMod(rgba.rg, 256.0);
  rgba.b = integerMod(rgba.b, 128.0);
  rgba.a = exponent*0.5 + 63.5;
  rgba.ba += vec2(integerMod(exponent+127.0, 2.0), sign) * 128.0;
  rgba = floor(rgba);
  rgba *= 0.003921569; // 1/255
  return rgba;
}
// Dragons end here

highp float index;
highp vec3 threadId;

highp vec3 indexTo3D(highp float idx, highp vec3 texDim) {
  highp float z = floor(idx / (texDim.x * texDim.y));
  idx -= z * texDim.x * texDim.y;
  highp float y = floor(idx / texDim.x);
  highp float x = integerMod(idx, texDim.x);
  return vec3(x, y, z);
}

highp float get(highp sampler2D tex, highp vec2 texSize, highp vec3 texDim, highp float z, highp float y, highp float x) {
  highp vec3 xyz = vec3(x, y, z);
  xyz = floor(xyz + 0.5);
  highp float index = round(xyz.x + texDim.x * (xyz.y + texDim.y * xyz.z));
  highp float w = round(texSize.x);
  vec2 st = vec2(integerMod(index, w), float(int(index) / int(w))) + 0.5;
  highp vec4 texel = texture2D(tex, st / texSize);
  return decode32(texel);
}

highp float get(highp sampler2D tex, highp vec2 texSize, highp vec3 texDim, highp float y, highp float x) {
  return get(tex, texSize, texDim, 0.0, y, x);
}

highp float get(highp sampler2D tex, highp vec2 texSize, highp vec3 texDim, highp float x) {
  return get(tex, texSize, texDim, 0.0, 0.0, x);
}

highp vec4 actualColor;
void color(float r, float g, float b, float a) {
  actualColor = vec4(r,g,b,a);
}

void color(float r, float g, float b) {
  color(r,g,b,1.0);
}

uniform highp sampler2D user_f;
uniform highp vec2 user_fSize;
uniform highp vec3 user_fDim;
const float constants_size = 4.0;
highp float kernelResult = 0.0;
void kernel() {
float user_tsize=constants_size;
float user_x=(threadId.x*0.5);
float user_y=(threadId.y*0.5);
float user_ix=floor(((user_x*0.5)-0.5));
float user_iy=floor(((user_y*0.5)-0.5));
float user_a=((user_x+0.25)-(user_ix+0.5));
float user_b=((user_y+0.25)-(user_iy+0.5));
float user_ix0=mod(user_ix, user_tsize);
float user_ix1=mod((user_ix+1.0), user_tsize);
float user_iy0=mod(user_iy, user_tsize);
float user_iy1=mod((user_iy+1.0), user_tsize);
float user_p0q0=get(user_f, vec2(user_fSize[0],user_fSize[1]), vec3(user_fDim[0],user_fDim[1],user_fDim[2]), user_ix0,user_iy0);
float user_p1q0=get(user_f, vec2(user_fSize[0],user_fSize[1]), vec3(user_fDim[0],user_fDim[1],user_fDim[2]), user_ix1,user_iy0);
float user_p0q1=get(user_f, vec2(user_fSize[0],user_fSize[1]), vec3(user_fDim[0],user_fDim[1],user_fDim[2]), user_ix0,user_iy1);
float user_p1q1=get(user_f, vec2(user_fSize[0],user_fSize[1]), vec3(user_fDim[0],user_fDim[1],user_fDim[2]), user_ix1,user_iy1);
float user_pInterp_q0=mix(user_p0q0, user_p1q0, user_a);
float user_pInterp_q1=mix(user_p0q1, user_p1q1, user_a);
kernelResult = mix(user_pInterp_q0, user_pInterp_q1, user_b);return;
}
void main(void) {
  index = floor(vTexCoord.s * float(uTexSize.x)) + floor(vTexCoord.t * float(uTexSize.y)) * uTexSize.x;
  threadId = indexTo3D(index, uOutputDim);
  kernel();
  gl_FragColor = encode32(kernelResult);
}

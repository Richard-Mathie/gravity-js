N = 2**8;
M = 1000;

g = zeros(N,N,N);
W = e^(2*pi*i/N);


px = ceil(rand(1,M)*N);
py = ceil(rand(1,M)*N);
pz = ones(1,M) * round(N/2);
p = sub2ind([N,N,N], px,py,pz);
#g(:,:,N/2) = rand(N,N)*0.01;


g(p) = 1;

G = fftn(g);
k = 0:(N-1);
l = k;
m = l;

[k,l,m] = meshgrid(k,l,m);

weights = W.^k + W.^-k + W.^l + W.^-l + W.^m + W.^-m - 6;

R = G./weights;
R(1) = 0;
v = real(ifftn(R));
imagesc(v(:,:,round(N/2)))

#dvx = diff_2O(v);
#dvy = diff_2O(v')';

#Fmag = sqrt(dvy.^2 + dvx.^2);

#dvx(p)
#dvy(p)

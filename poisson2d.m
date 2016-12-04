N = 4; 2**9;
M = 2;

g = zeros(N,N);
W = e^(2*pi*i/N);


#px = ceil(mod(randn(1,M)*N/2/2+N/2,N));
#py = ceil(mod(randn(1,M)*N/2/2+N/2,N));
px = [N/2];#,3*N/4]#ceil(rand(1,M)*N);
py = [N/2]+1;#,N/2+100]#ceil(rand(1,M)*N);
p = sub2ind([N,N], px,py);

g(p) = 1;

G = fft2(g);
k = 0:(N-1);
l = k;

[k,l] = meshgrid(k,l);

weights = 1./(W.^k + W.^-k + W.^l + W.^-l - 4);
weights(1) = 0

R = G.*weights;
v = real(ifft2(R));

dvy = diff_2O(v);
dvx = diff_2O(v')';
Fmag = sqrt(dvy.^2 + dvx.^2);

fx = dvx(p);
fy = dvy(p);

clf 
imagesc(v')
axis(axis(),'square')
hold on
plot(px,py,'.')
quiver(px,py,-fx,-fy)

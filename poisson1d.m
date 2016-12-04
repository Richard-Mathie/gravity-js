N = 1000;
g = zeros(1,N);
W = e^(2*pi*i/N);

g(ceil(N/3)) = 1;
g(ceil(2*N/3)) = 1;
G = fft(g);
k = 0:(N-1);
weights = W.^k + W.^-k - 2;

R = - G./weights;
R(1) = 0;
r = real(ifft(R));
clf
subplot(3,1,1)
plot(r)
subplot(3,1,2)
plot(diff_2O(r))
subplot(3,1,3)
plot(diff_2O(diff_2O(r)))

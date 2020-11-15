function y = DoG_wavelet(x,s,tau)

x =(x-tau)/s;
y= (1/sqrt(s)) *(2/sqrt(3)) *pi^-0.25 * (1-x^2)*exp(-x^2);


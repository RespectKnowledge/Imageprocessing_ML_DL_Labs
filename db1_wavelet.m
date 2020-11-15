function y = db1_wavelet(x,tau,s)

arg=x; %(2^s-tau); %(x-tau)/(2^s); %(2^j)*x-i;

if arg >= 0 && arg<0.5
    y=1;
elseif arg>=0.5 && arg<=1
    y=-1;
else
    y=0;
end

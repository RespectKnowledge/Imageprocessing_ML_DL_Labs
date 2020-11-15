
function y=Meyer_wavelet(t)

temp=hcurve_Meyer(t);
y=sqrt(2)*(sin((pi*t)/3)/(2*t) + (1/pi) * temp);

end
function f = hcurve_Meyer(t)

for w=pi/2:0.01:2*pi/4
    f=cos( (pi/2)* beta_fun((3*w)/pi -1))*cos(w*t);
end

end

function beta_out= beta_fun(x)
 beta_out= x.^5*(126-420*x+540*x^2-315*x^3+70*x^4);
end
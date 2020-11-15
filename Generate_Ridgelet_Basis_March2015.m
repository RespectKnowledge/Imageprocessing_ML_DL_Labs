clear all; close all; clc;

basis=[];

M=8; N=M;  % size of path (M by N)
%% 2-D Ridgelet Basis
s={}; temp=zeros(M,N); 
Dl=zeros(M^2,10); 
count1=0;  count=1;
for s1=0.2:0.4:M-1
    s1
        count1=count1+1;
        count2=0;
        for tau1=0.1:0.2:M-1
                for theta=-pi:0.05:pi
                    count2=count2+1;
                    for n1=1:M
                        for n2=1:N                          
                            t=(n1*cos(theta)+n2*sin(theta)-tau1)/s1;
                            %t2=(n2-tau2)/s2;
%                             temp(n1,n2)= 2^(s1/2)*
%                             db1_wavelet(t,tau1,s1); % db1  % (2^s-tau);%(x-tau)/(2^s); %(2^j)*x-i;
%                              temp(n1,n2)= s1^(1/2)*Meyer_wavelet(t);
                            temp(n1,n2)= s1^(1/2)* sin(5*t)*exp(-t^2/2); %2^(s1/2) % Morelet wavelet
%                              temp(n1,n2) = (2/sqrt(3)*pi^(-0.25))*(1-t^2)*exp(-t^2/2); % Maxican Hat
%                             temp(n1,n2) =2^(s1/2)*DoG_wavelet(n1,s1,t); %*2^(s2/2)*DoG_wavelet(n2,s2,tau2);  % DoG
%                             basis{count1, count2}=temp;
                        end
                    end
 
                    if sum(abs(temp(:)))~=0
                        Dl(:,count)=temp(:);
                        count=count+1;
                    end
                end
        end
end


xp=randperm(size(Dl,2));
Dl=Dl(:,xp); 

%% Select bases having variance greater than a certain threshold
threshold=1.2; %M^2*0.05;
pvars = var(Dl, 0, 1);

idx = pvars > threshold;

Dl = Dl(:, idx);

dict_size   = 1024;          % dictionary size
Dl=Dl(:,1:dict_size );
imshow(Dl)

dict_path = ['D_' num2str(dict_size) '_' num2str(M) '.mat' ];

save(dict_path, 'Dl');


 


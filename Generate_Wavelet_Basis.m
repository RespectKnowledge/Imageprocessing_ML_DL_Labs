clear all; close all; clc;

basis=[];


M2=8; N2=8;
M=3; N=3; % patch size
sigma=3;

%% 2-D Wavelet Basis
s={};
count1=0; count2=0;
for s1=1:6    
    for s2=1:6
        count1=count1+1;
        count2=0;
        for tau1=0:0.1:0.2
            for tau2=0:0.1:0.2
                count2=count2+1;
                for n1=1:M
                    for n2=1:N
                        t1=(n1-tau1)/s1;
                        t2=(n2-tau2)/s2;
                       temp(n1,n2)= 2^(s1/2)* sin(5*t1)*exp(-t1^2/2)*2^(s2/2)* sin(5*t2)*exp(-t2^2/2); %2^(s/2) % Morelet wavelet
%                         temp(n1,n2) =2^(s1/2)*DoG_wavelet(n1,s1,tau1)*2^(s2/2)*DoG_wavelet(n2,s2,tau2);  % DoG
%                         temp(n1,n2)=(4/(36*pi^0.5))*(1-t1^2/sigma^2)*exp(-t1^2/(2*sigma^2))*... 
%                              (1-t2^2/sigma^2)*exp(-t2^2/(2*sigma^2)); % 
%                           temp(n1,n2)=Meyer_wavelet(t1)* Meyer_wavelet(t2); % Meyer wavelet
                        basis{count1, count2}=temp;
                    end
                end
            end
        end
    end
end
[M1 N1]= size(basis);
count=1;
for i=1:M1
    for j=1:N1
        temp=basis{i,j};
        basis_linear(:,count)=temp(:);
        count=count+1;
    end
end

Dl=basis_linear; clear basis_linear;

xp=randperm(size(Dl,2));
% Dl=Dl(:,xp);

 
DicDorm = sqrt(sum(Dl.^2));
lNorm = sqrt(sum(Dl.^2));
Idx = find(lNorm);
Dl = Dl(:, Idx);

Dl = Dl./repmat(sqrt(sum(Dl.^2)), size(Dl, 1), 1);

DicWavelet=Dl;

% Dl = Dl./repmat(sqrt(sum(Dl.^2)), size(Dl, 1), 1);

% % %%%%%%%%% Displaying Dictionary %%%%%%%%%%
basis=Dl;
count=1;
sZ=8; %floor(sqrt(size(Dl,2)))
d=ones(sZ*(N+1),sZ*(N+1));
for n1=1:sZ
    lx=(n1-1)*(N+1)+1; hx=lx+N-1;
    for n2=1:sZ
        ly=(n2-1)*(N+1)+1; hy=ly+N-1;
        d(lx:hx, ly:hy)=reshape(basis(:, count),N,N);

        count=count+4;
%         if count > 256
%             break
%         end
    end
%     if count > 256
%         break
%     end
end

% d(find(d==1))=max(max(d));

imshow(abs(d),[])
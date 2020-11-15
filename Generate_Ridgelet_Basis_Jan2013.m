clear all; close all; clc;

basis=[];

M=3; N=3;  % size of path (M by N)
%% 2-D Ridgelet Basis
s={}; temp=zeros(M,N); 
Dl=zeros(M^2,10); 
count1=0;  count=1;
for s1=0.2:0.4:M-1
    s1
        count1=count1+1;
        count2=0;
        for tau1=0.1:0.1:M-1
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



% imshow(imresize(temp,4),[])
% figure; imshow(imresize(temp2,2),[])

xp=randperm(size(Dl,2));
Dl=Dl(:,xp); 

 
% lNorm = sqrt(sum(Dl.^2));
% Idx = find(lNorm);
% Dl = Dl(:, Idx);
% Dl = Dl./repmat(sqrt(sum(Dl.^2)), size(Dl, 1), 1);

%% Displaying Dictionaries

% 
% basis=Dl;
% DicRidgelet=Dl;
% 
% count=1;
% d=zeros(256,256);
% for n1=1:32
%     lx=(n1-1)*M+1; hx=lx+M-1;
%     for n2=1:32
%         ly=(n2-1)*M+1; hy=ly+M-1;
%         d(lx:hx, ly:hy)=reshape(basis(:, count),M,N);
% %         if n1==8
% %             n1
% %             reshape(basis(:, count),8,8)
% %         end
%         count=count+1;
%         if count > size(basis,2)
%             break
%         end
%     end
%     if count > size(basis,2)
%         break
%     end
% end
% 
% figure; imshow((d),[])

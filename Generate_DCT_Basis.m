clear all; close all; clc;
%% 2-D DCT Basis (Final)
basis={};
M=16; N=16;
M1=M^2; N1=N^2;
M2=M; N2=M;

count1=0;
for k1=1:0.5:M1
    if k1==1; nf1=sqrt(1/M); else nf1=sqrt(2/M); end
    for k2=1:0.5:N1
        count1=count1+1;
        if k2==1; nf2=sqrt(1/N); else nf2=sqrt(2/N); end
        for n1=1:M2
            for n2=1:N2
                temp(n1,n2)=nf1*nf2*cos(( (2*(n1-1)+1)*(k1-1)*pi)/(2*M)) * cos(( (2*(n2-1)+1)*(k2-1)*pi)/(2*N));
            end
        end
        Dl(:,count1)=temp(:);
    end
end


xp=randperm(size(Dl,2));
Dl=Dl(:,xp);
DicDorm = sqrt(sum(Dl.^2));
lNorm = sqrt(sum(Dl.^2));
Idx = find(lNorm);
Dl = Dl(:, Idx);
% % Dl = Dl./repmat(sqrt(sum(Dl.^2)), size(Dl, 1), 1);
dict_size   = 1024;          % dictionary size
Dl=Dl(:,1:dict_size );
% %%%%%%%%% Displaying Dictionary %%%%%%%%%%
basis=Dl;
count=1;
d=zeros(128,128);
for n1=1:16
    lx=(n1-1)*N+1; hx=lx+N-1;
    for n2=1:16
        ly=(n2-1)*N+1; hy=ly+N-1;
        d(lx:hx, ly:hy)=reshape(basis(:, count),N,N);
%         if n1==8
%             n1
%             reshape(basis(:, count),8,8)
%         end
        count=count+1;
        if count > 256
            break
        end
    end
    if count > 256
        break
    end
end

imshow(abs(d),[])
%imwrite(d,'dct.png')
    
    
        
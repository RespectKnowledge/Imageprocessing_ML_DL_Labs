
%% compute sparse cofficents using overcmplete dictionary  
D=randn(64,1024); % compute over-complete dictionary with size 200*1024.
% load Dic_Dl_1024_8_woSbtMean.mat
% D=Dl;
patch_size = sqrt(size(Dl, 1));
I=double(imread('left1.png')); % input image
I=left1;
I=new_tumor1841_2;
I=I(1:100,1:100);

[m n]=size(I);
Mean_I1=mean(mean(I))
for ii=1:m-patch_size+1%:m-patch_size+1,
    count=1;
    if (ii>1 && ii<m-patch_size+1)
    for jj = 1:n-patch_size+1%:n-patch_size+1,
        if (jj>1 && jj<m-patch_size+1)
        patch=I(ii:ii+patch_size-1, jj: jj+patch_size-1);
        Mean_patch = mean(patch(:));
        patch1=single(patch-Mean_patch);
        sparse_coeff = SolveOMP(Dl, patch1(:), size(Dl,2),10);
        sparse_coeff1(jj,count)=max(sparse_coeff); % store spares coefficents of first image.Similarly you can store all spare coefficents based on different number of images
        end
    end 
    count=count+1;
    end
end
            
        

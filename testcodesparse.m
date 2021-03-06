
%% compute sparse cofficents using overcmplete dictionary  
D=randn(64,1024); % compute over-complete dictionary with size 200*1024.
% load Dic_Dl_1024_8_woSbtMean.mat
% D=Dl;
patch_size = sqrt(size(D, 1));
I=double(imread('left1.png')); % input image
I=left1;
I=I(1:100,1:100);
[m n]=size(I);
Mean_I1=mean(mean(images))
for ii=1:m-patch_size+1%:m-patch_size+1,
    count=1;
    if (ii>1 && ii<m-patch_size+1)
    for jj = 1:n-patch_size+1%:n-patch_size+1,
        if (jj>1 && jj<m-patch_size+1)
        patch=images(ii:ii+patch_size-1, jj: jj+patch_size-1);
        Mean_patch = mean(patch(:));
        patch1=single(patch-Mean_patch);
        sparse_coeff = SolveOMP(Dl, patch1(:), size(Dl,2),10);
        sparse_coeff1(jj,count)=max(sparse_coeff); % store spares coefficents of first image.Similarly you can store all spare coefficents based on different number of images
        end
    end 
    count=count+1;
    end
end
 
%% # design dictionary with different sizes and load as a matfile.
D1=load('D_1024_8.mat')
D=D1.Dl;
% images ='C:\Users\moona\Desktop\Mylecture\Lab3\dataset\glioma';
% jpgfiles=dir(fullfile(images,'\*.jpg*'))
% n=numel(jpgfiles);
% for jj=1:n
% idx=randi(jj);
% im=jpgfiles(idx).name;
% im1=imread(fullfile(images,im));

Dl=D;
patch_size = sqrt(size(D, 1));
directory = 'C:\Users\moona\Desktop\Mylecture\Lab3\dataset\glioma';
files = dir([directory '/*.jpg']);
for i=1:length(files)
    images= imread([directory '/' files(i).name]);
    [m n]=size(images);
    Mean_I1=mean(mean(images))
    for ii=1:m-patch_size+1%:m-patch_size+1,
    count=1;
    if (ii>1 && ii<m-patch_size+1)
    for jj = 1:n-patch_size+1%:n-patch_size+1,
        if (jj>1 && jj<m-patch_size+1)
        patch=images(ii:ii+patch_size-1, jj: jj+patch_size-1);
        Mean_patch = mean(patch(:));
        patch1=single(patch-Mean_patch);
        sparse_coeff = SolveOMP(Dl, patch1(:), size(Dl,2),10);
        sparse_coeff1(jj,count)=max(sparse_coeff); % store spares coefficents of first image.Similarly you can store all spare coefficents based on different number of images
        end
    end 
    count=count+1;
    end
    end
    sparse_glioma(i,:)=sparse_coeff1;
end
save('sparse_glioma.mat',sparse_glioma)
%%
% D1=load('D_1024_8.mat')
% D=D1.Dl;
% images ='C:\Users\moona\Desktop\Mylecture\Lab3\dataset\glioma';
% jpgfiles=dir(fullfile(images,'\*.jpg*'))
% n=numel(jpgfiles);
% for jj=1:n
% idx=randi(jj);
% im=jpgfiles(idx).name;
% im1=imread(fullfile(images,im));

Dl=D;
patch_size = sqrt(size(D, 1));
directory = 'C:\Users\moona\Desktop\Mylecture\Lab3\dataset\meningioma';
files = dir([directory '/*.jpg']);
for i=1:length(files)
    images= imread([directory '/' files(i).name]);
    [m n]=size(images);
    Mean_I1=mean(mean(images));
    for ii=1:m-patch_size+1%:m-patch_size+1,
    count=1;
    if (ii>1 && ii<m-patch_size+1)
    for jj = 1:n-patch_size+1%:n-patch_size+1,
        if (jj>1 && jj<m-patch_size+1)
        patch=images(ii:ii+patch_size-1, jj: jj+patch_size-1);
        Mean_patch = mean(patch(:));
        patch1=single(patch-Mean_patch);
        sparse_coeff = SolveOMP(Dl, patch1(:), size(Dl,2),10);
        sparse_coeff1(jj,count)=max(sparse_coeff); % store spares coefficents of first image.Similarly you can store all spare coefficents based on different number of images
        end
    end 
    count=count+1;
    end
    end
    sparse_meningioma(i,:)=sparse_coeff1;
end
save('sparse_meningioma.mat','sparse_meningioma')
%%
% D1=load('D_1024_8.mat')
% D=D1.Dl;
% images ='C:\Users\moona\Desktop\Mylecture\Lab3\dataset\glioma';
% jpgfiles=dir(fullfile(images,'\*.jpg*'))
% n=numel(jpgfiles);
% for jj=1:n
% idx=randi(jj);
% im=jpgfiles(idx).name;
% im1=imread(fullfile(images,im));

Dl=D;
patch_size = sqrt(size(D, 1));
directory = 'C:\Users\moona\Desktop\Mylecture\Lab3\dataset\pituitary_tumor';
files = dir([directory '/*.jpg']);
for i=1:length(files)
    images= imread([directory '/' files(i).name]);
    [m n]=size(images);
    Mean_I1=mean(mean(images))
    for ii=1:m-patch_size+1%:m-patch_size+1,
    count=1;
    if (ii>1 && ii<m-patch_size+1)
    for jj = 1:n-patch_size+1%:n-patch_size+1,
        if (jj>1 && jj<m-patch_size+1)
        patch=images(ii:ii+patch_size-1, jj: jj+patch_size-1);
        Mean_patch = mean(patch(:));
        patch1=single(patch-Mean_patch);
        sparse_coeff = SolveOMP(Dl, patch1(:), size(Dl,2),10);
        sparse_coeff1(jj,count)=max(sparse_coeff); % store spares coefficents of first image.Similarly you can store all spare coefficents based on different number of images
        end
    end 
    count=count+1;
    end
    end
    sparse_pituitary_tumor(i,:)=sparse_coeff1;
end
save('sparse_pituitary_tumor.mat','sparse_pituitary_tumor')
%% %%%%%%%%%%%%%%%%%%%%desgin feature matrix for all classes%%%%%%%%%%%%%%%%%
giloma_labes=ones(1,1426)   % 1426 number of labels
meningioma_labes=2*ones(1,708) % 708 number of labes
pituitary_tumor_labes=3*ones(1,930) % 930 number of labels
% concatenate labels for 3 classes
total_labels=[giloma_labes,meningioma_labes,pituitary_tumor_labes];
total_labels=total_labels'
%%%%%%%%%%%%%%%%%%%%%% Feature matrix%%%%%%%%%%%%%%%%%%%%%%%%%%%5
giloma_features=load('sparse_glioma.mat')       % length of matrix should be 1426x504
giloma_features=giloma_features.sparse_glioma
meningioma_features=load('sparse_meningioma.mat')   % % length of matrix should be 708x504
meningioma_features=meningioma_features.sparse_meningioma
pituitary_features=load('sparse_pituitary_tumor.mat')       % length of matrix should be 930x504
pituitary_features=pituitary_features.sparse_pituitary_tumor
% concatenate all features based on three classes
Feature_matrixdata=[giloma_features;meningioma_features;pituitary_features]
%append labels at the last column in the feature matrix dataset
Feature_matrix=[Feature_matrixdata,total_labels]


% giloma_features=load('feature.mat')       % length of matrix should be 1426x504
% giloma_features=giloma_features.sparse_coeff2
% meningioma_features=load('sparse_meningioma.mat')   % % length of matrix should be 708x504
% meningioma_features=meningioma_features.sparse_meningioma
% pituitary_features=load('sparse_pituitary_tumor.mat')       % length of matrix should be 930x504
% pituitary_features=pituitary_features.sparse_pituitary_tumor
% % concatenate all features based on three classes
% Feature_matrixdata=[giloma_features;meningioma_features;pituitary_features]
% total_labes=ones(1,14)
% total_labes=total_labes'
% %append labels at the last column in the feature matrix dataset
% Feature_matrix=[Feature_matrixdata,total_labes]


%%

[m n]=size(I);
Mean_I1=mean(mean(I))
for ii=1:m-patch_size+1%:m-patch_size+1,
    count=1;
    if (ii>1 && ii<m-patch_size+1)
    for jj = 1:n-patch_size+1%:n-patch_size+1,
        if (jj>1 && jj<m-patch_size+1)
        patch=I(ii:ii+patch_size-1, jj: jj+patch_size-1);
        Mean_patch = mean(patch(:));
        patch1=patch-Mean_patch;
        sparse_coeff = SolveOMP(Dl, patch1(:), size(Dl,2),10);
        sparse_coeff1(jj,count)=max(sparse_coeff); % store spares coefficents of first image.Similarly you can store all spare coefficents based on different number of images
        end
    end 
    count=count+1;
    end
end








yourFolder='C:\Users\moona\Desktop\Mylecture\Lab3\dataset\glioma'
for k = 1:1426
  jpgFilename = sprintf('%d.jpg', k);
  fullFileName = fullfile(yourFolder, jpgFilename);
  if exist(fullFileName, 'file')
    imageData = imread(fullFileName );
  else
    warningMessage = sprintf('Warning: image file does not exist:\n%s', fullFileName);
    uiwait(warndlg(warningMessage));
  end
  imshow(imageData);
end








imshow(im1);
















     
        

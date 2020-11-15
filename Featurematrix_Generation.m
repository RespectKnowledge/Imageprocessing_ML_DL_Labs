%% # design dictionary with different sizes and load as a matfile.

% I have generated dictionary of size 64x1024 based on riglet transform,
% You should generate other dictionary with different size and the rest of
% the code will remain same.
% if any issue let me know.  engr.qayyum@gmail.com
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
save('Feature_matrix.mat','Feature_matrix')
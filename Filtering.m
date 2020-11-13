
% read the image
ic=imread('ic.tif')
px=[-1 0 1;-1 0 1;-1 0 1]
icx=filter2(px,ic);
figure,imshow(icx/255)

py=px';
icy=filter2(py,ic);
figure,imshow(icy/255)

edge_p=sqrt(icx.^2+icy.^2);
figure,imshow(edge_p/255)
% edges produced by the threshold say 0.5
edge_t=im2bw(edge_p/255,0.3);
%We can obtain edges by the Prewitt filters directly by using the command
edge_p=edge(ic,'prewitt');
% roberts edge fileter
edge_r=edge(ic,'roberts');
figure,imshow(edge_r)
% sobel edge detector
edge_s=edge(ic,'sobel');
figure,imshow(edge_s)

% Second differences
% We see that the Laplacian (after taking an absolute value, or squaring) gives double edges. It is also
%extremely sensitive to noise. However, the Laplacian does have the advantage of detecting edges in
%all directions equally well. To see an example, suppose we enter the Matlab commands:

l=fspecial('laplacian',0);
ic_l=filter2(l,ic);
figure,imshow(mat2gray(ic_l))
% In Matlab, Laplacians of all sorts can be generated using the fspecial function, in the form
fspecial('laplacian',ALPHA)
% If the parameter ALPHA (which is optional) is omitted, it is assumed to be 0.2
% The value 0 gives the Laplacian developed earlier.

% We now have a further method of edge detection:
% take the zero-crossings after a laplace filtering. This is implemented in Matlab with the zerocross
% option of edge, which takes the zero crossings after filtering with a given filter:
l=fspecial('laplace',0);
icz=edge(ic,'zerocross',l);
imshow(icz)

% This method was designed to provide a edge detection method to be as close as possible to biological
% vision. The first two steps can be combined into one, to produce a Laplacian of Gaussian or LoG
% filter. These filters can be created with the fspecial function. If no extra parameters are provided
% to the zerocross edge option, then the filter is chosen to be the LoG filter found by
fspecial('log',13,2)
%This means that the following command:
edge(ic,'log')
%produces exactly the same result as the commands:
log=fspecial('log',13,2);
edge(ic,'zerocross',log);
% In fact the LoG and zerocross options implement the same edge finding method; the difference
% being that the zerocross option allows you to specify your own filter. 

% Edge enhancement
% So far we have seen how to isolate edges in an image. A related operation is to make edges in an
% image slightly sharper and crisper, which generally results in an image more pleasing to the human
% eye. The operation is variously called edge enhancement, edge crispening, or unsharp masking.
% This last term comes from the printing industry.

% Suppose we have an image x of type uint8. The we can apply unsharp masking by the following
% sequence of commands:
f=fspecial('average');
xf=filter2(f,x);
xu=double(x)-xf/1.5
imshow(xu/70)

p=imread('pelicans.tif');
u=fspecial('unsharp',0.5);
pu=filter2(u,p);
imshow(p),figure,imshow(pu/255)

%High boost filtering
f=[-1 -1 -1;-1 11 -1;-1 -1 -1]/9;
xf=filter2(x,f);
imshow(xf/80)
id=[0 0 0;0 1 0;0 0 0];
f=fspecial('average');
hb1=3*id-2*f

hb2=1.25*id-0.25*f

x1=filter2(hb1,x);
imshow(x1/255)
% If each of the filters hb1 and hb2 are applied to an image with filter2, the result will have enhanced
% edges.

Im=[201 195 203 203 199 200 204 190 198 203;201 204 209 197 210 202 205 195 202 199;
    205 198 46 60 53 37 50 51 194 205;208 203 54 50 51 50 55 48 193 194;
    200 193 50 56 42 53 55 49 196 211;200 198 203 49 51 60 51 205 207 198;
    205 196 202 53 52 34 46 202 199 193;199 202 194 47 51 55 48 191 190 197;194 206 198 212 195 196 204 204 199 200;
    201 189 203 200 191 196 207 203 193 204];
figure,imshow(Im/255)
px=[-1 0 1;-1 0 1;-1 0 1]
icx=filter2(px,Im);
figure,imshow(icx/255)
py=px';
icy=filter2(py,Im);
figure,imshow(icy/255)

edge_t=im2bw(Im/255,0.3)
figure,imshow(edge_t)


% generate horizontal edge emphasis kernel
h = fspecial('sobel');
% invert kernel to detect vertical edges
h = h';
J = imfilter(c,h);
%edge_t=im2bw(c/255,0.3)
figure,imshow(J)

J = imfilter(I,h);
%% Edge detection on noisy images

c=imread('cameraman.tif');
c1=imnoise(c,'salt & pepper',0.1);
c2=imnoise(c,'gaussian',0,0.02);

% generate horizontal edge emphasis kernel
h = fspecial('sobel');
% invert kernel to detect vertical edges
h = h';
J = imfilter(c1,h);
%edge_t=im2bw(c/255,0.3)
figure,imshow(J)

% generate horizontal edge emphasis kernel
h = fspecial('sobel');
% invert kernel to detect vertical edges
%h = h';
J = imfilter(c1,h);
%edge_t=im2bw(c/255,0.3)
figure,imshow(J)

%We can obtain edges by the Prewitt filters directly by using the command
edge_p=edge(c1,'prewitt');
figure,imshow(edge_p)
% roberts edge fileter
edge_r=edge(c1,'roberts');
figure,imshow(edge_r)
%figure,imshow(edge_r)
% sobel edge detector
edge_s=edge(c1,'sobel');
figure,imshow(edge_s)

% Second differences
% We see that the Laplacian (after taking an absolute value, or squaring) gives double edges. It is also
%extremely sensitive to noise. However, the Laplacian does have the advantage of detecting edges in
%all directions equally well. To see an example, suppose we enter the Matlab commands:

l=fspecial('laplacian',0);
ic_l=filter2(l,c1);
figure,imshow(mat2gray(ic_l))
% In Matlab, Laplacians of all sorts can be generated using the fspecial function, in the form
fspecial('laplacian',ALPHA)
% If the parameter ALPHA (which is optional) is omitted, it is assumed to be 0.2
% The value 0 gives the Laplacian developed earlier.

% We now have a further method of edge detection:
% take the zero-crossings after a laplace filtering. This is implemented in Matlab with the zerocross
% option of edge, which takes the zero crossings after filtering with a given filter:
l=fspecial('laplacian',0);
icz=edge(c1,'zerocross',l);
figure, imshow(icz)

% This method was designed to provide a edge detection method to be as close as possible to biological
% vision. The first two steps can be combined into one, to produce a Laplacian of Gaussian or LoG
% filter. These filters can be created with the fspecial function. If no extra parameters are provided
% to the zerocross edge option, then the filter is chosen to be the LoG filter found by
fspecial('log',13,2)
%This means that the following command:
edf=edge(c1,'log')
figure, imshow(edf)
%produces exactly the same result as the commands:
log=fspecial('log',13,2);
edglog=edge(c1,'zerocross',log);
figure, imshow(edglog)

BW1 = edge(c1,'Canny');
%figure, imshow(BW1/255)
figure,imshow(mat2gray(BW1))

img=imread('MRI_clean.tif');
figure,imshow(mat2gray(img))
c1noise=imnoise(img,'salt & pepper',0.1);
figure,imshow(mat2gray(c1noise))
c2noise=imnoise(img,'gaussian',0,0.02);
figure,imshow(mat2gray(c2noise))
c3noise=imnoise(img,'speckle', 0.1);
figure,imshow(mat2gray(c3noise))

cannedge = edge(c1noise,'Canny');
figure,imshow(mat2gray(cannedge))

edge_p=edge(c1noise,'prewitt');
figure,imshow(edge_p)
% roberts edge fileter
edge_r=edge(c1noise,'roberts');
figure,imshow(edge_r)
%figure,imshow(edge_r)
% sobel edge detector
edge_s=edge(c1noise,'sobel');
figure,imshow(edge_s)
% median filtering
t_sp2_m5=medfilt2(c1noise,[5,5])
figure,imshow(c1noise)
figure,imshow(t_sp2_m5)
% edge detection
edge_s=edge(t_sp2_m5,'sobel');
figure,imshow(t_sp2_m5)
figure,imshow(edge_s)
% average filtering
a3=fspecial('average',[5,5]);
%t_sp_a3=filter2(c1noise,a3);
imff=imfilter(c1noise,a3);
figure,imshow(c1noise)
figure,imshow(imff)
%a5=fspecial('average',[5,5]);
% wiener filter
t1=wiener2(t_ga);
t2w=wiener2(t2,[7,7]);
% laplacian filter
f=fspecial('laplacian')
%t_sp_a3=filter2(c1noise,f);
imff=imfilter(c1noise,f);
figure,imshow(c1noise)
figure,imshow(imff)
figure,imshow(t_sp_a3)
f1=fspecial('average');
% 
f1=fspecial('log')
imff=imfilter(c1noise,f1);
figure,imshow(c1noise)
figure,imshow(imff)







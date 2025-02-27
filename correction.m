clc; clear all; close all;

%% input data

load('./data/msk.mat');

load('./data/phs.mat');
phs = phs .* single(msk);

load('./data/cos.mat');
cos = cos .* single(msk);

%% Define Dipole kernel in Fourier Domain

N = size(phs);
spatial_res = [1 1 1];
[ky,kx,kz] = meshgrid(-N(1)/2:N(1)/2-1, -N(2)/2:N(2)/2-1, -N(3)/2:N(3)/2-1);
kx = (kx / max(abs(kx(:)))) / spatial_res(1);
ky = (ky / max(abs(ky(:)))) / spatial_res(2);
kz = (kz / max(abs(kz(:)))) / spatial_res(3);

% Compute magnitude of kernel and perform fftshift
k2 = kx.^2 + ky.^2 + kz.^2;
kernel = 1/3 - (kz.^2 ./ (k2 + eps)); % Z is the B0-direction
kernel = fftshift(kernel);

%% predicted data

load('./output/prediction.mat');
susc = susc .* single(msk);
pred = susc; % x0

%% correction

max_iter = 10;
tol = 0.75;
alpha = 1;
pHy = real(ifftn(fftn(phs).*kernel)).*single(msk);

for iter = 1:max_iter   

    % compute the error
    err  = (real(ifftn(fftn(susc).*kernel))-phs).*single(msk);
    err  = 0.5*(norm(err(:),2)^2);                    
    
    % perform the update
    pHpx = real(ifftn(fftn(susc).*kernel.*kernel)).*single(msk);
    susc = susc + alpha * (pHy - pHpx);
    susc = susc.*single(msk);
    
    % check the gradient norm
    grad_f= -1*(pHy - pHpx).*single(msk);
    grad_norm = norm(grad_f(:), 2);
    disp(['iter:', num2str(iter),'   err:', num2str(err,'%.4f'),'   grad norm:', num2str(grad_norm,'%.4f')])
    
    if grad_norm < tol
        break
    end
end


%% save

save('./output/correction.mat', 'susc');


%% display

figure;

im1 = imrotate(cos(:,:,55),  -90);
im2 = imrotate(pred(:,:,55), -90);
im3 = imrotate(susc(:,:,55), -90);

subplot(131);imagesc(im1, [-0.1 0.1]);colormap('gray'); colorbar; xlabel('COSMOS');
subplot(132);imagesc(im2, [-0.1 0.1]);colormap('gray'); colorbar; xlabel('Prediction');
subplot(133);imagesc(im3, [-0.1 0.1]);colormap('gray'); colorbar; xlabel('Correction');


%% metrics

disp(['Metrics Before Correction : ' ,newline, 'SSIM',newline, 'PSNR',newline, 'NMSE',newline, 'HFEN']);

disp([compute_ssim(pred, cos);compute_psnr(pred, cos);compute_rmse(pred, cos);compute_hfen(pred, cos)]);

disp(['Metrics After Correction  : ',newline,  'SSIM',newline, 'PSNR',newline, 'NMSE',newline, 'HFEN']);

disp([compute_ssim(susc, cos);compute_psnr(susc, cos);compute_rmse(susc, cos);compute_hfen(susc, cos)]);

%%









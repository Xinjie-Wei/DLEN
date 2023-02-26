clc
% clear all;

HR_list='G:\Dataset\eval15\high\';                      % 	normal image path
modify_list = '..\result\LOLv1\';                       %  enhanced image path
B=[];
psnr_img = [];
imgModify  = dir([modify_list '*.png']);                % Iterate over all png format files
imgHR  = dir([HR_list '*.png']);                        % Iterate over all png format files
length(imgModify)

for i = 1:length(imgModify)                             % Iterate through all images

    imgM = imread([modify_list imgModify(i).name]);     % Read each image after reconstruction
    
    imgM = imgM(1:size(imgM,1),1:size(imgM,2),:);
    
%     Hr_name = replace(imgModify(i).name, 'low', 'normal');    % LOLV2
    Hr_name = imgModify(i).name;                                % LOLV1
    imgHr = imread([HR_list Hr_name]); 
    imgHr = imgHr(1:size(imgHr,1),5:size(imgHr,2)-4,:);
%     imgHr = imgHr(1:size(imgHr,1),1:size(imgHr,2),:);
    [peaksnr,~] = psnr(imgM,imgHr);
    psnr_img(i) = peaksnr;
    [ssimval] = ssim(imgM,imgHr,[0.01 0.03], fspecial('gaussian', 11, 1.5),255);
    B(i)=ssimval;

    
end

SSIM = mean(B)
PSNR = mean(psnr_img)

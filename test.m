
clc; clear; close all;
addpath(pwd);


imagePath = fullfile('1', 'images');        
saveDir   = fullfile('1', 'binary_out');    
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end


fprintf('========== 1. Load sequence ==========\n');

exts = {'*.bmp','*.png','*.jpg','*.jpeg','*.tif','*.tiff'};
imgFiles = [];

for i = 1:numel(exts)
    imgFiles = [imgFiles; dir(fullfile(imagePath, exts{i}))]; %#ok<AGROW>
end

if isempty(imgFiles)
    error('未在路径下找到图片: %s', imagePath);
end

[~, idx] = sort({imgFiles.name});
imgFiles = imgFiles(idx);

img1 = imread(fullfile(imagePath, imgFiles(1).name));
if size(img1,3) == 3
    img1 = rgb2gray(img1);
end

[H, W] = size(img1);
F = numel(imgFiles);

Img_Seq = zeros(H, W, F, 'double');

for k = 1:F
    img = imread(fullfile(imagePath, imgFiles(k).name));
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    Img_Seq(:,:,k) = im2double(img);
end

fprintf('Loaded %d frames.\n', F);


fprintf('========== 2. Set ELH-STD parameters ==========\n');

opts = struct();

opts.alphaTop4 = 0.5;
opts.seRadius = 9;
opts.d = 3;
opts.cellRadius = 1;

opts.alphaSigma = 0.35;
opts.alphaTemp  = 0.60;

opts.maxIter = 30;
opts.kappa = 0.05;
opts.useMotionPosterior = false;
opts.globalPruneRatio = 0.025;


opts.hgTauScale    = 1.10;
opts.hgBeta        = 10.0;
opts.hgProxyMix    = 0.10;
opts.hgSmoothSigma = 0.6;
opts.hgEps         = 1e-6;


fprintf('========== 3. Run ELH-STD ==========\n');

tic;
[Target_Tensor, Background_Tensor, out] = ELH_STD_Solver(Img_Seq, opts); 
elapsed_time = toc;

fprintf('ELH-STD finished. Time = %.2f s\n', elapsed_time);

fprintf('========== 4. Generate binary maps for all frames ==========\n');

k_thresh = 3.0;      
border   = 10;       
min_area = 2;        

for k = 1:F

    SaliencyMap = Target_Tensor(:,:,k);

   
    SaliencyMap(1:border, :) = 0;
    SaliencyMap(end-border+1:end, :) = 0;
    SaliencyMap(:, 1:border) = 0;
    SaliencyMap(:, end-border+1:end) = 0;

    
    SaliencyMap = mat2gray(SaliencyMap);

    
    mu = mean(SaliencyMap(:));
    sigma = std(SaliencyMap(:));
    BinaryMap = SaliencyMap > (mu + k_thresh * sigma);

    
    BinaryMap = bwareaopen(BinaryMap, min_area);

    
    [~, name, ~] = fileparts(imgFiles(k).name);
    outName = sprintf('%s_ELH_STD_binary.png', name);
    outPath = fullfile(saveDir, outName);

    imwrite(BinaryMap, outPath);

    fprintf('Saved %d / %d: %s\n', k, F, outName);
end

fprintf('========== Done! All binary maps saved in: %s ==========\n', saveDir);
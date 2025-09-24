clear; clc; close all;

% Convert .mat calibration to binary

[fname, dir0] = uigetfile('*.mat','Find Calibration File');

newFileName = fname(1:end-4);
load([dir0 fname],'wavelength');

%%
% load('wavelength-OH.01.2019.0006-created-01-May-2019.mat');
fid=fopen([dir0 newFileName],'w+');
fwrite(fid,wavelength,'double');
fclose(fid);

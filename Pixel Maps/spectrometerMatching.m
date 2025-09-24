clear; clc; close all;

% Convert .mat calibration to binary

[fname, dir0] = uigetfile('*.RAW','Find Calibration File');

res_axis = 2048;

fname1 = [fname(1:end-5) '0.RAW']; 
fname2 = [fname(1:end-5) '1.RAW']; 

m = memmapfile([dir0 fname1],'Format','uint16');
raw1 = single(m.data);
cnt = length(raw1);
clear m;

totalNumberOfAlines = cnt/res_axis;
raw1 = reshape(raw1,[res_axis, totalNumberOfAlines]);

m = memmapfile([dir0 fname2],'Format','uint16');
raw2 = single(m.data);
cnt = length(raw2);
clear m;

raw2 = reshape(raw2,[res_axis, totalNumberOfAlines]);

%% Crop raw data

numAlines = 5000;

all_sA = raw1';
all_sA = all_sA(100:100+numAlines-1,:);
all_sB = raw2';
all_sB = all_sB(100:100+numAlines-1,:);

corr_matrix = zeros(1,1);
for i = 1:size(all_sA,2)
    i
    for j = 1:size(all_sB,2)
        corr_val = corrcoef(all_sA(:,i), all_sB(:,j));
        corr_matrix(i,j) = corr_val(1,2);
    end
end

hold_max = zeros(1,1);
for k = 1:size(all_sA,2)
    [m,ind] = max(corr_matrix(:,k));
    hold_max(k) = ind;
end

figure; plot(hold_max,'linewidth', 3); hold on; 
title('Calibration Map: Spectrometer A to B')
xlabel('bA')
ylabel('bB');

%% Fit Pixel Map

f = polyfit(70:2000,hold_max(70:2000),3);
remappedPix = polyval(f,1:2048);

remappedPix(remappedPix<0) = 0;
remappedPix(remappedPix>2048) = 0;

fname = ['pixMap_' fname(1:12)];
fid=fopen([dir0 fname],'w+');
fwrite(fid,remappedPix,'double');
fclose(fid);

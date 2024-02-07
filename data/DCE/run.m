clear all; close all; clc;

load('Breast_Data_partial.mat');

for n = 1 : 21
    
    [simImg,smap,parMap,temp,ID,aif,T10,aifci_1s,cts] = gen_simIMG2(new_data, n);
    
    if ~exist(ID, 'dir')
        mkdir(ID);
    else
        disp([ID, ' already exists.']);
    end
    
    
    save(strcat(ID, '/cart_images.mat'), 'simImg');
    save(strcat(ID, '/coil_sens.mat'), 'smap');
end
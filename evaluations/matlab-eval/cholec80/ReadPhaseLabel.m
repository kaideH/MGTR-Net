function [ outp ] = ReadPhaseLabel( file )

% Read the phase label (annotation and prediction)
fid = fopen(file, 'r');

% read the header first
tline = fgets(fid); 

% read the labels
[outp] = textscan(fid, '%d %s\n');

end


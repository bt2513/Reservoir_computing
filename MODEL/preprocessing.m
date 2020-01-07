% Script used to preprocess the audios

addpath('../AuditoryToolbox/')
% List the names of the wav files in the directory
L = dir('*.WAV');
files = {L.name};

% Draw the cochlea of one .wav file
file = files(3);    % The number can be changed here to preprocess another file
file = file{1}
[data, fs] = audioread(file);
coch = LyonPassiveEar(data, fs, 170);
coch = coch(:);
%imagesc(coch/max(max(coch)));

% --> Save the 'coch' variable in the Workspace to use it on Python




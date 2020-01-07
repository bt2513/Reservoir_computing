neurones = 400;
sparsity = 0.98;
mask = {0.49, 0.51};

addpath('../AuditoryToolbox/')
% List the names of the wav files in the directory
L = dir('*.WAV');
files = {L.name};

% Draw the cochlea of one wav file
file = files(1);
file = file{1};
[data, fs] = audioread(file);
coch = LyonPassiveEar(data, fs, 120);
imagesc(coch/max(max(coch)));pause;

% Construction of the connectivity matrix
[r, c] = size(coch);
con_matrix = zeros(neurones, r);
for k=1:neurones
	for j=1:r
        if rand() < (1-sparsity)
            if rand() < 0.5
                con_matrix(k, j) = mask{1};
            else
                con_matrix(k, j) = mask{2};
            end
        end
	end
end

% Draw the (connectivity matrix * cochlea) output
M = con_matrix*coch;
imagesc(M/max(max(M)));



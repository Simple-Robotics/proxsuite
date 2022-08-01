% Command if ismac () decodecmd = 'cutest2matlab_osx';
else decodecmd = 'cutest2matlab';
end

  % Extract files files = dir('sif_data/*.SIF');
for
  file = files' [~, probname, ~] = fileparts(file.name); sif2mat(probname,
                                                                 decodecmd);
end

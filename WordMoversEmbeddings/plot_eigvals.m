clear all;
close all;
clc;

dataset = "kong_tps";
% format the original matrix
% original_matrix = readNPY(strcat("../../GYPSUM/",dataset,"_predicts_0.npy"));
original_matrix = readNPY("../../GYPSUM/kong_tps_similarity.npy");
% original_matrix = load(strcat("./mat_files/",dataset,"_K_set1.mat")).trainData;
% original_matrix = load(strcat("./mat_files/","twitter_K_set1")).trainData;
% original_matrix = original_matrix(:,2:end);

% [max_vals, max_indices] = max(original_matrix, [], 2);
% max_indices(max_indices==1) = 0;
% max_indices(max_indices==2) = 1; 

max_indices = original_matrix;

% id_count = sqrt(length(max_indices));
id_count = length(max_indices);
reshaped_matrix = reshape(max_indices, id_count, id_count);
% reshaped_matrix = 1 - reshaped_matrix;

% non_ero_prcntg = nnz(reshaped_matrix) / (size(reshaped_matrix,1)*size(reshaped_matrix,2))
% non_zero_vals = nonzeros(reshaped_matrix);
% scatter(non_zero_vals)

% symmetrize the matrix
% reshaped_matrix = (reshaped_matrix+reshaped_matrix')/2 ;

% get the eigenvalues
eigenvals = eig(reshaped_matrix);

% scatter([1:length(eigenvals)], eigenvals)

ipsd = all(eigenvals > 0);
k = rank(reshaped_matrix);
% 
absolute_eigenvalues = abs(eigenvals);
[M,I] = sort(absolute_eigenvalues);
eigenvals = eigenvals(I);
% 
% 
reversed_eigenvals = eigenvals(end:-1:1);
curtailed_eigenvals = eigenvals(end-1:-1:(end-1)-199);
% 
% 
% % activate for unsymmetrized version
real_eigenvals = real(curtailed_eigenvals);
% abs_eigenvalues = abs(curtailed_eigenvals);
real_reversed_eigenvals = real(reversed_eigenvals);
% abs_reversed_eigenvals = abs(reversed_eigenvals);
% 
if strcmp(dataset, "rte")
    plot_str = "RTE";
    method = "BERT similarities";
end
if strcmp(dataset, "stsb")
    plot_str = "STS-B";
    method = "BERT similarities";
end
if strcmp(dataset, "mrpc")
    plot_str = "MRPC";
    method = "BERT similarities";
end
if strcmp(dataset, "twitter")
    plot_str = "TWITTER";
    method = "WMD similarities";
end
if strcmp(dataset, "oshumed")
    plot_str = "OHSUMED";
    method = "WMD similarities";
end
if strcmp(dataset, "20News")
    plot_str = "20NEWS";
    method = "WMD similarities";
end
if strcmp(dataset, "recipeL")
    plot_str = "RECIPE-L";
    method = "WMD similarities";
end

save(strcat("eigavls_info_",dataset,".mat"), "real_eigenvals", "ipsd", "k")

% figure('units','normalized','outerposition',[0 0 1 1]);
% figure;
% scatter([1:200], real_eigenvals, 'filled');
% set(gca,'fontsize',18)
% xlabel("Eigenvalue indices", 'fontsize',20)
% ylabel("Eigenvalues", 'fontsize',20)
% title(strcat("Eigenvalues 2 to 201 of ",plot_str," predictions for ",method),'fontsize',20)
% str = {strcat('Is PSD? ', string(ipsd)), strcat('Rank: ', string(k))};
% text([.7 .7],[.7 .6],str, 'Units','normalized','fontsize',20)
% h=gcf;
% set(h,'PaperOrientation','landscape');
% set(h,'PaperPosition', [0.1 1 10 7]);
% print(gcf, '-dpdf', strcat('..\figures\sym_',dataset,'_eigenvalues_1_200.pdf'));
% 
% figure('units','normalized','outerposition',[0 0 1 1]);
% figure;
% scatter([1:length(reversed_eigenvals)], real_reversed_eigenvals, 'filled');
% set(gca,'fontsize',18)
% xlabel("Eigenvalue indices", 'fontsize',20)
% ylabel("Eigenvalues", 'fontsize',20)
% title(strcat("Eigenvalues of ",plot_str," predictions for ",method), 'fontsize',20)
% str = {strcat('Is PSD? ', string(ipsd)), strcat('Rank: ', string(k))};
% text([.7 .7],[.7 .65],str, 'Units','normalized','fontsize',20)
% h=gcf;
% set(h,'PaperOrientation','landscape');
% set(h,'PaperPosition', [0.1 1 10 7]);
% print(gcf, '-dpdf', strcat('..\figures\sym_',dataset,'_eigenvalues_all.pdf'));

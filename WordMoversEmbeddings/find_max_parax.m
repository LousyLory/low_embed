function [max_values_for_parax, best_cs, best_gammas, best_lambda_inverses] = find_max_parax(parax, lambda_inverse, gamma_list, c_for_eig, sample_size_list)
    % finds best validation score per sample_size
    length_lambda_inverse_list = length(lambda_inverse);
    length_gamma_list = length(gamma_list);
    length_c_for_eig = length(c_for_eig);
    length_sample_size_list = length(sample_size_list);
    step = length_lambda_inverse_list * length_gamma_list * length_c_for_eig;
    
    best_cs = zeros(1, length_sample_size_list);
    best_gammas = zeros(1, length_sample_size_list);
    best_lambda_inverses = zeros(1, length_sample_size_list);
    max_values_for_parax = zeros(1, length_sample_size_list);
    sz = [length_lambda_inverse_list, length_gamma_list, length_c_for_eig];
    
    for i=1:length_sample_size_list
        [max_val, ind] = max(parax(1,(i-1)*step+1:(i-1)*step+step-1));
        max_values_for_parax(1, i) = max_val;
        [j,k,l] = ind2sub(sz, [ind]);
        best_cs(1,i) = c_for_eig(1, l);
        best_gammas(1, i) = gamma_list(1, k);
        best_lambda_inverses(1, i) = lambda_inverse(1, j);
    end
end

%%% sample usage
% [max_values_for_parax, best_cs, best_gammas, best_lambda_inverses] = find_max_parax(parax, lambda_inverse, gamma_list, c_for_eig, sample_size_list);
% plot(sample_size_list, max_values_for_parax)
% xlabel(sample size)
% ylabel(validation accuracy)
% title(best validation accuracy per sample size)
function [feats] = optimal_rank(matrix, sample_size)
    [U,S,V] = svd(matrix);
        U = U(:, 1:sample_size);
        S = S(1:sample_size, 1:sample_size);
        V = V(:, 1:sample_size);
        feats = U*S^(-1/2);
end
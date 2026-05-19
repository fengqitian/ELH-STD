function [Target_Tensor, Background_Tensor, out] = ELH_STD_Solver(Img_Seq, opts)


    if nargin < 2, opts = struct(); end
    p = get_opts(opts);

    [H, W, F] = size(Img_Seq);
    D = double(Img_Seq);

    fprintf('========== ELH-STD Solver ==========');
    fprintf('\nH=%d, W=%d, F=%d\n', H, W, F);

   
    fprintf('-> Stage 1: build Top-4 + explicit local hypergraph spatiotemporal prior ...\n');

    I_top_seq = zeros(H, W, F);
    for k = 1:F
        I_top_seq(:,:,k) = imtophat(D(:,:,k), strel('disk', p.seRadius));
    end

    W_prior = ones(H, W, F);
    S_top4_seq = zeros(H, W, F);
    S_hg_seq   = zeros(H, W, F);
    T_map_seq  = zeros(H, W, F);
    Joint_seq  = zeros(H, W, F);

    for k = 1:F
        idx_prev = max(1, k-1);
        idx_next = min(F, k+1);

        I_top = I_top_seq(:,:,k);
        V_seq = cat(3, I_top_seq(:,:,idx_prev), I_top, I_top_seq(:,:,idx_next));
        T_raw = max(V_seq, [], 3) - min(V_seq, [], 3);
        T_map = normalize01(T_raw);

       
        S_top4 = calc_top4_response(I_top, p.d);
        S_top4 = normalize01(S_top4);

       
        S_hg = calc_hg_spatial_response(I_top, T_raw, p);
        S_hg = normalize01(S_hg);

        S_fuse = p.alphaTop4 * S_top4 + (1 - p.alphaTop4) * S_hg;
        S_fuse = normalize01(S_fuse);

        Joint_P = sqrt(max(S_fuse,0) .* max(T_map,0));
        Joint_P = imfilter(Joint_P, fspecial('gaussian', [3 3], 1.0), 'replicate');
        Joint_P = normalize01(Joint_P);

        W_prior(:,:,k) = 1 ./ (0.99 * Joint_P + 0.01);
        S_top4_seq(:,:,k) = S_top4;
        S_hg_seq(:,:,k) = S_hg;
        T_map_seq(:,:,k) = T_map;
        Joint_seq(:,:,k) = Joint_P;
    end


    fprintf('-> Stage 2: run TRPCA backend ...\n');

    T = I_top_seq;
    B = D - T;
    Y = zeros(H, W, F);

    lambda = p.lambdaFactor / sqrt(max(H, W) * F);
    mu = p.muFactor / max(norm(D(:), 2), eps);
    rho = p.rho;
    max_mu = p.maxMu;
    tol = p.tol;

    err_hist = zeros(p.maxIter, 1);
    for iter = 1:p.maxIter
       
        Temp_T = D - B + (1 / mu) * Y;
        W_dynamic = W_prior .* (p.kappa ./ (abs(T) + p.kappa));
        thresh_T = (lambda / mu) .* W_dynamic;
        T = sign(Temp_T) .* max(abs(Temp_T) - thresh_T, 0);

       
        Temp_B = D - T + (1 / mu) * Y;
        [B, ~] = prox_tnn(Temp_B, 1 / mu);

        
        leq = D - B - T;
        Y = Y + mu * leq;
        mu = min(max_mu, mu * rho);

        err = norm(leq(:), 2) / max(norm(D(:), 2), eps);
        err_hist(iter) = err;
        fprintf('iter %02d | err = %.3e | max(T)=%.6f | mean(T)=%.6f\n', ...
            iter, err, max(T(:)), mean(T(:)));
        if err < tol
            err_hist = err_hist(1:iter);
            break;
        end
    end
    if err_hist(end) == 0
        nz = find(err_hist > 0, 1, 'last');
        if ~isempty(nz), err_hist = err_hist(1:nz); end
    end


    Target_Tensor = max(T, 0);

    if p.useMotionPosterior
        Tmax = max(Target_Tensor, [], 3);
        Tmin = min(Target_Tensor, [], 3);
        MT = (Tmax - Tmin) > p.motionEps;
        Target_Tensor = Target_Tensor .* repmat(MT, [1 1 F]);
    end

    if p.globalPruneRatio > 0
        gmax = max(Target_Tensor(:));
        if gmax > 0
            Target_Tensor = Target_Tensor .* (Target_Tensor > p.globalPruneRatio * gmax);
        end
    end

    for k = 1:F
        Target_Tensor(:,:,k) = normalize01(Target_Tensor(:,:,k));
    end

    Background_Tensor = B;

    out = struct();
    out.I_top_seq = I_top_seq;
    out.W_prior = W_prior;
    out.S_top4_seq = S_top4_seq;
    out.S_hg_seq = S_hg_seq;
    out.T_map_seq = T_map_seq;
    out.Joint_seq = Joint_seq;
    out.err_hist = err_hist;
    out.lambda = lambda;
end


function p = get_opts(opts)
    p = struct();

    p.seRadius = 9;       
    p.d = 3;              
    p.cellRadius = 1;     

    p.alphaSigma = 0.35;  
    p.alphaTemp  = 0.60;  
    p.alphaTop4  = 0.50;  

   
    p.lambdaFactor = 1.2;
    p.muFactor = 1.25;
    p.rho = 1.5;
    p.maxMu = 1e10;
    p.maxIter = 30;
    p.tol = 1e-6;
    p.kappa = 0.05;
    p.useMotionPosterior = false;
    p.motionEps = 0.05;
    p.globalPruneRatio = 0.025;  
    
    p.hgTauScale    = 1.10;   
    p.hgBeta        = 10.0;   
    p.hgProxyMix    = 0.10;   
    p.hgSmoothSigma = 0.60;   
    p.hgEps         = 1e-6;

    if ~isempty(opts)
        fn = fieldnames(opts);
        for i = 1:numel(fn)
            p.(fn{i}) = opts.(fn{i});
        end
    end
end

function S_top4 = calc_top4_response(I_top, d)
    I_pad = padarray(I_top, [d, d], 'replicate');
    V0 = I_top;
    V1 = I_pad(1:end-2*d, d+1:end-d);      V2 = I_pad(2*d+1:end, d+1:end-d);
    V3 = I_pad(d+1:end-d, 1:end-2*d);      V4 = I_pad(d+1:end-d, 2*d+1:end);
    V5 = I_pad(1:end-2*d, 1:end-2*d);      V6 = I_pad(2*d+1:end, 2*d+1:end);
    V7 = I_pad(1:end-2*d, 2*d+1:end);      V8 = I_pad(2*d+1:end, 1:end-2*d);

    ds1 = max(0, V0 - V1); ds2 = max(0, V0 - V2);
    ds3 = max(0, V0 - V3); ds4 = max(0, V0 - V4);
    ds5 = max(0, V0 - V5); ds6 = max(0, V0 - V6);
    ds7 = max(0, V0 - V7); ds8 = max(0, V0 - V8);

    DS_cat = cat(3, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8);
    DS_sort = sort(DS_cat, 3, 'descend');
    S_top4 = mean(DS_sort(:,:,1:4), 3);
end

function S_hg = calc_hg_spatial_response(I_top, T_raw, p)


    r = p.cellRadius;
    win = fspecial('average', [2*r+1, 2*r+1]);

  
    mu  = imfilter(I_top, win, 'replicate');
    mu2 = imfilter(I_top.^2, win, 'replicate');
    sig = sqrt(max(mu2 - mu.^2, 0));
    tmp = imfilter(T_raw, win, 'replicate');

    [mu0, muN] = shifted_9grid(mu,  p.d);
    [sg0, sgN] = shifted_9grid(sig, p.d);
    [tm0, tmN] = shifted_9grid(tmp, p.d);

    
    D8 = cell(1, 8);
    for i = 1:8
        D8{i} = feat_dist(mu0, sg0, tm0, muN{i}, sgN{i}, tmN{i}, p);
    end
    Dcat  = cat(3, D8{:});
    dmin  = min(Dcat, [], 3);
    dmax  = max(Dcat, [], 3);
    dmean = mean(Dcat, 3);

    
    tau = p.hgTauScale * 0.5 * (dmin + dmax);

    
    IsoSoft = cell(1, 8);
    for i = 1:8
        IsoSoft{i} = 1 ./ (1 + exp(-p.hgBeta * (D8{i} - tau)));
    end

    

    E = { ...
        [1 3 5], ...
        [1 4 7], ... 
        [2 3 8], ... 
        [2 4 6], ... 
        [1 2],   ...
        [3 4],   ... 
        [5 6],   ... 
        [7 8]    ... 
    };

    edge_score_sum = zeros(size(I_top));
    edge_weight_sum = zeros(size(I_top));

    for e = 1:numel(E)
        idx = E{e};

        
        center_sep = zeros(size(I_top));
        iso_avg    = zeros(size(I_top));
        for t = 1:numel(idx)
            center_sep = center_sep + IsoSoft{idx(t)} .* D8{idx(t)};
            iso_avg    = iso_avg    + IsoSoft{idx(t)};
        end
        center_sep = center_sep / numel(idx);
        iso_avg    = iso_avg / numel(idx);

        
        bg_cohesion = hyperedge_neighbor_cohesion(muN, sgN, tmN, idx, p);

        
        w_e = exp(-bg_cohesion ./ (dmean + p.hgEps));

       
        edge_score = w_e .* (center_sep ./ (bg_cohesion + p.hgEps)) .* iso_avg;

        edge_score_sum  = edge_score_sum + edge_score;
        edge_weight_sum = edge_weight_sum + w_e;
    end

    S_edge = edge_score_sum ./ (edge_weight_sum + p.hgEps);

    
    iso_ratio = zeros(size(I_top));
    for i = 1:8
        iso_ratio = iso_ratio + IsoSoft{i};
    end
    iso_ratio = iso_ratio / 8;

    
    H_nb = explicit_neighbor_homogeneity(muN, sgN, tmN, p);
    H_nb = 1 ./ (1 + H_nb);

    
    dstd = std(Dcat, 0, 3);
    uniformity = 1 - dstd ./ (dmean + p.hgEps);
    uniformity = max(min(uniformity, 1), 0);
    S_proxy = min(Dcat, [], 3) .* uniformity;

    
    S_explicit = normalize01(S_edge) .* sqrt(normalize01(iso_ratio) .* normalize01(H_nb));
    S_hg = (1 - p.hgProxyMix) * S_explicit + p.hgProxyMix * normalize01(S_proxy);

    S_hg = imfilter(S_hg, fspecial('gaussian', [3 3], p.hgSmoothSigma), 'replicate');
    S_hg = normalize01(S_hg);
end

function d = feat_dist(muA, sgA, tmA, muB, sgB, tmB, p)
    d = sqrt( ...
        (muA - muB).^2 + ...
        p.alphaSigma * (sgA - sgB).^2 + ...
        p.alphaTemp  * (tmA - tmB).^2 );
end

function c = hyperedge_neighbor_cohesion(muN, sgN, tmN, idx, p)
    pairs = nchoosek(idx, 2);
    if isempty(pairs)
        c = zeros(size(muN{idx(1)}));
        return;
    end

    c = zeros(size(muN{idx(1)}));
    for k = 1:size(pairs, 1)
        a = pairs(k,1);
        b = pairs(k,2);
        c = c + feat_dist(muN{a}, sgN{a}, tmN{a}, muN{b}, sgN{b}, tmN{b}, p);
    end
    c = c / size(pairs, 1);
end

function v = explicit_neighbor_homogeneity(muN, sgN, tmN, p)
    pairs = [1 3; 1 4; 2 3; 2 4; 5 6; 7 8; 5 7; 6 8];
    v = zeros(size(muN{1}));
    for ii = 1:size(pairs,1)
        a = pairs(ii,1);
        b = pairs(ii,2);
        v = v + feat_dist(muN{a}, sgN{a}, tmN{a}, muN{b}, sgN{b}, tmN{b}, p);
    end
    v = v / size(pairs,1);
end

function v = local_neighbor_variation(muN, sgN, tmN, p)
    pairs = [1 3; 1 4; 2 3; 2 4; 5 7; 5 8; 6 7; 6 8];
    tmp = 0;
    for ii = 1:size(pairs,1)
        a = pairs(ii,1); b = pairs(ii,2);
        dab = sqrt((muN{a}-muN{b}).^2 + p.alphaSigma*(sgN{a}-sgN{b}).^2 + p.alphaTemp*(tmN{a}-tmN{b}).^2);
        tmp = tmp + dab;
    end
    v = tmp / size(pairs,1);
end

function [c0, neigh] = shifted_9grid(X, d)
    Xp = padarray(X, [d d], 'replicate');
    c0 = X;
    neigh = cell(1,8);
    neigh{1} = Xp(1:end-2*d, d+1:end-d);      
    neigh{2} = Xp(2*d+1:end, d+1:end-d);      
    neigh{3} = Xp(d+1:end-d, 1:end-2*d);     
    neigh{4} = Xp(d+1:end-d, 2*d+1:end);      
    neigh{5} = Xp(1:end-2*d, 1:end-2*d);      
    neigh{6} = Xp(2*d+1:end, 2*d+1:end);     
    neigh{7} = Xp(1:end-2*d, 2*d+1:end);      
    neigh{8} = Xp(2*d+1:end, 1:end-2*d);     
end

function Y = normalize01(X)
    X = double(X);
    xmin = min(X(:));
    xmax = max(X(:));
    if xmax - xmin < eps
        Y = zeros(size(X));
    else
        Y = (X - xmin) / (xmax - xmin + eps);
    end
end

function [X, tnn] = prox_tnn(Y, rho)

    [n1, n2, n3] = size(Y);
    Yf = fft(Y, [], 3);
    Xf = zeros(n1, n2, n3);
    tnn = 0;
    halfn3 = ceil((n3 + 1) / 2);
    for i = 1:halfn3
        [U, S, V] = svd(Yf(:,:,i), 'econ');
        s = diag(S);
        s_shrink = max(s - rho, 0);
        rank_i = sum(s_shrink > 0);
        if rank_i >= 1
            Xf(:,:,i) = U(:,1:rank_i) * diag(s_shrink(1:rank_i)) * V(:,1:rank_i)';
            tnn = tnn + sum(s_shrink);
        end
    end
    for i = halfn3+1:n3
        Xf(:,:,i) = conj(Xf(:,:,n3 - i + 2));
    end
    X = ifft(Xf, [], 3, 'symmetric');
    tnn = tnn / n3;
end
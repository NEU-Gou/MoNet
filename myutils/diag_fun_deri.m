function  out_S = diag_fun_deri(S, epsilon, type, par)

        switch type
            case {'power'}
                
                out_S = par .* ((S + epsilon) .^ (par - 1));
                
            case {'log'}
                
                out_S = 1 ./ (S + epsilon);
                
            otherwise
                    error('derivative of diagonal matrix function not supported!');
        end
end

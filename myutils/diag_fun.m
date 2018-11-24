function  out_S = diag_fun(S, epsilon, type, par)

        switch type
            case {'power'}
                
                out_S =  (S  + epsilon) .^ par ;
                
            case {'log'}
                
                out_S = log(S+epsilon);
                
            otherwise
                    error('diagonal matrix function not supported!');
        end
end
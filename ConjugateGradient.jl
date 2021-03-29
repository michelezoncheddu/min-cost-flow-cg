module ConjugateGradient

    function conjugate_gradient(A::Matrix, b::Vector)
        n = size(A, 1)
        x = zeros(n)
        d = b
        r_old = b
        
        for i in 1:n
            alpha = (r_old'r_old) / (d'A*d)
            x = x + alpha * d
            r = r_old - alpha * A * d
            beta = (r'r) / (r_old'r_old)
            d = r + beta * d

            r_old = r
        end

        return x
    end

    export conjugate_gradient
end

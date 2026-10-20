# U3: dist_dsl.jl:29's covariance 2 .* ones(2, 2) is singular. What does Gen.jl compute?
using Gen, LinearAlgebra, Printf
import Distributions
C = 2 .* ones(2, 2)
F = cholesky(Symmetric(C); check=false)
@printf("cholesky issuccess: %s, factor diagonal: %s\n", issuccess(F), string(diag(F.U)))
@printf("logpdf(mvnormal, zeros(2), ones(2), 2 .* ones(2, 2)) = %.6f\n", logpdf(mvnormal, zeros(2), ones(2), C))
@dist transformed_mvnormal(mu, sigma) = mvnormal(broadcast(+, mu, 1.0), broadcast(*, sigma, 2.0))
@printf("logpdf(transformed_mvnormal, zeros(2), zeros(2), ones(2, 2)) = %.6f\n", logpdf(transformed_mvnormal, zeros(2), zeros(2), ones(2, 2)))
@printf("logpdf at [1, 1] (on the line the matrix spans) = %.6f; at [1, 0] (off it) = %.6g\n",
        logpdf(mvnormal, [1.0, 1.0], ones(2), C), logpdf(mvnormal, [2.0, 0.0], ones(2), C))
println("logpdf_grad x: ", logpdf_grad(mvnormal, zeros(2), ones(2), C)[1])

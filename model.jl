begin
    using Parameters
    using Printf
    using Distributions 
    using Roots
    using LinearAlgebra
    using Plots
end


function tauchen(N, ρ, σ; μ = 0.0, m = 3.0)
	s1    = μ/(1-ρ) - m*sqrt(σ^2/(1-ρ^2))
   	sN    = μ/(1-ρ) + m*sqrt(σ^2/(1-ρ^2))
    s = collect(range(s1, sN, length = N))
    step    = (s[N]-s[1])/(N-1)  #evenly spaced grid
    P      = fill(0.0, N, N)

    for i = 1:ceil(Int, N/2)
    	P[i, 1] = cdf.(Normal(), (s[1] - μ - ρ*s[i] + step/2.0)/σ)
        P[i, N]  = 1 - cdf.(Normal(), (s[N] - μ - ρ*s[i]  - step/2.0)/σ)
        for j = 2:N-1
        	P[i,j]  = cdf.(Normal(), (s[j] - μ - ρ*s[i]  + step/2.0)/σ) -
                            cdf.(Normal(), (s[j] - μ - ρ*s[i] - step/2.0)/σ)
        end
        P[floor(Int, (N-1)/2+2):end, :]=P[ceil(Int ,(N-1)/2):-1:1, end:-1:1]
	end

    ps = sum(P, dims = 2)
    P = P./ps

    return s, P
end

function set_par(;
    β = 0.8,
    ρ = 0.9,
    σ = 0.2,
    ϕ̄ = 1.0,
    α = 2/3,
    c_e = 40.0,
    c_f = 20.0,
    D̄ = 100.0,
    μᵍ = 0.0,
    σᵍ = 0.2,
    nΦ = 101,
    w = 1.0)
    
    ϕ̄ = ϕ̄*(1-ρ)
    gΦ, Fᵀ = tauchen(nΦ,ρ,σ;μ = ϕ̄,m = 4.0)
    gΦ = @. exp(gΦ)
    invΦ = Fᵀ^1000
    invΦ = invΦ[1,:]

    G_prob = invΦ

    return(β = β, Fᵀ = Fᵀ, gΦ = gΦ, α = α, c_e = c_e, 
    c_f = c_f, D̄ = D̄, nΦ = nΦ, w = w, G_prob = G_prob)

end

param = set_par()


function solver(p_guess, param)
    @unpack β, Fᵀ, gΦ, α, c_f, nΦ, w = param
    # println(β, Fᵀ, gΦ, α, c_f, nΦ, w)

    # static decision: 
    gN = @. (p_guess*α*gΦ/w)^(1/(1.0-α))
    gΠ = @. p_guess*gΦ*gN^α - w*gN - c_f*w 

    # solve problem
    tol = 1.0e-9
    max_iter = 500
    iter_count = 10
    print_it = false
    V = gΠ # initial guess

    function vfi(V)
        v_guess = fill(0.0,nΦ)
        for iter = 1:max_iter
            v_guess .= V # next period guess
            
            V = gΠ + β*max.(0,Fᵀ*v_guess)

            sup = maximum(abs.(v_guess - V)) # check convergence
            if sup<tol*maximum(abs.(V))
                if print_it; println("Converged in $iter iterations", "Tol. achieved in $sup "); end;
                    break; end
            if (iter == max_iter) & print_it; println("No convergence in $max_iter iterations", "Tol. achieved in $sup "); end;
            if (iter%iter_count==0) & print_it; println("Iteration $iter, sup = $sup"); end;
        end
        χ = fill(0.0, nΦ)
        χ[Fᵀ*V .< 0.0] .= 1.0
        return (V, χ)
    end
    V, χ = vfi(V)
    return V,χ,gN, gΠ
end


sol_bellman = solver(2.0, param)

function solve_price(param)
        
    function entry(p_guess)
        @unpack β, c_e, G_prob, w = param
        V, χ, gN, gΠ = solver(p_guess, param)
        excess_entry = β*sum(V.*G_prob)-c_e*w
        return excess_entry
    end
    
    p = find_zero(entry, 5.0)
    V, χ, gN, gΠ = solver(p, param)

    return (p = p, V = V, χ = χ, gN = gN, gΠ = gΠ)
end

test1 =solve_price(param)


function solve_m(param, solution)
    @unpack Fᵀ, nΦ, gΦ, α, G_prob, D̄ = param
    @unpack χ, gN, p = solution
    
    # construct transition probability:
    P̂ = ((1 .- χ).*Fᵀ)'
    
    # invariant distribution is just a homogeneous function of M
    inv_dist(M) = M*inv(I - P̂)*G_prob # this is a function
    
    # supply: integrate the total production
    y = @. gΦ*gN^α
    supply = sum(inv_dist(1).*y) # just use the function with an arbitrary M

    # demand
    demand = D̄/p
    
    # find mass of entrants (exploit linearity of the invariant distribution)
    M = demand/supply 
	mu = inv_dist(M)
	
    return M, mu
end



function ModelStats(param, sol_price, M, mu)
    @unpack Fᵀ, nΦ, gΦ, α, G_prob, D̄ = param
	@unpack gN, p, χ, gΠ = sol_price
	
	# productivity distribution
	pdf_dist = mu./sum(mu)
	cdf_dist = cumsum(pdf_dist)
	
	# employment distribution
	emp_dist = mu.*gN
	pdf_emp = emp_dist./sum(emp_dist)
	cdf_emp = cumsum(pdf_emp)
	
	# exit productivity
	cut_index = findfirst(χ .== 0)
	phicut=	param.gΦ[cut_index]

	# stats
	avg_firm_size = sum(emp_dist)/sum(mu)
	exit_rate = M/sum(mu)
	Y = sum((gΦ.*gN.^α).*mu) ## agg production
	emp_prod = sum(emp_dist) # employment used in production
	Π =  sum(gΠ.*mu)   # aggregate profits
	
	# employment share
	#size_array = [10, 20, 50, 100, 500]
	
	return (pdf_dist, cdf_dist, pdf_emp, cdf_emp, avg_firm_size, exit_rate, Y, emp_prod, phicut, Π)
	
end


function SolveModel(param)
	
	# Solve For Prices
	sol_price =solve_price(param)  
	M, mu = solve_m(param, sol_price)
	
	if M<=0;
		println("Warning: No entry, eq. not found.")
	end
	
	stats = ModelStats(param, sol_price, M, mu)
	return (sol_price, M, mu, stats)
end


begin
	solution_info = SolveModel(set_par(; α = 2/3))
	
	function printStats(solution_info)
		sol_price = solution_info[1]
		M = solution_info[2]
		mu = solution_info[3]
		stats = solution_info[4]
	
		p = sol_price.p
		avg_firm_size = stats[5]
		exit_rate = stats[6]
		Y = stats[7]
		cutoff = stats[9]
		Pi = stats[10]

        println("Price: $p")
        println("Avg. Firm Size: $avg_firm_size")
        println("Exit/entry Rate: $exit_rate")
        println("Productivity Cutoff: $cutoff")	
        println("Aggregate Output: $Y")	
        println("Aggregate Profits: $Pi")	

	end
	
	printStats(solution_info)
	
end

begin
	@unpack gN = solution_info[1]
	
	plot(gN, [solution_info[4][1], solution_info[4][3]], label=["Firm Share" "Emp. Share" ], lw = 3)
	title!("Stationary Distribution")
	xlims!(0,2000) # little mass over 50
	#ylims!(0,0.05) # Ps the mass of constrained HH is about 20% of individuals in S)1
	xlabel!("Size")
	ylabel!("density")
	
	
end


begin
	solution_info2 = SolveModel(set_par(; c_e = 60))
	printStats(solution_info2)
	#solution_info2[4]
end

begin
	solution_info3 = SolveModel(set_par(; c_f = 30))
	printStats(solution_info3)
end
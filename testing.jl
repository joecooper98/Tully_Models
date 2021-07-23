#!/usr/bin/julia

using LinearAlgebra
using DifferentialEquations
using Random

function Tully_I(x::Float64,A::Float64=0.01,B::Float64=1.6,C::Float64=0.005,D::Float64=1.0)
        #
        # - input
        #   - x - value of the function to be evaluated
        #   - A, B, C, D - parameters for the model, Tully's original parameters are the default
        # 
        # - output
        #   - mat - 2x2 matrix of the diabatic potential, as defined in Tully's paper
        #   - dmat - 2x2 matrix of the derivatives of the diabatic potential matrix elements

        #init matrices of zeros
        mat = zeros(Float64,(2,2))
        #dmat will become the gradient matrix - i.e. the matrix of the derivative of each matrix element
        dmat = zeros(Float64,(2,2))
        if x > 0. # from definition in paper - all that follows is setting up the elements
                mat[1,1] = A*(1-exp(-B*x))
                dmat[1,1] = A*B*exp(-B*x)
        else 
                mat[1,1] = -A*(1-exp(B*x))
                dmat[1,1] = A*B*exp(B*x)
        end
        mat[2,2] = -mat[1,1]
        dmat[2,2] = -dmat[1,1]
        mat[1,2] = C*exp(-D*x^2)
        mat[2,1] = mat[1,2]
        dmat[1,2] = -2*D*x*mat[1,2]
        dmat[2,1] = dmat[1,2]

        return mat, dmat
end

function Tully_II(x::Float64,A::Float64=0.1,B::Float64=0.28,C::Float64=0.015,D::Float64=0.06, E::Float64=0.05)
        #
        # - input
        #   - x - value of the function to be evaluated
        #   - A, B, C, D, E - parameters for the model, Tully's original parameters are the default
        # 
        # - output
        #   - mat - 2x2 matrix of the diabatic potential, as defined in Tully's paper
        #   - dmat - 2x2 matrix of the derivatives of the diabatic potential matrix elements
        
        #init matrices of zeros
        mat = zeros(Float64,(2,2))
        #dmat will become the gradient matrix - i.e. the matrix of the derivative of each matrix element
        dmat = zeros(Float64,(2,2))
        
        mat[2,2] = -A*exp(-B*x^2)+E
        dmat[2,2] = 2*A*B*x*exp(-B*x^2)
        mat[1,2] = C*exp(-D*x^2)
        mat[2,1] = mat[1,2]
        dmat[1,2] = -2*D*x*mat[1,2]
        dmat[2,1] = dmat[1,2]

        return mat, dmat
end

function Tully_III(x::Float64,A::Float64=0.0006,B::Float64=0.1,C::Float64=0.9)
        #
        # - input
        #   - x - value of the function to be evaluated
        #   - A, B, C - parameters for the model, Tully's original parameters are the default
        # 
        # - output
        #   - mat - 2x2 matrix of the diabatic potential, as defined in Tully's paper
        #   - dmat - 2x2 matrix of the derivatives of the diabatic potential matrix elements
        
        #init matrices of zeros
        mat = zeros(Float64,(2,2))
        #dmat will become the gradient matrix - i.e. the matrix of the derivative of each matrix element
        dmat = zeros(Float64,(2,2))
        mat[1,1] = A
        mat[2,2] = -A
        if x < 0
                mat[1,2] = B*exp(C*x)
                dmat[1,2] = C*mat[1,2]
        else
                mat[1,2] = B*(2-exp(-C*x))
                dmat[1,2] = C*B*exp(-C*x)
        end
        mat[2,1] = mat[1,2]
        dmat[2,1] = dmat[1,2]

        return mat, dmat
end


function Dumbbell(x::Float64,A::Float64=0.0006,B::Float64=0.1,C::Float64=0.9,Z::Float64=10.)
        # Dumbbell Hamiltonian from J. E. Subotnik and N. Shenvi, J. Chem. Phys. 134, 2011
        # 
        # - input
        #   - x - value of the function to be evaluated
        #   - A, B, C, Z - parameters for the model, Subotnik's original parameters are the default
        # 
        # - output
        #   - mat - 2x2 matrix of the diabatic potential, as defined in Tully's paper
        #   - dmat - 2x2 matrix of the derivatives of the diabatic potential matrix elements
        
        #init matrices of zeros
        mat = zeros(Float64,(2,2))
        #dmat will become the gradient matrix - i.e. the matrix of the derivative of each matrix element
        dmat = zeros(Float64,(2,2))
        mat[1,1] = A
        mat[2,2] = -A
        if x < -Z
                mat[1,2]  =   B*( exp( C*(x-Z))+2-exp( C*(x+Z)))
                dmat[1,2] = C*B*( exp( C*(x-Z))-exp( C*(x+Z)))
        elseif x < Z
                mat[1,2]  =   B*( exp( C*(x-Z))+exp(-C*(x+Z)))
                dmat[1,2] = C*B*( exp( C*(x-Z))-exp(-C*(x+Z)))
        else
                mat[1,2]  =   B*(2-exp(-C*(x-Z))+exp(-C*(x+Z)))
                dmat[1,2] = C*B*( exp(-C*(x-Z))-exp(-C*(x+Z)))
        end
        mat[2,1] = mat[1,2]
        dmat[2,1] = dmat[1,2]

        return mat, dmat
end

function Double_Arch(x::Float64,A::Float64=0.0006,B::Float64=0.1,C::Float64=0.9,Z::Float64=4.)
        # Double arch hamiltonian from J. E. Subotnik and N. Shenvi, J. Chem. Phys. 134, 2011
        # 
        # I found a Z=0.4 problem works well also
        #
        # - input
        #   - x - value of the function to be evaluated
        #   - A, B, C, Z - parameters for the model, Subotnik's original parameters are the default
        # 
        # - output
        #   - mat - 2x2 matrix of the diabatic potential, as defined in Tully's paper
        #   - dmat - 2x2 matrix of the derivatives of the diabatic potential matrix elements
        
        #init matrices of zeros
        mat = zeros(Float64,(2,2))
        #dmat will become the gradient matrix - i.e. the matrix of the derivative of each matrix element
        dmat = zeros(Float64,(2,2))
        mat[1,1] = A
        mat[2,2] = -A
        if x < -Z
                mat[1,2]  =   B*(-exp( C*(x-Z))+exp( C*(x+Z)))
                dmat[1,2] = C*B*(-exp( C*(x-Z))+exp( C*(x+Z)))
        elseif x < Z
                mat[1,2]  =   B*(-exp( C*(x-Z))-exp(-C*(x+Z)))+2*B
                dmat[1,2] = C*B*(-exp( C*(x-Z))+exp(-C*(x+Z)))
        else
                mat[1,2]  =   B*( exp(-C*(x-Z))-exp(-C*(x+Z)))
                dmat[1,2] = C*B*(-exp(-C*(x-Z))+exp(-C*(x+Z)))
        end
        mat[2,1] = mat[1,2]
        dmat[2,1] = dmat[1,2]

        return mat, dmat
end

function Tully_V(x::Float64,A::Float64=0.05,B::Float64=0.1,C::Float64=12.,D::Float64=2.,E::Float64=0.01)
        # I found this online too, idk where it's from - it's effectively a modification of model III
        # to extend it to making going further...
        # 
        # - input
        #   - x - value of the function to be evaluated
        #   - A, B, C, D, E - parameters for the model, Tully's original parameters are the default
        # 
        # - output
        #   - mat - 2x2 matrix of the diabatic potential, as defined in Tully's paper
        #   - dmat - 2x2 matrix of the derivatives of the diabatic potential matrix elements
        t = pi/C
        #init matrices of zeros
        mat = zeros(Float64,(2,2))
        #dmat will become the gradient matrix - i.e. the matrix of the derivative of each matrix element
        dmat = zeros(Float64,(2,2))
        
        mat[1,1] = B*exp(-D*x)+E/2-A/2*cos(t)*exp(-D*x);
        mat[2,2] = B*exp(-D*x)-E/2+A/2*cos(t)*exp(-D*x);
        mat[1,2] = -A/2*sin(t)*exp(-D*x)
        mat[2,1] = mat[1,2]
        dmat[1,1] = -B*D*exp(-D*x)+A/2*D*cos(t)*exp(-D*x)
        dmat[2,2] = -B*D*exp(-D*x)-A/2*D*cos(t)*exp(-D*x)
        dmat[1,2] = D*A/2*sin(t)*exp(-D*x)
        dmat[2,1] = dmat[1,2]
        return mat, dmat
end

function Cooper_I(x::Float64,A::Float64=0.001,B::Float64=0.01,C::Float64=1.5,D::Float64=0.005)
        # Just an example of an less abrupt avoided crossing - should be no issues with any decoherence corrections...
        #
        # interesting also as a example of how to make your own potentials if you wish...


        #init matrices of zeros
        mat = zeros(Float64,(2,2))
        #dmat will become the gradient matrix - i.e. the matrix of the derivative of each matrix element
        dmat = zeros(Float64,(2,2))
        mat[1,1] = -A*x
        mat[2,2] = -mat[1,1]
        dmat[1,1] = -A
        dmat[2,2] = A
        mat[1,2] = D
        mat[2,1] = D
        return mat, dmat
end

function ADIABATISER(mat::Array{Float64}, dmat::Array{Float64})
        #
        # - input
        #   - mat - nxn matrix of diabatic potential 
        #   - dmat - nxn matrix of the derivatives of the diabatic potential matrix elements
        # 
        # - output (in form of Tuple)
        #   - evals - n-vector eigenvalues (i.e. adiabatic potential energy surfaces)
        #   - evecs - nxn array of eigenvectors (each column is an eigenvector)
        #   - grad - n-vector of gradients of the eigenvalues
        #   - coupling_mat - nxn array of non-adiabatic couplings. should be anti-symmetric and zero-diagonal.

        # use inbuilt functions to get eigenvalues and eigenvectors
        #evals = eigvals(mat)
        #evecs = eigvecs(mat)

        # use analytical function to get eigenvalues and eigenvectors
        evals = zeros(Float64,(2,2))
        evecs = ones(Float64,(2,2))
        evals[1,1] = 0.5 * (mat[1,1]+mat[2,2]-sqrt(mat[1,1]^2+mat[2,2]^2+4*mat[1,2]*mat[2,1]-2*mat[1,1]*mat[2,2]))
        evals[2,2] = 0.5 * (mat[1,1]+mat[2,2]+sqrt(mat[1,1]^2+mat[2,2]^2+4*mat[1,2]*mat[2,1]-2*mat[1,1]*mat[2,2]))
       
        evecs[1,1] = -(-mat[1,1]+mat[2,2]+sqrt(mat[1,1]^2+mat[2,2]^2+4*mat[1,2]*mat[2,1]-2*mat[1,1]*mat[2,2]))/(2*mat[2,1])
        evecs[1,2] = -(-mat[1,1]+mat[2,2]-sqrt(mat[1,1]^2+mat[2,2]^2+4*mat[1,2]*mat[2,1]-2*mat[1,1]*mat[2,2]))/(2*mat[2,1])
        evecs[:,1] /= norm(evecs[:,1])
        evecs[:,2] /= norm(evecs[:,2])


        #init matrices of zeros
        grad = zeros(Float64,size(mat,1))
        coupling_mat = zeros(Float64, size(mat))

        for i in 1:size(mat,1) # from Hellmann-Feynman theorem - d(E_i)/dx = ⟨i|dV/dx|i⟩
@views                grad[i] = evecs[:,i]' * dmat * evecs[:,i]
        end
        for i in 1:size(mat,1), j in 1:size(mat,1) # from equation d_{ij} = ⟨i|dV/dx|j⟩/(E_i-E_j)
                if i>j
@views                  coupling_mat[i,j] = evecs[:,i]' * dmat * evecs[:,j]/(evals[i]-evals[j]) 
                        coupling_mat[j,i] = -coupling_mat[i,j]
                end
        end
        return evals, evecs, grad, coupling_mat
end

function xprop(x::Float64, v::Float64,a::Float64,dt::Float64)
        return x + v*dt + 0.5*a*dt^2
end

function vprop(v::Float64,olda::Float64,newa::Float64,dt::Float64)
        return v+0.5(olda+newa)*dt
end

function grad(ham::Tuple,m::Float64,state::Int64)
        return -ham[3][state]/m
        
end

function sigmadot(sigma::Array{Complex{Float64}}, a, t::Float64) #, potmat::Array{Float64}, coupling_mat::Array{Float64},v::Float64)
        
        potmat::Array{Float64} = a[1]
        coupling_mat::Array{Float64} = a[2]
        v::Float64 = a[3]

        return -im * (2*π) * potmat * sigma - sigma * potmat - v * (coupling_mat * sigma - sigma * coupling_mat)
end

function prop_c(sigma::Array{Complex{Float64}}, t::Float64, dt::Float64,v::Float64,potmat::Array{Float64},coupling_mat::Array{Float64})
        prob = ODEProblem(sigmadot,sigma,(t-dt,t),(potmat, coupling_mat, v))
        return solve(prob)
end

function cdot(c::Array{Complex{Float64}},e,v,d)
        cd = zeros(Complex{Float64},2)
        cd[1] = -im * e[1,1]*c[1] - v*d[1,2]*c[2]
        cd[2] = -im * e[2,2]*c[2] - v*d[2,1]*c[1]
        return cd
end

function rk4(c,oldham, newham, oldv, newv, dt)
        # Runge-Kutta 4th order propagator
        a = cdot(c,oldham[1],newv,oldham[4])
        k1 = a*dt

#       linearly interp
        line = 0.5*(oldham[1]+newham[1])
        lind = 0.5*(oldham[4]+newham[4])
        linv = 0.5*(oldv+newv)
        a = cdot(c+k1/2,line,linv,lind)
        k2 = a*dt
        a = cdot(c+k2/2,line,linv,lind)
        k3 = a*dt
        a = cdot(c+k3,newham[1],newv,newham[4])
        k4 = a*dt
        c += 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        return c / norm(c)
        end

function hop(v::Float64,coupling_mat::Array{Float64},dt::Float64,coeff,state)
        #hop from state to different ones
        if state == 1
                j = 2
        else
                j = 1
        end

        g::Float64 = v*real(coupling_mat[state,j]*conj(coeff[j])*coeff[state])*dt/(abs2(coeff[state]))
        
        if g < 0.
                g = 0
        end


        rand = Random.rand()
        
        println(rand,"  ", g)
        if rand < g
                return true, j
        else
                return false, state
        end
end



function Surface_Hopping(x::Float64,v::Float64,m::Float64,dt::Float64,t::Float64,initial_state::Int64)

newx = x
newv = v
state=initial_state
egap=100.
newc = zeros(Complex{Float64},(2))
newc[2] = 1.0
newham = ADIABATISER(Tully_I(x)...)
newa = grad(newham,m,state)
newe = newham[1]
etot = 0.5*m*newv^2+newe[state]
# initial conditions 


println("Time        State      x        Etot      Epot")
println(t,"   ",state, "   ", newx,"   ",etot, "   ", newham[1][state,state])
println("σ = ", abs2.(newc))


for i in 1:1000
        t+=dt
        oldham = newham
        oldx = newx
        oldv = newv
        olda = newa
        oldc = newc

        newx = xprop(oldx,oldv,olda,dt)
        newham = ADIABATISER(Tully_I(newx)...)
        newa = grad(newham,m,state)
        newv = vprop(oldv,olda,newa,dt)
        newc = rk4(oldc,oldham,newham,oldv,newv,dt)
        fsh = hop(newv,newham[4],dt,newc,state)
        etot = 0.5*m*newv^2+newe[state]
        if fsh[1]
                if newham[1][state,state] < etot
                        oldstate = state
                        state=fsh[2]
                        newv = sqrt((etot-newham[1][oldstate,oldstate])/(etot-newham[1][state,state]))*newv
                        newa = grad(newham,m,state)
                        println("Hop from state ",oldstate," to ",state)
                else
                        println("Frustrated hop!!")
                end
        end

        println(t,"   ",state, "   ", newx,"   ",etot, "   ", newham[1][state,state], "    ", newham[4][1,2])
        println("σ = ", abs2.(newc))
        println("-----------------------------------------")
end
end


x = -10.
dt = 0.1
t=0.
v=1.
m=2000.
initial_state = 2
Surface_Hopping(x,v,m,dt,t,initial_state)
@time Surface_Hopping(x,v,m,dt,t,initial_state)






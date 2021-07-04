#!/usr/local/bin/julia

using LinearAlgebra
using Plots
using UnicodePlots

# Set of routines for calculating the energies, gradients, and couplings for the Tully Models
# See J. C. Tully, J. Chem. Phys. 1990, 93 for more details
#
# There are 3 individual routines which calculate the ⟨i|V|j⟩ and ⟨i|dV/dx|j⟩ for each of the 
# models
#
# The function ADIABATISER then performs the adiabatisation, and calculates the eigenvectors
# & eigenvalues (i.e. the electronic energies and states), the gradients of the eigenvectors
# and the non-adiabatic couplings 
#
# At the moment, there are still some problems with the phase of the NACs - the eigval func-
# tion I use seems to occasionally produce phase flips. I'll either work out an analytical 
# form of the eigenvectors or just create a phase tracker.
#
# The usage is to calculate a set of matrix elements at a given x with e.g. Tully_I(1.2)
# and then adiabatise is using ADIABATISER(Tully_I(1.2)...) - the ellipsis is crucial.
# At the moment, there are some problems with the phase. I use an analytical form of the 
# eigenvectors, but it still seems that the coupling occasionally has the wrong sign (seemingly
# in model III, at least in comparison with tully's original paper. This can be easily fixed
# in an ad-hoc manner (just multiply by -1...) but I'll have a look at fixing it properly

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

function Tully_IV(x::Float64,A::Float64=0.0006,B::Float64=0.1,C::Float64=0.9,D::Float64=0.4)
        # I found this online too, idk where it's from - it's effectively a modification of model III
        # to extend it to making going further...
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
        mat[1,1] = -A
        mat[2,2] = A
        if x < -D
                mat[1,2]  =   B*(-exp( C*(x-D))+exp( C*(x+D)))
                dmat[1,2] = C*B*(-exp( C*(x-D))+exp( C*(x+D)))
        elseif x < D
                mat[1,2]  =   B*(-exp( C*(x-D))-exp(-C*(x+D)))+2*B
                dmat[1,2] = C*B*(-exp( C*(x-D))+exp(-C*(x+D)))
        else
                mat[1,2]  =   B*( exp(-C*(x-D))-exp(-C*(x+D)))
                dmat[1,2] = C*B*(-exp(-C*(x-D))+exp(-C*(x+D)))
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
        evals = zeros(Float64,2)
        evecs = ones(Float64,(2,2))

        evals[1] = 0.5 * (mat[1,1]+mat[2,2]-sqrt(mat[1,1]^2+mat[2,2]^2+4*mat[1,2]*mat[2,1]-2*mat[1,1]*mat[2,2]))
        evals[2] = 0.5 * (mat[1,1]+mat[2,2]+sqrt(mat[1,1]^2+mat[2,2]^2+4*mat[1,2]*mat[2,1]-2*mat[1,1]*mat[2,2]))
       
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



# this section just plots them. 

length = 1000

x = range(-10.,10.,length=length)

pot_1 = zeros(Float64,length)
pot_2 = zeros(Float64,length)
der_1 = zeros(Float64,length)
der_2 = zeros(Float64,length)
coup = zeros(Float64,length)

@time @views  for i in 1:length
        a=ADIABATISER(Tully_V(x[i])...)
        pot_1[i] = a[1][1]
        pot_2[i] = a[1][2]
        der_1[i] = a[3][1]
        der_2[i] = a[3][2]
        coup[i] = a[4][1,2]
        #a = Cooper_I(x[i])
        #pot_1[i] = a[1][1,1]
        #pot_2[i] = a[1][2,2]
        #der_1[i] = a[1][1,2]
end
plot(x ,pot_1, label = "E_1", title="Tully Model V")
plot!(x,pot_2, label = "E_2")
plot!(x,der_1, label = "∇E_1")
plot!(x,der_2, label = "∇E_2")
plot!(x ,coup, label = "D_12")
savefig("Tully_V.png")

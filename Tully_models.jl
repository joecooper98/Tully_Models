#!/usr/local/bin/julia

using LinearAlgebra
using Plots
using UnicodePlots


function Tully_I(x::Float64,A::Float64=0.01,B::Float64=1.6,C::Float64=0.005,D::Float64=1.0)
   # Returns the 1st model from J. C. Tully, J. Chem. Phys. 1990, 93

        mat = zeros(Float64,(2,2))
        a = zeros(Float64,2)
        if x > 0.
                mat[1,1] = A*(1-exp(-B*x))
        else 
                mat[1,1] = -A*(1-exp(B*x))
        end
        mat[2,2] = -mat[1,1]
        mat[1,2] = C*exp(-D*x^2)
        mat[2,1] = mat[1,2]
        smat = Symmetric(mat) 
        return eigvecs(smat)
end


function Tully_Ib(x::Float64,A::Float64=0.01,B::Float64=1.6,C::Float64=0.005,D::Float64=1.0)
   # Returns the 1st model from J. C. Tully, J. Chem. Phys. 1990, 93

        mat = Symmetric([0. C*exp(-D*x^2); C*exp(-D*x^2)  0.])
        if x > 0.
                mat[1,1] = A*(1-exp(-B*x))
        else 
                mat[1,1] = -A*(1-exp(B*x))
        end
        mat[2,2] = -mat[1,1]
        return eigvals(mat),eigvecs(mat)
end




function Tully_Ia(x::Float64,A::Float64=0.01,B::Float64=1.6,C::Float64=0.005,D::Float64=1.0)
   # Returns the 1st model from J. C. Tully, J. Chem. Phys. 1990, 93
         lamb = sqrt(A^2*(1-exp(-abs(B*x)))^2+C^2*exp(-2*D*x^2))
         deriv = (x*((A^2)*abs(B)*exp(-abs(B)*abs(x)))*(1-exp(-abs(B)*abs(x)))-2*C^2*D*abs(x)*exp(-2*D*x^2))/(abs(x)*sqrt(A^2*(1-exp(-abs(B)*abs(x)))^2+C^2*exp(-2*D*x^2)))

        return -lamb, lamb, -deriv, deriv
end

function Tully_IIa(x::Float64,A::Float64=0.1,B::Float64=0.28,C::Float64=0.015,D::Float64=0.06, E::Float64=0.05)
   # Returns the 1st model from J. C. Tully, J. Chem. Phys. 1990, 93
   lamb_1 = (-A*exp(-B*x^2)+E) - sqrt((-A*exp(-B*x^2)+E)^2 + 4*(C^2*exp(-2*D*x^2)))
   lamb_2 = (-A*exp(-B*x^2)+E) + sqrt((-A*exp(-B*x^2)+E)^2 + 4*(C^2*exp(-2*D*x^2)))
   deriv_1 = 2*exp(-B*x^2)*A*B*x-(2*x*(exp(-B*x^2)*A*B*(-exp(-B*x^2)*A+E)-4*exp(-2*x^2*D)*C^2*D))/(sqrt(4*exp(-2*x^2*D)*C^2+(-exp(-B*x^2)*A+E)^2))
   deriv_2 = 2*exp(-B*x^2)*A*B*x+(2*x*(exp(-B*x^2)*A*B*(-exp(-B*x^2)*A+E)-4*exp(-2*x^2*D)*C^2*D))/(sqrt(4*exp(-2*x^2*D)*C^2+(-exp(-B*x^2)*A+E)^2))

        return lamb_1, lamb_2, deriv_1, deriv_2 
end

function Tully_IIIa(x::Float64,A::Float64=0.1,B::Float64=0.28,C::Float64=0.015,D::Float64=0.06, E::Float64=0.05)
   # Returns the 1st model from J. C. Tully, J. Chem. Phys. 1990, 93
        if x < 0
                lamb = sqrt(A^2+(B*exp(C*x))^2)
                deriv = exp(2*C*x)*B^2*C/(lamb)        
        else
                lamb = sqrt(A^2+(B*(2-exp(-C*x)))^2)
                deriv = exp(-C*x)*B^2*C*(-exp(-C*x)+2)/lamb
        end

        return -lamb, lamb, -deriv, deriv 
end




length = 1000

x = range(-10.,10.,length=length)

pot_1 = zeros(Float64,length)
pot_2 = zeros(Float64,length)
der_1 = zeros(Float64,length)
der_2 = zeros(Float64,length)

for i in 1:length
        a=Tully_IIIa(x[i])
        pot_1[i] = a[1]
        pot_2[i] = a[2]
        der_1[i] = a[3]
        der_2[i] = a[4]
end

plot(x,pot_1)
plot!(x,pot_2)
plot!(x,der_1)
plot!(x,der_2)
savefig("plot.png")

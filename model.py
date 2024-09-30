from fourier_poisson import fourier_poisson_single
from gradients import periodic_laplace_N,periodic_gradient
from poisson_bracket import arakawa_vec
from jax import jit
import jax.numpy as jnp
from functools import partial

class HW:
    def __init__(
            self,
            dx: float,
            N: int,
            c1:float,
            nu: float,
            kappa:float
    ):
        self.poisson_solver=fourier_poisson_single
        self.diffuse_N=periodic_laplace_N
        self.poisson_bracket=arakawa_vec
        self.gradient_func=periodic_gradient
        self.N=N
        self.dx=dx
        self.c1=c1
        self.nu=nu
        self.kappa=kappa

    @partial(jit, static_argnums=(0,))
    def gradient_2d(self,density,omega,dx,phi):
        diff=phi-density
        # new omega
        o=self.c1*diff
        bracket_phi_omega=-self.poisson_bracket(zeta=phi,psi=omega,dx=dx)
        o+=bracket_phi_omega
        diffuse_omega=-self.nu*self.diffuse_N(arr=omega,dx=dx,N=self.N)
        o+=diffuse_omega

        # new density
        n = self.c1 * diff
        bracket_phi_n=self.poisson_bracket(zeta=phi,psi=density,dx=dx)
        n-=bracket_phi_n
        gradient_phi=self.kappa*self.gradient_func(input_field=phi,dx=dx,axis=0)
        n-=gradient_phi
        diffuse_n=self.nu*self.diffuse_N(arr=density,dx=dx,N=self.N)
        n-=diffuse_n

        # assert o.shape == omega.shape and n.shape == density.shape
        # assert o.dtype == omega.dtype and n.dtype == density.dtype

        return o,n

    @partial(jit, static_argnums=(0,))
    def get_phi(self,omega,dx):
        o_mean=jnp.mean(omega)
        centered_omega=omega-o_mean
        phi=self.poisson_solver(tensor=centered_omega,dx=dx)
        return phi

    @partial(jit, static_argnums=(0,))
    def rk4_step(self,dt,dx,pn,n,o):
        # yn
        # phi=pn, density=n, omega=o

        # y1
        # k1 is a dictionary, representing the gradient of n and o w.r.t. time
        do1,dn1=self.gradient_2d(density=n,omega=o,dx=dx,phi=pn)
        # obtain o1,n1,p1 to update k2
        o1=o+do1*dt*0.5
        n1=n+dn1*dt*0.5
        p1=self.get_phi(omega=o1,dx=dx)

        #y2
        do2,dn2=self.gradient_2d(density=n1,omega=o1,dx=dx,phi=p1)
        o2=o1+do2*dt*0.5
        n2=n1+dn2*dt*0.5
        p2=self.get_phi(omega=o2,dx=dx)

        # y3
        do3,dn3=self.gradient_2d(density=n2,omega=o2,dx=dx,phi=p2)
        o3=o2+do3*dt
        n3=n2+dn3*dt
        p3=self.get_phi(omega=o3,dx=dx)

        # y4
        do4,dn4=self.gradient_2d(density=n3,omega=o3,dx=dx,phi=p3)

        #yn+1
        n_new=n+(1/6*dt)*(dn1 + 2*dn2 + 2*dn3 + dn4)
        o_new=o+(1/6*dt)*(do1 + 2*do2 + 2*do3 + do4)
        phi_new=self.get_phi(omega=o_new,dx=dx)

        # assert parameters and outputs have consistent shape and dtype to prevent jit from recompiling
        # assert n_new.shape == n.shape and o_new.shape == o.shape and phi_new.shape == pn.shape
        # assert n_new.dtype == n.dtype and o_new.dtype == o.dtype and phi_new.dtype == pn.dtype

        return n_new,o_new,phi_new

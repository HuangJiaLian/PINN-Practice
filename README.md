# PINN-Practice
Physical Informed Neural Network Practice

## Diffusion Equation

我们要求解的一个微分方程, 其中 $x, t\in[0,1]$,
$$
\left\{ \begin{aligned} u_t(x,t)  &=  \ u_{xx}(x,t) -c*sin(2\pi x)*u(x,t) \\   u(x,t)& =  \ u(x+1,t) \\    u(x,0) & = 1 \end{aligned} \right.   (1)
$$

定义$f(x,t)$ , $g(x,t)$ , $h(x)$分别为:

$$
\left\{ \begin{aligned} f: &=u_t(x,t)  -  \ u_{xx}(x,t) +c*sin(2\pi x)*u(x,t) \\   g :&= u(x,t) - u(x+1,t) \\    h :&= u(x,0) - 1 \end{aligned} \right.    (2)
$$



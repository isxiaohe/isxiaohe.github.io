---
title: "Flow Matching and Diffusion Models"
date: 2026-10-12
tldr: "Notes on ODE and SDE generation, conditional flow matching, score matching, diffusion models, and conditional generation with classifier guidance and classifier-free guidance."
math: true
---

The common goal is to transform an easy base distribution into the data distribution:

$$
p_0=p_{\mathrm{base}}\approx\mathcal N(0,I),
\qquad
p_1=p_{\mathrm{data}}.
$$

I use the convention that generation runs **forward** from noise at $t=0$ to data at $t=1$. Many DDPM papers use the reverse convention, so signs and schedule endpoints must be translated rather than copied mechanically.

## 1. Generation with SDE

### ODE and SDE

Let $X_t\in\mathbb R^d$ be a random variable with density $p_t$. An ODE generator specifies a time-dependent velocity field $u_t$:

$$
\frac{\mathrm dX_t}{\mathrm dt}=u_t(X_t),
\qquad X_0\sim p_{\mathrm{base}}.
$$

The solution gives a flow map $\psi_t$ with $X_t=\psi_t(X_0)$. A successful model satisfies $(\psi_1)_\#p_{\mathrm{base}}=p_{\mathrm{data}}$. This is a statement about the distribution of many particles, not a claim that a particular Gaussian noise vector has a meaningful, unique matching data point.

Probability conservation yields the continuity equation

$$
\partial_t p_t(x)=-\nabla_x\cdot\bigl(p_t(x)u_t(x)\bigr).
\tag{1}
$$

The product $p_tu_t$ is probability flux: density changes at $x$ only when mass enters or leaves a small neighborhood of $x$. Sampling is numerical ODE integration, for example Euler's method

$$
X_{t+h}=X_t+h\,u_t(X_t)+O(h^2).
$$

An SDE generator instead has the form

$$
\mathrm dX_t=f_t(X_t)\,\mathrm dt+g_t\,\mathrm dW_t,
$$

where $W_t$ is Brownian motion. Since $W_{t+h}-W_t\sim\mathcal N(0,hI)$, Euler--Maruyama is

$$
X_{t+h}=X_t+f_t(X_t)h+g_t\sqrt h\,\varepsilon,
\qquad\varepsilon\sim\mathcal N(0,I).
$$

### Fokker--Planck equation and probability-flow ODE

Its density satisfies the Fokker--Planck equation

$$
\partial_t p_t
=-\nabla_x\cdot(p_tf_t)+\frac12g_t^2\Delta_xp_t.
\tag{2}
$$

Using $\Delta p=\nabla\cdot(p\nabla\log p)$, an SDE with

$$
f_t(x)=u_t(x)+\frac12g_t^2\nabla_x\log p_t(x)
$$

has exactly the same marginal evolution as the ODE in (1). Conversely, a diffusion SDE with drift $f_t$ has the probability-flow ODE

$$
\frac{\mathrm dX_t}{\mathrm dt}
=f_t(X_t)-\frac12g_t^2\nabla_x\log p_t(X_t).
\tag{3}
$$

So an ODE and an SDE need not produce the same *sample trajectories*, but they can produce the same density $p_t$ at every time. Flow matching learns the ODE representation directly.

### Why the ODE and SDE are equivalent at the distribution level

The word *equivalent* has a precise and limited meaning here. If both processes start from the same $p_0$, they have the same one-time marginal density $p_t$ for every $t$. It does **not** mean that an ODE trajectory and an SDE trajectory with the same initial point coincide; the latter contains fresh Brownian randomness at every infinitesimal step.

To see the equivalence explicitly, start from the SDE Fokker--Planck equation (2) and choose

$$
f_t(x)=u_t(x)+\frac12g_t^2\nabla_x\log p_t(x).
$$

For the scalar, time-dependent $g_t$ used here,

$$
p_t(x)f_t(x)
=p_t(x)u_t(x)+\frac12g_t^2p_t(x)\nabla_x\log p_t(x)
=p_t(x)u_t(x)+\frac12g_t^2\nabla_xp_t(x).
$$

Substitution into (2) gives a term-by-term cancellation:

$$
\begin{aligned}
\partial_t p_t
&=-\nabla_x\cdot\left(p_tu_t+\frac12g_t^2\nabla_xp_t\right)
+\frac12g_t^2\Delta_xp_t\\
&=-\nabla_x\cdot(p_tu_t)
-\frac12g_t^2\Delta_xp_t
+\frac12g_t^2\Delta_xp_t\\
&=-\nabla_x\cdot(p_tu_t).
\end{aligned}
$$

This is exactly the ODE continuity equation (1). The SDE's Brownian term spreads probability mass, producing $+\frac12g_t^2\Delta p_t$. The extra score drift pulls particles toward higher-density regions and produces the compensating negative term. At the level of the whole particle cloud, the two effects cancel, leaving only the desired transport by $u_t$.

An instructive stationary example is $p_t=\mathcal N(0,I)$ for all $t$ and $u_t=0$. The ODE is simply $\mathrm dX_t/\mathrm dt=0$: each particle stays fixed. The associated SDE is

$$
\mathrm dX_t=-\frac12g^2X_t\,\mathrm dt+g\,\mathrm dW_t,
$$

because $\nabla_x\log\mathcal N(0,I)=-x$. This Ornstein--Uhlenbeck process makes every particle wander randomly while pulling it back toward the origin. Yet if $X_0\sim\mathcal N(0,I)$, then $X_t\sim\mathcal N(0,I)$ for all $t$, just as for the static ODE. Same distribution; very different paths.

Finally, in a practical diffusion model we specify a forward SDE and learn its score. Rearranging the construction above yields the probability-flow ODE in (3). This ODE is not the reverse-time SDE: it is a deterministic process with the same marginals as the chosen SDE. The simple formula here also assumes $g_t$ has no spatial dependence; state-dependent or matrix-valued diffusion requires the corresponding divergence correction.

## 2. Flow Matching

### Flow model (recall)

The continuity equation says which field would transport a chosen path $p_t$, but normally $p_t$ and $u_t$ are unknown. Flow matching constructs a family of easy conditional paths and then mixes them.

### Probability paths

Draw a clean datum $Z\sim p_{\mathrm{data}}$. For every fixed $z$, choose $p_t(x\mid z)$ with

$$
p_0(x\mid z)=p_{\mathrm{base}}(x),
\qquad
p_1(x\mid z)=\delta_z(x),
$$

where $\delta_z$ is the Dirac point mass, defined by

$$
\int\varphi(x)\delta_z(x)\,\mathrm dx=\varphi(z).
$$

The marginal path is the mixture

$$
p_t(x)=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,\mathrm dz.
\tag{4}
$$

It has the desired endpoints: at $t=0$ all components equal $p_{\mathrm{base}}$, and at $t=1$ integrating $\delta_z(x)$ against $p_{\mathrm{data}}(z)$ returns $p_{\mathrm{data}}(x)$.

### Conditional and marginal vector fields

Fix one target datum $z$. A **conditional vector field** $u_t(x\mid z)$ defines a flow map $\psi_t^z$ through

$$
\frac{\mathrm d}{\mathrm dt}\psi_t^z(x_0)
=u_t\bigl(\psi_t^z(x_0)\mid z\bigr),
\qquad \psi_0^z(x_0)=x_0.
$$

Its defining task is to transport the conditional initial distribution along the chosen conditional probability path:

$$
X_0\sim p_0(\cdot\mid z)
\qquad\Longrightarrow\qquad
X_t=\psi_t^z(X_0)\sim p_t(\cdot\mid z).
\tag{5}
$$

In our construction $p_0(\cdot\mid z)=p_{\mathrm{base}}$ and $p_1(\cdot\mid z)=\delta_z$, so every conditional field begins from the same noise distribution but ends concentrated at its own datum $z$.

The continuity equation now arises as a consequence of this transport requirement. Let $A$ be any fixed region. The probability inside $A$ can change only through its boundary. Since a particle at $x$ moves with velocity $u_t(x\mid z)$, the outward probability flux is $p_t(x\mid z)u_t(x\mid z)\cdot n(x)$, where $n$ is the outward unit normal. Thus

$$
\frac{\mathrm d}{\mathrm dt}\int_Ap_t(x\mid z)\,\mathrm dx
=-\int_{\partial A}p_t(x\mid z)u_t(x\mid z)\cdot n(x)\,\mathrm dS.
$$

Applying the divergence theorem and using that this holds for every region $A$ gives

$$
\partial_tp_t(x\mid z)
=-\nabla_x\cdot\bigl(p_t(x\mid z)u_t(x\mid z)\bigr).
\tag{6}
$$

So the continuity equation is not an additional modeling choice: for a sufficiently regular deterministic flow, it is exactly the density-level expression of $X_0\sim p_0(\cdot\mid z)\Rightarrow X_t\sim p_t(\cdot\mid z)$.

The required **marginal** velocity averages all conditional directions that could have produced the same intermediate point $x$:

$$
u_t(x)
=\int u_t(x\mid z)p_t(z\mid x)\,\mathrm dz
=\frac{\int u_t(x\mid z)p_t(x\mid z)p_{\mathrm{data}}(z)\,\mathrm dz}{p_t(x)}.
\tag{7}
$$

The posterior $p_t(z\mid x)$ matters: an intermediate noisy point can be compatible with many clean targets, so its correct direction is the weighted mean of their directions.

### Proof: the marginal field transports the marginal path

Differentiate (4), move the derivative through the integral, and use the conditional continuity equation (6):

$$
\begin{aligned}
\partial_tp_t(x)
&=\int\partial_tp_t(x\mid z)p_{\mathrm{data}}(z)\,\mathrm dz\\
&=-\nabla_x\cdot
\left(
\int p_t(x\mid z)u_t(x\mid z)p_{\mathrm{data}}(z)\,\mathrm dz
\right).
\end{aligned}
$$

By (7), the vector inside parentheses is exactly $p_t(x)u_t(x)$. Therefore

$$
\partial_tp_t(x)=-\nabla_x\cdot\bigl(p_t(x)u_t(x)\bigr),
$$

which is the marginal continuity equation. Thus solving $\dot X_t=u_t(X_t)$ from $X_0\sim p_{\mathrm{base}}$ reaches $X_1\sim p_{\mathrm{data}}$.

### Learning target: conditional flow matching

The ideal but inaccessible marginal loss is

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{t\sim\mathrm{Unif}[0,1],\,x\sim p_t}
\left\|u_\theta(t,x)-u_t(x)\right\|_2^2.
$$

Instead we train with the tractable conditional target:

$$
\mathcal L_{\mathrm{CFM}}(\theta)
=\mathbb E_{\substack{t\sim\mathrm{Unif}[0,1],\,z\sim p_{\mathrm{data}}\\x\sim p_t(\cdot\mid z)}}
\left\|u_\theta(t,x)-u_t(x\mid z)\right\|_2^2.
\tag{8}
$$

The network is given $(t,x)$, not $z$. This omission is essential: regression must average the conflicting conditional directions instead of memorizing the target datum.

To prove that (8) has the same parameter gradient as marginal flow matching, first expand the marginal loss and drop terms independent of $\theta$:

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{t,x\sim p_t}\|u_\theta(t,x)\|_2^2
-2\mathbb E_{t,x\sim p_t}\left[u_\theta(t,x)^\top u_t(x)\right]
+C_1.
$$

For its cross term, insert $p_t(x)u_t(x)$ from (7):

$$
\begin{aligned}
&\int p_t(x)u_\theta(t,x)^\top u_t(x)\,\mathrm dx\\
&=\iint u_\theta(t,x)^\top u_t(x\mid z)
p_t(x\mid z)p_{\mathrm{data}}(z)\,\mathrm dz\,\mathrm dx.
\end{aligned}
$$

The first quadratic term can likewise use the mixture identity (4). Hence

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{t,z,x\sim p_t(\cdot\mid z)}
\left[\|u_\theta(t,x)\|_2^2
-2u_\theta(t,x)^\top u_t(x\mid z)\right]+C_1.
$$

Expanding (8) gives the same two $\theta$-dependent terms:

$$
\mathcal L_{\mathrm{CFM}}(\theta)
=\mathbb E_{t,z,x\sim p_t(\cdot\mid z)}
\left[\|u_\theta(t,x)\|_2^2
-2u_\theta(t,x)^\top u_t(x\mid z)\right]+C_2.
$$

The constants $C_1,C_2$ differ, but neither depends on $\theta$. Therefore

$$
\nabla_\theta\mathcal L_{\mathrm{CFM}}
=\nabla_\theta\mathcal L_{\mathrm{FM}}.
$$

Equivalently, the squared-error optimum is the conditional expectation

$$
u_\theta^*(t,x)
=\mathbb E\left[u_t(x\mid Z)\mid X_t=x\right]
=u_t(x).
$$

### Example: Gaussian probability path

The standard conditional path is Gaussian:

$$
p_t(x\mid z)=\mathcal N\bigl(x;\alpha_tz,\beta_t^2I\bigr),
$$

with $\alpha_0=0$, $\beta_0=1$, $\alpha_1=1$, and $\beta_1=0$. Reparameterize it as

$$
X_t=\alpha_tZ+\beta_t\varepsilon,
\qquad\varepsilon\sim\mathcal N(0,I).
\tag{9}
$$

For fixed $Z=z$ and $\varepsilon$, differentiation gives

$$
\frac{\mathrm dX_t}{\mathrm dt}
=\dot\alpha_tz+\dot\beta_t\varepsilon.
$$

Since $\varepsilon=(x-\alpha_tz)/\beta_t$, the conditional velocity field is

$$
u_t(x\mid z)
=\left(\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t\right)z
+\frac{\dot\beta_t}{\beta_t}x.
\tag{10}
$$

For training, evaluate the equivalent and stable sampled form

$$
u_t(X_t\mid Z)=\dot\alpha_tZ+\dot\beta_t\varepsilon,
$$

rather than dividing by a very small $\beta_t$ near $t=1$. Thus one batch is simply

```python
t   = uniform(0, 1, batch_size)
z   = next(data_loader)
eps = randn_like(z)
x_t = alpha(t) * z + beta(t) * eps
v_t = d_alpha(t) * z + d_beta(t) * eps
loss = mean(sum((v_theta(t, x_t) - v_t)**2, dim=channels_and_space))
```

## 3. Score Matching and Diffusion Models

### Conditional and marginal score functions

The conditional score of this Gaussian is

$$
\nabla_x\log p_t(x\mid z)=-\frac{x-\alpha_tz}{\beta_t^2}.
\tag{11}
$$

Substituting (11) into (10) gives

$$
u_t(x\mid z)=a_t\nabla_x\log p_t(x\mid z)+b_tx,
$$

where

$$
a_t=\beta_t^2\frac{\dot\alpha_t}{\alpha_t}-\beta_t\dot\beta_t,
\qquad
b_t=\frac{\dot\alpha_t}{\alpha_t}.
\tag{12}
$$

Why does the same identity hold after marginalizing? Differentiate (4) with respect to $x$:

$$
\begin{aligned}
\nabla_x\log p_t(x)
&=\frac{\nabla_xp_t(x)}{p_t(x)}\\
&=\int\nabla_x\log p_t(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,\mathrm dz\\
&=\mathbb E\left[\nabla_x\log p_t(x\mid Z)\mid X_t=x\right].
\end{aligned}
$$

Taking the same posterior average of (11) therefore yields

$$
u_t(x)=a_t\nabla_x\log p_t(x)+b_tx.
\tag{13}
$$

This is the algebraic bridge between a velocity field and a score field. Formula (11) assumes $\alpha_t>0$; use the sampled velocity target near the endpoint rather than the singular coefficient form.

### Denoising score matching

Denoising score matching trains a score network $s_\theta$ using (11):

$$
\mathcal L_{\mathrm{DSM}}(\theta)
=\mathbb E_{t,z,\varepsilon}
\left[\lambda(t)
\left\|s_\theta(t,\alpha_tz+\beta_t\varepsilon)
+\frac{\varepsilon}{\beta_t}\right\|_2^2\right].
$$

The target $-\varepsilon/\beta_t$ is the conditional score. By the same expansion used for CFM, its gradient matches regression to the marginal score. If a diffusion network predicts noise $\varepsilon_\theta$, then $s_\theta=-\varepsilon_\theta/\beta_t$.

### The SDE view of a diffusion model

Keep the convention used everywhere else in this note:

$$
t=0:\;X_t\sim p_{\mathrm{base}},
\qquad
t=1:\;X_t\sim p_{\mathrm{data}}.
$$

The deterministic field in (13) already transports this path. To obtain a stochastic diffusion-model sampler for the **same** path, choose any nonnegative noise schedule $g_t$ and use the SDE

$$
\mathrm dX_t
=f_t(X_t)\,\mathrm dt+g_t\,\mathrm dW_t,
\qquad X_0\sim p_{\mathrm{base}},
$$

with drift

$$
f_t(x)
=u_t(x)+\frac12g_t^2\nabla_x\log p_t(x).
\tag{14}
$$

Section 1 proved why this choice works: the random Brownian increment spreads probability, while the extra score drift cancels that spreading at the density level. Consequently this SDE has exactly the same marginals $p_t$ as the flow-matching ODE, while its individual sample paths are random.

For the Gaussian path, (13) makes the dependence on the score explicit. Substituting $u_t(x)=a_t\nabla_x\log p_t(x)+b_tx$ into (14) gives

$$
f_t(x)
=\left(a_t+\frac12g_t^2\right)\nabla_x\log p_t(x)+b_tx.
$$

Thus a learned score model directly defines the stochastic sampler:

$$
\mathrm dX_t
=\left[
\left(a_t+\frac12g_t^2\right)s_\theta(t,X_t)
+b_tX_t
\right]\mathrm dt
+g_t\,\mathrm dW_t,
\qquad X_0\sim\mathcal N(0,I).
\tag{15}
$$

This is the SDE form of generation in the current time direction: start from noise and integrate from $t=0$ to $t=1$. The deterministic probability-flow ODE associated with any SDE is

$$
\frac{\mathrm dX_t}{\mathrm dt}
=f_t(X_t)-\frac12g_t^2\nabla_x\log p_t(X_t).
$$

Using (14), the right-hand side reduces to $u_t(X_t)$. Therefore the stochastic sampler (15) and the flow-matching ODE follow the same sequence of distributions $p_t$; setting $g_t=0$ recovers the deterministic sampler immediately.

Many diffusion papers introduce the same construction by first writing a *data-to-noise* forward SDE and then reversing it. That is only a different temporal naming convention. Here no reindexing is needed: the learned score appears directly in the noise-to-data generative SDE above.

## 4. Conditional Generation

So far the learned field transports the unconditional density $p_t(x)$. Conditional generation asks for a different endpoint distribution for every condition $y$:

$$
X_0\sim p_{\mathrm{base}}
\qquad\Longrightarrow\qquad
X_1\sim p_{\mathrm{data}}(x\mid y).
$$

The condition $Y$ may be a class label, text embedding, reference image, action goal, or another modality. It is held fixed during a sampling trajectory; the generated variable $X_t$ changes with $t$.

### Direct conditional flow matching

Let $(Z,Y)\sim p_{\mathrm{data}}(z,y)$. If the conditional probability path still corrupts only the target datum $z$, then the same Gaussian path is used:

$$
p_t(x\mid z,y)=p_t(x\mid z),
\qquad
X_t=\alpha_tZ+\beta_t\varepsilon.
$$

The condition is not needed to construct this elementary path; it is needed to tell the network *which conditional distribution* to generate. We train

$$
\mathcal L_{\mathrm{CCFM}}(\theta)
=\mathbb E_{t,(z,y)\sim p_{\mathrm{data}},\,x\sim p_t(\cdot\mid z)}
\left\|u_\theta(t,x,y)-u_t(x\mid z)\right\|_2^2.
\tag{16}
$$

The apparently $y$-independent target is correct. For fixed $y$, least-squares regression averages over the data points compatible with that condition:

$$
u_\theta^*(t,x,y)
=\mathbb E\left[u_t(x\mid Z)\mid X_t=x,Y=y\right]
=u_t(x\mid y).
$$

More explicitly, the conditional marginal field is

$$
u_t(x\mid y)
=\int u_t(x\mid z),p_t(z\mid x,y)\,\mathrm dz.
\tag{17}
$$

It transports the conditional marginal density

$$
p_t(x\mid y)=\int p_t(x\mid z)p_{\mathrm{data}}(z\mid y)\,\mathrm dz.
$$

The proof is the same as for the unconditional field: integrate the conditional continuity equation over $p_{\mathrm{data}}(z\mid y)$, then identify the probability flux with $p_t(x\mid y)u_t(x\mid y)$. At $t=1$, this density is $p_{\mathrm{data}}(x\mid y)$.

### Classifier guidance

For a score-affine field as in (13), Bayes' rule gives

$$
\nabla_x\log p_t(x\mid y)
=\nabla_x\log p_t(x)+\nabla_x\log p_t(y\mid x),
$$

and hence the conditional velocity is

$$
u_t(x\mid y)=u_t(x)+a_t\nabla_x\log p_t(y\mid x).
\tag{18}
$$

This provides a second route to conditional generation: train an unconditional score/velocity model, then separately train a time-conditioned classifier $p_\phi(y\mid x,t)$ on noisy inputs. Its gradient

$$
\nabla_x\log p_\phi(y\mid x,t)
$$

points toward perturbations of $x$ that make condition $y$ more likely. Equation (18) adds this direction to the unconditional vector field. A classifier trained only on clean data is not enough: during most of sampling, $x$ is a noisy intermediate state and the classifier must recognize $y$ at that noise level.

Guidance strength can be adjusted by a scalar $w$:

$$
u_{\mathrm{cg}}(t,x,y)
=u_\theta(t,x)+w\,a_t\nabla_x\log p_\phi(y\mid x,t).
$$

$w=1$ targets the nominal conditional field when the models are exact. Larger $w$ trades diversity for condition adherence and can push samples away from the true conditional distribution, causing saturation or artifacts.

### Classifier-free guidance

Classifier-free guidance (CFG) avoids a separate classifier. During training, replace $y$ by a null condition $\varnothing$ with probability $p_{\mathrm{drop}}$. The same network then learns both

$$
u_\theta(t,x,y)\approx u_t(x\mid y),
\qquad
u_\theta(t,x,\varnothing)\approx u_t(x).
$$

At sampling time use

$$
u_{\mathrm{cfg}}(t,x,y)
=u_\theta(t,x,\varnothing)
+w\left[u_\theta(t,x,y)-u_\theta(t,x,\varnothing)\right].
\tag{19}
$$

$w=0$ is unconditional sampling, $w=1$ is ordinary conditional sampling, and $w>1$ extrapolates the conditional effect. To see why this resembles classifier guidance, assume the two predictions are exact and use the score-affine form:

$$
\begin{aligned}
u_{\mathrm{cfg}}
&=(1-w)u_t(x)+w\,u_t(x\mid y)\\
&=u_t(x)+w\,a_t\left[
\nabla_x\log p_t(x\mid y)-\nabla_x\log p_t(x)
\right]\\
&=u_t(x)+w\,a_t\nabla_x\log p_t(y\mid x).
\end{aligned}
$$

Thus CFG estimates the classifier-guidance direction from two vector-field evaluations. In score language, it follows the score of an unnormalized, sharpened density

$$
p_t(x)\,p_t(y\mid x)^w.
$$

This also explains why $w>1$ deliberately changes the target distribution: it raises the conditional likelihood to a power larger than one.

### Sampling with a condition

For deterministic flow matching or a probability-flow ODE, sample $x_0\sim\mathcal N(0,I)$ and integrate

$$
\frac{\mathrm dx_t}{\mathrm dt}=u_{\mathrm{cfg}}(t,x_t,y)
$$

from $t=0$ to $t=1$. For the stochastic SDE form in (15), replace the learned field by the guided field consistently in the drift and add the Brownian increment at each solver step. In either case, the condition must be passed to every network evaluation, while the initial noise remains independent of $y$ unless the model explicitly adopts a condition-dependent prior.

The main implementation pitfalls are a reversed time convention, forgetting the dot in $\dot\alpha_t,\dot\beta_t$, confusing clean data $z$ with noisy state $x$, using a classifier that was not trained on noisy states, and applying an excessively large guidance scale without considering solver stability.

## References

- Lipman et al., *Flow Matching for Generative Modeling* (2023).
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* (2021).
- Ho et al., *Denoising Diffusion Probabilistic Models* (2020).

\renewcommand{\vec}[1]{\mathbf{#1}}
\newcommand{\R}{\mathbb{R}}


# Multi-camera NERF Project

We want to extend the NERF model (Mildenhall et al., 2020) to include multiple
camera inputs, where each camera has different perspectives and potentially
different intrinsics. The goal is to see if we can design a pipeline with
differentiable convex optimization layers that can be trained end-to-end. Each
[optimization](optimization.md) layer would correspond to some domain-specific problem such as
finding the intrinsic parameters of the camera. The goal would be to see if we
could achieve better sample complexity by using these domain-specific
optimization layers instead of training a baseline model entirely end to end.

## Related Work

## Preliminaries

### NeRF (Mildenhall et al., 2020)

The authors introduce a  method for implicitly representing a scene in a
multilayer perceptron (MLP).  We can represent the MLP as a non-linear function
$F_\Theta$ whose domain is $\R^5$ (3 dimensions for the spatial location
$\vec{x} = (x, y, z)$ and two dimensions for the viewing angle
$\vec{d} = (\theta, \phi)$) and whose co-domain is $\R^4$ (3 dimensions for color
$\vec{c} = (r, g, b)$ and 1 dimension for the volume density or opacity $\sigma$)
$F_\Theta : (\vec{x}, \vec{d}) \to (\vec{c}, \sigma)$.

Because volume rendering is differentiable, the only input required is a set of
images with known camera poses. Training is carried out by using standard
backpropagation with gradient descent to minimize the reconstruction loss
between the predicted image from the camera and the actual image. Using
multiple camera locations and viewing angles, the network is encouraged to
converge to a consistent representation of the scene.

Since $\sigma$ is not a function of the viewing angle, the MLP first processes
$\vec{x}$ using an 8 layer fully connected network with 1 skip connection at
layer 5, where every layer is 256 dimensional, with ReLU activations to predict
$\sigma$, rectified by another ReLU, and a 256 dimensional auxiliary feature
vector. $\vec{c}$ is predicted using a concatenation of $\vec{d}$ and the 256
dimensional feature vector by passing it through a single 128 dimension
fully-connected layer with ReLU activations and a final sigmoid layer.

The steps to generate the NeRF image are:

1. Extend rays from the camera through the scene and generate a sampled set of
   3D points along the ray.
2. Use the points with the viewing angle as input to the MLP to get the colors
   and volume densities.
3. Integrate over the colors and volume densities to get the final color for
   the pixel represented for each ray.

Let $\sigma(\vec{x})$ be the differential probability of a ray terminating at
location $\vec{x}$, and let $C(\vec{r})$ be the expected color of the camera
ray $\vec{r}(t) = \vec{o} + t \vec{d}$, where $t$ goes from $t_n$ to  $t_f$.

$$
C(\vec{r}) = \int_{t_n}^{t_f} T(t)\sigma(\vec{r}(t))\vec{c}(\vec{r}(t), \vec{d}) dt
$$

where $T(t) = \exp \left(-\int_{t_n}^{t}\sigma(\vec{r}(s))ds\right)$ is
the accumulated transmittance along the ray i.e. the probability that the ray
travels from $t_n$ to $t$ without hitting any other particle. To estimate
$C(\vec{r})$, the authors use numerical quadrature with a stratified sampling
approach that selects a point to sample uniformly at random from each one of $N$
evenly-spaced bins. The quadrature rule used is:

$$
\begin{aligned}
\hat C(\vec{r}) &= \sum_{i = [N]} T_i (1 - \exp(-\sigma_i \delta_i))\vec{c}_i \\
T_i &= \exp \left(-\sum_{j=[i-1]}\sigma_j \delta_j \right) \\
\delta_i &= t_{i+1} - t_i
\end{aligned}
$$

Using a basic implementation of this method does not converge to a sufficiently
high-resolution representation and is inefficient. Instead, the authors
transform the 5D input using a positional encoding so the MLP can represent
higher frequency functions. Also, a hierarchical sampling procedure is used to
reduce the number of queries required for this high-frequency scene
representation.

Neural networks are biased towards learning lower frequency functions so to
capture the higher frequency details the authors map the inputs into a higher
dimensional space using high frequency functions before passing them to the
network. The mapping is $\gamma : \R \to \R^{2L}$, where

$$
\begin{aligned}
\gamma(p) = \left( \sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p) \right)
\end{aligned}
$$ 

$\gamma$ is applied separately to every coordinate in $\vec{x}$ and $\vec{d}$
($\vec{x}$ and $\vec{d}$ are normalized so every element lies in $[-1, 1]$).
The authors used $L = 10$ for $\gamma(\vec{x})$ and $L = 4$ for  $\gamma(\vec{d})$.

Since many of the sampled points may lie in free space or occluded regions, the
authors propose a more efficient hierarchical sampling technique that allocates
samples proportional to the expected effect on the final rendering. They use
a **coarse** and **fine** network. For the coarse method, they use the
aforementioned stratified sampling technique for $N_c$ locations. With the
results from the coarse network, the fine network is trained based on 

$$
\begin{aligned}
\hat C_c(\vec{r}) &= \sum_{i \in [N_c]} w_i c_i \\
w_i &= T_i (1 - \exp(-\sigma_i \delta_i)).
\end{aligned}
$$ 

The normalized the weights $\hat\vec{w} = \vec{w} / ||\vec{w}||$ produce a pdf
along the ray. Sampling $N_f$ locations from this distribution using inverse
transform sampling, and evaluating the fine network at the union of both
samples gives final rendered color.

The input data for an scene are the RGB images of the scene, the corresponding
camera poses and the intrinsic parameters, and the scene bounds. The authors
use the COLMAP package to estimate the parameters from real data. For each
iteration a random batch of rays are sampled as a batch of pixels from the
dataset. The hierarchical sampling is then carried out on the batch of rays.
Let $\mathcal R$ be the set of rays in each batch. Then

$$
\begin{aligned}
\mathcal L = \sum_{\vec{r} \in \mathcal R} \left[ \| \hat C_c(\vec{r}) - C(\vec{r}) \|_2^2 + \| \hat C_f(\vec{r}) - C(\vec{r}) \|_2^2 \right]
\end{aligned}
$$ 

The authors used a batch size of 4096 rays, each sampled with $N_c = 64$ and
$N_f = 128$. They use an Adam optimizer with learning rate that begins at 5e-4
and decays exponentially to 5e-5 over the course of the optimization. The rest
of the hyperparameters were left at default values. Convergence took around
100-300k iterations on a single NVIDIA V100 GPU.

One of the datasets they used for this task is the DeepVoxels dataset that is a
synthetic set of 3D objects. Another dataset is a set of 8 scenes captured with
a handheld cellphone, captured with 20 to 62 images and $\frac{1}{8}$ held out
as a test set.

### Convex Optimization Layers

## Methodology

## References

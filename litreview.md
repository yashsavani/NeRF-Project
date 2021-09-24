\renewcommand{\vec}[1]{\mathbf{#1}}
\newcommand{\R}{\mathbb{R}}

---
title:  'Literature review on current state of implicit neural scene representations'
author:
- Yash Savani,
- Computer Science Department, Carnegie Mellon University,
- ysavani@cs.cmu.edu
---

# Introduction

Since it's release in 2020, the NeRF model [@mildenhall_nerf_2020] has
created a stir in the Machine Learning and Graphics communities. The paper
showed that an implicit model in the form of an MLP could be used to
convincingly represent complex 3-dimensional scenes as continuous vector
fields. We examine some of the preliminary and contemporary work in the neural
scene representation space and suggest potential future research directions to
explore.

Many of the papers listed here were suggested in the fantastic lit review by
@dellaert_neural_2021 and @lin_awesome_2021 in their Github collection.

# NeRF [@mildenhall_nerf_2020]

In this section we will go through the NeRF paper in some detail as it the
foundation on which we consider the other work.

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

# Pre-NeRF

There was a wealth of research in neural implicit surfaces that led up to the
NeRF paper that we examine in some detail here.

- Occupancy Networks [@mescheder_occupancy_2019]
  - Introduced the idea of an implicit model that learned a decision boundary
    that could be used to classify whether points were in the object or not.
  - The model was trained on a dataset of 3D scenes unlike NeRF, which is
    trained on images of 3D scenes.
  - The authors also train a VAE to learn a latent space for the 3D shapes that
    can be used to interpolate between the objects.
- Im-Net [@chen_learning_2019]
  - Another work that trains an implicit network as a binary classifier that
    takes a feature vector as input along with a point coordinate and predicts
    whether the point is in the shape.
  - The feature vector is extracted by a shape encoder to represent the shape
    and can be interpolated to get intermediary shapes.
  - This model is also trained on 3D data.
- DeepSDF [@park_deepsdf_2019]
  - This is the true parent of NeRF. The method trains a deep model to learn a
    deep signed distance function that takes a point as input and returns a
    scalar that represents the distance of the point from the boundary of the
    shape.
  - They also have a latent code to represent the shape.
  - This model is also trained on 3D data.
- PIFu [@saito_pifu_2019]
  - Learns a 3D implicit model for human forms from images.
  - Uses a fully convolutional image encoder to get spatial features for every
    3D point that can be used to see if a point is inside the surface.
  - It can also be used to learn RGB color information to perform texture
    painting.
  - They use a reconstruction loss and compare to a 3D human surfaces.

Beyond these initial forays into neural implicit surfaces that were still
trained with 3D data, there was some work on generalizing to training from 2D
images as well as work that built on top of the implicit functions to make them
more efficient and more accurate.

- CvxNet [@deng_cvxnet_2020]
  - The authors proposed a technique to learn a convex decomposition of
    topology.
  - Any topology can be modeled using a point-wise maximum of a union of convex
    meshes.
  - The convex topologies can then be used to model physics since many rigid
    body physics simulations require convex meshes.
  - The encoder network predicts the parameters for a convex polytope that can
    then be used to train the model.
  - The model is still trained on 3D data using a reconstruction loss along
    with other auxiliary losses to improve convergence.
- BSP-NET [@chen_bsp-net_2020]
  - Very similar to CvxNet in that it predicts a series of convex shapes to
    represent the object.
  - In contrast they use a binary space partition tree to create the convexes
    that ends up being much more efficient.
- Deep Local Shapes [@chabra_deep_2020]
  - Stores the DeepSDF weights in a voxel grid so it can represent large scenes
    more efficiently.
- Scene Representation Networks [@sitzmann_scene_2019]
  - Differentiable ray marching that can be used to learn from images.
- Differentiable Volumetric Rendering [@niemeyer_differentiable_2020]
  - Uses a differentiable renderer so it can be trained from images.
  - They do not integrate over volume like NeRF though.
- Implicit Differentiable Renderer [@yariv_universal_2020]
  - Similar to the above two, but the model proposed can be used to refine the
    camera pose during training. Essentially they are able to learn the camera
    details from the images in a differentiable way.
- Neural Articulated Shape Approximation [@deng_nasa_2020]
  - Learns implicit models for deformable bodies.
  - Takes as input coordinate frames (bones) and learns an occupancy function
    that is a mixture of rigid compositions and a pose layer that allows for
    non-rigid deformations.
  - They get the deformations by convolving with a gaussian kernel

A contemporary of NeRF was Neural Volume Rendering (Lombardi et al., 2019) that
was very similar to NeRF but still learned a 3D volume of density and color as
a voxel-based representation.

# Post-NeRF

After NeRF was able to create such spectacularly convincing results, there was
a mass of excitement in the field. Since then several new papers have been
released extending, improving and analyzing NeRF. We consider some of them in
this section.

## Analysis

- Fourier Features [@tancik_fourier_2020]

## Performance

- Neural sparse voxel fields [@liu_neural_2020]
- NeRF++: Analyzing and Improving Neural Radiance Fields [@zhang_nerf_2020]
- Derf: Decomposed radiance fields [@rebain_derf_2021]
- Autoint: Automatic integration for fast neural volume rendering [@lindell_autoint_2021]
- Learned initializations for optimizing coordinate-based neural representations [@tancik_learned_2021]
- JaxNeRF [@noauthor_google-researchjaxnerf_nodate]
- NeRF-pytorch [@lin_nerf-pytorch_2021]

## Dynamic Scenes

- Deformable neural radiance fields [@park_deformable_2020]


## Other Relevant Papers

- Structure-from-motion revisited [@schonberger_structure--motion_2016]
- Nerf in the wild: Neural radiance fields for unconstrained photo collections [@martin-brualla_nerf_2021]
- Self-Calibrating Neural Radiance Fields [@jeong_self-calibrating_2021]
- CodeNeRF: Disentangled Neural Radiance Fields for Object Categories [@jang_codenerf_2021]
- SIMONe: View-Invariant, Temporally-Abstracted Object Representations via Unsupervised Video Decomposition [@kabra_simone_2021]
- NeRF-VAE: A Geometry Aware 3D Scene Generative Model [@kosiorek_nerf-vae_2021]
- Animatable Neural Radiance Fields for Human Body Modeling [@peng_animatable_2021]

# Potential Directions

While the field of implicit neural scene representations is relatively new, as
you can see from the long and yet incomplete list of references there has
already been a lot of work in this area. The prospect of attaining allocentric
continuous 3D representations of the world through egocentric 2D images is
tantalizing and offers several interesting directions. There are still several
references that I have not had time to go through, and as a result it is
possible that several of the ideas in this section may already have been
explored. However, based on the literature review of the references I have
provided above here are some suggestions for potential directions.

## Exploiting NeRF-techniques for simulation and control.

NeRF is a fascinating tool for rendering the real world. However, most of the
work has been focused on capturing the real world exclusively through images or
video. One project idea is to look at potential ways to capture dynamics of the
world through images and interaction. Rather than just represent the world as
it is, we may be able to represent the world along with dynamics of how the
world might look if interacted with. We could run these experiments in 3D
interaction simulators like Unity Engine.

## Uncertainty NeRF

As of right now, all the NeRF models still aren't perfect. It would be
extremely useful to have a NeRF model that could generate uncertainty estimates
for it's volume density and color predictions.

## End-to-End Differentiable Pose Estimation

While [@jeong_self-calibrating_2021] has been able to have an end-to-end model
that is able to learn intrinsic, extrinsic, and distortion paramaters from
images of the scene, COLMAP [@schonberger_structure--motion_2016] is still used
to find the poses of the camera. Potentially, we could learn the camera poses
using a differentiable method. This would make the entire NeRF generation
process differentiable and end-to-end learnable. We could also try to leverage
any other meta-data we may have like accelerometer kinematic information to get
better pose estimation.

## NeRF meta-learning

[@tancik_learned_2021] have shown that using meta-learning initializations like
MAML and Reptile can have an enormous impact on the convergence rate of NeRF.
It is possible that other meta-learning techniques such as finding latent
structure in the weights of the MLP or learning end-to-end trainable functions
that are capable of adjusting the weights of the MLP for novel scenes might
lead to even more significant results. Furthermore, it may be possible to learn
homeomorphisms between different objects. It might be interesting to study the
algebraic topology of an object through the lens of the weights of the network.

## Using auxiliary architectures with NeRF

While NeRF does a great job of capturing the 3-dimensional richness of a scene,
the architecture used to train the model is very simple. It would be
interesting to see what would happen if we were to use an auxiliary
architecture like a Transformer or a CNN along with the MLP model to make the
model more performant.

## Exploring some of the Theoretical aspects of NeRF further

[@tancik_fourier_2020] has already done some theoretical work to show how the
NeRF NLP acts as an NTK and how using randomized Fourier Features can improve
the NTK convergence for the high-frequency components. We could extend some of
this work and see if there is a way we may be able use techniques such as
sketching to improve the convergence rate even more.


# References


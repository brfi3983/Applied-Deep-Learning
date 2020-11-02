<!-- https://www.codecogs.com/latex/eqneditor.php -->
# Topology of Deep Neural Networks (2020) - Paper Review

## Introduction
This report will go over the [paper](https://arxiv.org/pdf/2004.06093.pdf) by Gregory Naitzat, Andrey Zhitnikov, and Lek-Heng Lim. In general they used tools to evaluate how the topology of the input data changes as it passes through a deep neural network. To do this, they used Betti numbers, Pesistent Homology, and Simplical Complexes to evaluate the topology at each given layer given various architectures. Additionally, they also corroborated their findings from simulated data with real-world datasets.
## Theoretical Background
In this section, I will not go over in strict detail of all the concepts as they were out of the scope of my knowledge; however, I will cover the most important aspects that layed the foundation of this paper.
### Betti Numbers
Betti numbers are used to describe the topology of a space. In general, it counts the number of "holes." Now, a 2D hole could simply be a circle lacking space. If you extend this to 3D, it becomes a void. That is, think of a hollow sphere. Although we can not visualize this, if you extend it to n-dimensions, we repeat the same process.

In general, given a topological space $M \subseteq \mathbb{R}^d$, we can say $\beta_0(M)$ counts the number of connection components in $M$ and $\beta_k(M), k \ge 1$ counts the number of k-dimensional holes in $M$. And so, we can define the set
```math
\beta(M) = \{\beta_0(M), \beta_1(M),~\cdots , \beta_d(M)\}
```
to be the overall topological description of a space $M \subseteq \mathbb{R}^d$.

As this is a set and not a scaler, we can define a metric, $w(M)$, that captures the **topological complexity** of the space, to be the sum of all the elements in $\beta(M)$. Formally we say
```math
w(M):= \beta_0(M) + \beta_1(M) + \cdots + \beta_d(M)
```
This is similar to the Euler characteristic but is XXX.

Notice, the higher the number of Betti numbers, the more "complex" the topology of the space is. Thus, a common theme in this paper is to reduce the complexity of the topological space so that our network can separate the different classes easily. We will be monitoring this metric with different *activation functions, network widths, and network depths.* More importantly, this will be evaluated for each layer, $v_k(M), k = 1, \cdots , l-1 $ the data is passed through and so we track
```math
\beta(M) \rightarrow \beta_0(M) \rightarrow \beta_1(M) \rightarrow ~\cdots \rightarrow \beta_d(M)
```
However, in the world of modern deep learning, problems arise with using high-dimensional data with unknown topoogy. As you will see, we will be estimating the Betti numbers by sampling from a point cloud and using persistent homology to estimate the topology of our original space $M$.
### Simplical Complexes
Simplical Complexes will play an extremely important role in this paper as it allows us to construct a geometric or abstract interpretation from our point cloud so that we can determine the topology as it passes through our network.

**Simplex:** A $k$-dimensional simplex, or $k$-simplex, $\sigma$ in $\mathbb{R}^d$ is the convex hull of $k+1$ affinly independent points $v_0, v_1,~\cdots, v_k \in \mathbb{R}^d$, where $\sigma = \left[ v_0, v_1, \cdots, v_k \right]$.
XXX
Additionally, we have a definition for the **Faces** of a simplex. That is

faces($\sigma$): Simplices of dimensions $0$ to $k-1$ formed by convex hulls of proper subsets of its vertex set $\{v_0, v_1, \cdots, v_k\}$.
YYY
Given these definitions, we can define a **Simplical Complex** to be a $m$-dimensional geometrical complex $K$ in $\mathbb{R}^d$ to be a finite collection of simplices in $\mathbb{R}^d$ of dimensions at most $m$ that satisfies
1) intersections along faces exist
2) include all faces

Abstract simplical complex?

### Persistent Homology

## Results
### Simulated vs Realistic Data
The *main issue with using this process to analyze neural networks given a certain dataset is the computational task.* That is, real-world datasets have an unknown topology, and thus requires exploration of a scale for $\varepsilon$ to be found via persistent homology. And so, computational needs are dramatically increased for real-world datasets.

Additionally, with simulated data, we have clean data that does not need de-noising while realistic data requires this step. Lastly, this paper was done for "well-trained" neural networks which is unrealistic sometimes as the dataset may be sufficiently complicated and have a generalization error greater than what this method allows.
### Simulated


<!-- ### Homeomorphisms
A function $f$, is said to be a homeomorphism if it satisfies
f is bijective
f is continuous
f^-1 is continuous -->

![eqn](https://latex.codecogs.com/gif.latex?%5Cint_1%5E2%20x%5E2%20%5C%2C%20dx%20%3D%203)
### Feature Extraction in Images: Technical Details, Mathematical Formulations, and Advanced Architectures

---

#### **1. Definition of Feature Extraction**
**Concept**: Feature extraction transforms raw image data into a lower-dimensional, discriminative representation preserving semantically meaningful patterns. Formally, given an image $I \in \mathbb{R}^{H \times W \times C}$ (height $H$, width $W$, channels $C$), feature extraction learns a mapping $f: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^D$, where $D \ll H \times W \times C$.

**Mathematical Goal**: Minimize information loss while maximizing class separability. For a dataset $\mathcal{D} = \{(I_i, y_i)\}$, optimize:
$$
\min_{f} \sum_{i} \mathcal{L}\left(y_i, g(f(I_i))\right),
$$
where $g$ is a classifier and $\mathcal{L}$ is a loss function.

---

#### **2. Traditional Techniques (Pre-DL)**

##### **2.1 Edge Detection (Sobel, Canny)**
**Concept**: Detect intensity discontinuities using gradient operators.  
**Sobel Operator**:  
Horizontal and vertical kernels:  
$$
G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad  
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}.
$$  
Gradient magnitude:  
$$
|\nabla I| = \sqrt{(I * G_x)^2 + (I * G_y)^2}.
$$

##### **2.2 Scale-Invariant Feature Transform (SIFT)**
**Key Steps**:  
1. **Scale-space extrema detection** using Difference of Gaussians (DoG):  
$$
D(x,y,\sigma) = (G(x,y,k\sigma) - G(x,y,\sigma)) * I(x,y),
$$
   where $G$ is a Gaussian kernel.  
2. **Keypoint orientation** via gradient histogram.  
3. **Descriptor generation** using 128D histogram of oriented gradients.

**Invariance**: Robust to rotation, scale, and affine transformations.

---

#### **3. Deep Learning-Based Techniques**

##### **3.1 Convolutional Neural Networks (CNNs)**
**Core Operations**:  
- **Convolution**: For input $I$, kernel $K \in \mathbb{R}^{k \times k}$, output feature map $F$:  
$$
F(i,j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i+m, j+n) \cdot K(m,n) + b,
$$
  where $b$ is bias.  
- **Activation (ReLU)**:  
$$
f(x) = \max(0, x).
$$
- **Pooling (Max)**:  
$$
F(i,j) = \max_{m,n \in \Omega} I(i+m, j+n),
$$
  where $\Omega$ is a local window (e.g., $2\times 2$).

**Hierarchy**:  
- Early layers detect edges/textures (Gabor-like filters).  
- Deeper layers capture parts/objects (hierarchical compositionality).

---

##### **3.2 Residual Networks (ResNet)**
**Residual Block**: Solves vanishing gradients by learning residual functions:  
$$
y = \mathcal{F}(x, \{W_i\}) + x,
$$
where $\mathcal{F}$ is a stack of convolutions, and $x$ is the identity shortcut.  

**Backpropagation**: Gradient flows through both $\mathcal{F}$ and $x$:  
$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left(1 + \frac{\partial \mathcal{F}}{\partial x}\right).
$$

**Analogy**: Highway networks where gradients can "skip" layers.

---

##### **3.3 Inception Networks (GoogLeNet)**
**Inception Module**: Parallel multi-scale convolutions:  
1. $1\times 1$ conv for dimensionality reduction.  
2. $3\times 3$, $5\times 5$ convs for spatial patterns.  
3. Max-pooling for robustness.  

**Concatenation**: Outputs merged along channel dimension:  
$$
F_{\text{out}} = \text{Concat}(F_{1\times1}, F_{3\times3}, F_{5\times5}, F_{\text{pool}}).
$$

**Efficiency**: Bottleneck layers (1x1 convs) reduce computational cost.

---

##### **3.4 Vision Transformers (ViT)**
**Patch Embedding**: Split image into $N$ patches $\{p_i\}$, linearly project to $D$ dimensions:  
$$
z_0 = [p_1\mathbf{E}; p_2\mathbf{E}; \dots; p_N\mathbf{E}] + \mathbf{E}_{\text{pos}},
$$
where $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the embedding matrix and $P$ is patch size.  

**Multi-Head Self-Attention (MSA)**:  
For queries $Q$, keys $K$, values $V$:  
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V,
$$
where $d_k$ is the dimension of $K$. Multi-head attention concatenates outputs from $h$ heads.  

**Analogy**: Global context aggregation via weighted sums of all patches.

---

##### **3.5 Autoencoders**
**Architecture**: Encoder $f$ compresses input to latent code $z$, decoder $g$ reconstructs it.  
**Loss**: Minimize reconstruction error:  
$$
\mathcal{L} = \|I - g(f(I))\|_2^2.
$$
**Variational Autoencoder (VAE)**: Introduces probabilistic latent space:  
$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|I)}[\log p(I|z)] - \beta \cdot D_{\text{KL}}(q(z|I) \| p(z)),
$$
where $p(z)$ is a prior (e.g., $\mathcal{N}(0,I)$).

---

#### **4. Advanced Architectures**

##### **4.1 DenseNet**
**Dense Block**: Each layer connects to all subsequent layers. For layer $l$:  
$$
x_l = H_l([x_0, x_1, \dots, x_{l-1}]),
$$
where $[·]$ denotes concatenation. Promotes feature reuse.

##### **4.2 U-Net (for Segmentation)**
**Encoder-Decoder**: Skip connections concatenate encoder features to decoder. Preserves spatial details:  
$$
F_{\text{decoder}}^{(i)} = \text{Concat}(F_{\text{encoder}}^{(i)}, \text{UpSample}(F_{\text{decoder}}^{(i+1)})).
$$

##### **4.3 Capsule Networks**
**Capsules**: Groups of neurons representing object properties. Routing-by-agreement:  
$$
c_{j|i} = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})}, \quad  
s_j = \sum_i c_{j|i} \hat{u}_{j|i},
$$
where $\hat{u}_{j|i}$ is prediction from capsule $i$ to $j$.

---

#### **5. Mathematical Comparison of Techniques**
| **Method**       | **Key Equation**                          | **Invariance**       | **Use Case**         |
|-------------------|-------------------------------------------|----------------------|----------------------|
| CNN               | $F = \sigma(K \ast I + b)$                | Translation          | General-purpose      |
| ResNet            | $y = \mathcal{F}(x) + x$                  | Depth robustness     | Very deep networks   |
| ViT               | $\text{Attention}(Q,K,V)$                 | Global context       | High-resolution      |
| Autoencoder       | $\min \|I - g(f(I))\|^2$                  | Data distribution    | Unsupervised learning |

---

#### **6. Summary**
Feature extraction leverages mathematical operations (convolution, attention, residuals) to convert raw pixels into task-specific representations. Advanced architectures address limitations like depth degradation (ResNet), scale variance (Inception), and global dependency (ViT), each with distinct mathematical formulations.
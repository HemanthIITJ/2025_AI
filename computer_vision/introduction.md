### Deep Learning for Computer Vision: Technical Deep Dive

---

#### **1. Hierarchical Feature Learning in Deep Neural Networks**
**Concept**: Deep learning models for vision tasks exploit hierarchical representations, mimicking the ventral stream in the primate visual cortex. Each layer extracts progressively complex features.  
**Mathematical Framework**:  
- Let $x \in \mathbb{R}^{H \times W \times C}$ denote an input image tensor with height $H$, width $W$, and channels $C$.  
- A convolutional layer applies $K$ filters (kernels) $\\{ \mathbf{W}_k \\}_{k=1}^K$ to $x$:  
  $$
  \mathbf{z}^{(l)}_k = \mathbf{W}_k^{(l)} * \mathbf{a}^{(l-1)} + b_k^{(l)}
  $$  
  where $*$ denotes convolution, $\mathbf{a}^{(l-1)}$ is the activation from layer $l-1$, and $b_k^{(l)}$ is the bias.  
- **Activation Function (ReLU)**:  
  $$
  \mathbf{a}^{(l)}_k = \max(0, \mathbf{z}^{(l)}_k)
  $$  
  Non-linearity introduces sparsity and enables modeling complex mappings.

---

#### **2. Convolution Operation: Mathematical Specifics**  
**Discrete 2D Convolution**:  
For a filter $\mathbf{W} \in \mathbb{R}^{f \times f \times C}$ and input patch $\mathbf{x}_{i,j} \in \mathbb{R}^{f \times f \times C}$ centered at position $(i,j)$:  
$$
(\mathbf{W} * \mathbf{x})_{i,j} = \sum_{c=1}^C \sum_{m=1}^f \sum_{n=1}^f \mathbf{W}_{m,n,c} \cdot \mathbf{x}_{i+m-\lfloor f/2 \rfloor, j+n-\lfloor f/2 \rfloor, c}
$$  
- **Stride ($s$)**: Step size for sliding the filter. Output dimension reduces as $\left\lfloor \frac{H - f}{s} + 1 \right\rfloor$.  
- **Padding**: Zero-padding preserves spatial resolution.  

---

#### **3. Pooling Layers: Dimensionality Reduction**  
**Max-Pooling**: Extracts translational invariant features by downsampling:  
$$
\mathbf{a}^{(l)}_{k,i,j} = \max_{p,q \in \mathcal{N}(i,j)} \mathbf{a}^{(l-1)}_{k,p,q}
$$  
where $\mathcal{N}(i,j)$ is a local neighborhood (e.g., $2 \times 2$ window).  

---

#### **4. Fully Connected (FC) Layers & Softmax Classifier**  
- **FC Layer**: Flatten spatial dimensions to vector $\mathbf{h} \in \mathbb{R}^D$:  
  $$
  \mathbf{h} = \mathbf{W}_{\text{fc}} \cdot \text{vec}(\mathbf{a}^{(L)}) + \mathbf{b}_{\text{fc}}
  $$  
- **Softmax**: For $C$ classes, output probability:  
  $$
  p(y=c|\mathbf{h}) = \frac{e^{\mathbf{w}_c^T \mathbf{h} + b_c}}{\sum_{j=1}^C e^{\mathbf{w}_j^T \mathbf{h} + b_j}}
  $$  

---

#### **5. Loss Function: Cross-Entropy**  
For labeled data $(x, y)$, the loss is:  
$$
\mathcal{L} = -\sum_{c=1}^C y_c \log p(y=c|\mathbf{h})
$$  
where $y_c$ is a one-hot encoded label.  

---

#### **6. Backpropagation in CNNs**  
Gradients are computed via chain rule:  
- **Convolutional Layer Gradient**:  
  $$
  \frac{\partial \mathcal{L}}{\partial \mathbf{W}_k^{(l)}} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}_{k,i,j}} \cdot \mathbf{a}^{(l-1)}_{i,j}
  $$  
- **ReLU Gradient**:  
  $$
  \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}} = \mathbb{I}(\mathbf{z}^{(l)} > 0)
  $$  

---

#### **7. Computer Vision Tasks & Architectures**  
1. **Image Classification (e.g., ResNet)**:  
   - Residual block: $\mathbf{a}^{(l+1)} = \mathcal{F}(\mathbf{a}^{(l)}) + \mathbf{a}^{(l)}$.  
2. **Object Detection (e.g., Faster R-CNN)**:  
   - Region Proposal Network (RPN): Predicts bounding boxes via:  
     $$
     t_x = (x - x_a)/w_a, \quad t_y = (y - y_a)/h_a
     $$  
   - Multi-task loss: Classification + Regression.  
3. **Semantic Segmentation (e.g., U-Net)**:  
   - Encoder-decoder architecture with skip connections.  
   - Pixel-wise cross-entropy loss:  
     $$
     \mathcal{L} = -\sum_{i=1}^{H \times W} \sum_{c=1}^C y_{i,c} \log p_{i,c}
     $$  

---

#### **8. The "Cambrian Explosion" Analogy**  
The rapid proliferation of vision architectures (2012–present) mirrors the biological Cambrian Explosion, driven by:  
- **Data**: ImageNet (14M images, 20k classes).  
- **Compute**: GPUs enabling efficient training of high-dimensional non-convex models.  
- **Algorithmic Innovations**: Batch normalization, dropout, attention mechanisms.  

---

#### **9. Camera Obscura to CNNs**  
- **Camera Obscura**: Early optical device projecting light through a pinhole. Mathematical model:  
  $$
  I(x,y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} L(u,v) \cdot \delta\left(\frac{u}{d} - x, \frac{v}{d} - y\right) du \, dv
  $$  
  where $L(u,v)$ is scene radiance and $d$ is depth.  
- **Modern Vision**: Replaces pinhole with learnable hierarchical filters (CNNs).  

---

#### **10. Optimization & Regularization**  
- **Adam Optimizer**: Adaptive learning rates with momentum:  
  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\  
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\  
  \theta_{t+1} = \theta_t - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
  $$  
- **Dropout**: Randomly deactivate neurons during training:  
  $$
  \mathbf{a}^{(l)} = \mathbf{a}^{(l)} \odot \mathbf{m}, \quad \mathbf{m} \sim \text{Bernoulli}(p)
  $$  

---

This framework underpins modern vision systems, combining biological inspiration with rigorous mathematical optimization.


### **1. Hubel and Wiesel (1959): Biological Basis of Visual Processing**

#### **Concept & Definition**
Hubel and Wiesel's experiments on cat and monkey visual cortices revealed **hierarchical processing** of visual stimuli. They identified **simple cells** (edge detectors) and **complex cells** (motion/direction detectors). This work laid the foundation for **convolutional neural networks (CNNs)**.

#### **Mathematical Framework**
- **Simple Cells**: Modeled as **Gabor filters**, which are linear operators capturing orientation and spatial frequency.  
  The 2D Gabor function is:  
  $$
  G(x, y) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cdot \exp\left(i\left(2\pi \frac{x'}{\lambda} + \psi\right)\right)
  $$  
  where:  
  - $x' = x \cos\theta + y \sin\theta$  
  - $y' = -x \sin\theta + y \cos\theta$ (rotation by orientation $\theta$)  
  - $\gamma$: spatial aspect ratio, $\sigma$: standard deviation, $\lambda$: wavelength, $\psi$: phase offset.  

- **Complex Cells**: Perform **max-pooling** over simple cell outputs:  
  $$
  C(x, y) = \max_{\Delta x, \Delta y} \left\{ S(x + \Delta x, y + \Delta y) \right\}
  $$  
  where $S$ is the simple cell response.

#### **Technical Significance**
This hierarchical structure inspired the **convolution-pooling architecture** in CNNs, formalized later by Fukushima (Neocognitron, 1980) and LeCun (LeNet, 1998).

---

### **2. Larry Roberts (1963): Early 3D Object Recognition**

#### **Concept & Definition**
Roberts pioneered **3D object recognition** using **line drawings** and **polyhedral models**. His system transformed 3D objects into 2D projections via **perspective geometry** and matched them to input edges.

#### **Mathematical Framework**
- **Perspective Projection**:  
  For a 3D point $\mathbf{P} = [X, Y, Z]^T$, its 2D projection $\mathbf{p} = [x, y]^T$ is:  
  $$
  x = \frac{fX}{Z}, \quad y = \frac{fY}{Z}
  $$  
  where $f$ is focal length.  
  Matrix form using homogeneous coordinates:  
  $$
  \begin{bmatrix}
  x \\ y \\ 1
  \end{bmatrix}
  \sim
  \begin{bmatrix}
  f & 0 & 0 & 0 \\
  0 & f & 0 & 0 \\
  0 & 0 & 1 & 0
  \end{bmatrix}
  \begin{bmatrix}
  X \\ Y \\ Z \\ 1
  \end{bmatrix}
  $$  

- **Edge Extraction**: Roberts used **gradient-based edge detection**:  
  $$
  \nabla I(x, y) = \left[\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right]^T
  $$  
  Edges identified where $\|\nabla I\| > \tau$ (threshold $\tau$).

#### **Technical Limitations**
- Assumed **polyhedral objects** with known geometry.  
- No robustness to occlusion or noise.  

---

### **3. Recognition via Parts (1970s): Structural Models**

#### **Concept & Definition**
The **"Recognition by Components"** theory (Biederman, 1987) proposed decomposing objects into **geometric primitives** (e.g., cylinders, cubes). Mathematically, this is a **graph matching problem**.

#### **Mathematical Framework**
- **Primitive Representation**: Each part $P_i$ is a parameterized shape, e.g., a cylinder:  
  $$
  P_i = \{ \text{radius}, \text{height}, \text{orientation}, \text{position} \}
  $$  
- **Spatial Relations**: Modeled as a graph $G = (V, E)$, where:  
  - $V = \{ P_1, P_2, \dots, P_n \}$ (vertices = parts)  
  - $E = \{ (P_i, P_j, \mathbf{T}_{ij}) \}$ (edges = spatial transformations between parts).  

- **Energy Minimization**: Recognition involves minimizing:  
  $$
  E(G, I) = \sum_{P_i \in V} \| \text{Observed}(P_i) - \text{Projected}(P_i) \|^2 + \sum_{(P_i, P_j) \in E} \| \mathbf{T}_{ij} - \mathbf{\hat{T}}_{ij} \|^2
  $$  
  where $\mathbf{\hat{T}}_{ij}$ is the observed spatial relation.

#### **Technical Challenges**
- **Combinatorial explosion**: Graph matching is NP-hard.  
- **Sensitivity to occlusion**: Missing parts break the graph structure.  

---

### **4. Recognition via Edge Detection (1980s): Marr-Hildreth & Canny**

#### **Concept & Definition**
Marr and Hildreth proposed **edge detection** as the first stage of vision, using **zero-crossings** of the **Laplacian of Gaussian (LoG)**. Canny later formalized optimal edge detection.

#### **Mathematical Framework**
- **Laplacian of Gaussian (LoG)**:  
  $$
  \nabla^2 G(x, y) = \frac{\partial^2 G}{\partial x^2} + \frac{\partial^2 G}{\partial y^2}
  $$  
  where $G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$.  

- **Canny Edge Detector**:  
  1. **Gradient Computation**:  
     $$
     \|\nabla I\| = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2}, \quad \theta = \arctan\left(\frac{\partial I}{\partial y} / \frac{\partial I}{\partial x}\right)
     $$  
  2. **Non-Maximum Suppression**: Thin edges by retaining only local maxima in $\|\nabla I\|$ along $\theta$.  
  3. **Hysteresis Thresholding**: Use dual thresholds ($\tau_{\text{high}}$, $\tau_{\text{low}}$) to link edges.  

#### **Technical Limitations**
- **Scale sensitivity**: LoG requires tuning $\sigma$.  
- **Fragility**: Edges break under noise or texture.  

---

### **5. Arriving at the "AI Winter": Technical Shortcomings**

#### **Key Causes**
1. **Combinatorial Complexity**:  
   - Graph matching for recognition has time complexity $O(n!)$ for $n$ parts.  
   - NP-hard problems became intractable with 1980s hardware.  

2. **Lack of Robustness**:  
   - Edge detection failed under noise:  
     SNR = $\frac{\|\nabla I\|_{\text{signal}}}{\|\nabla I\|_{\text{noise}}}$ often $< 1$.  
   - No statistical learning frameworks to handle variability.  

3. **Limited Data & Compute**:  
   - Training structural models required labeled data, which was scarce.  
   - No GPUs: A 1980s VAX-11/780 ran at 5 MHz (vs. modern GPUs at 1-2 GHz).  

#### **Mathematical "Killer Problems"**
- **Perceptron Limitations** (Minsky & Papert, 1969): Linear classifiers fail on XOR.  
  $$
  \nexists \mathbf{w} \text{ such that } \mathbf{w}^T \mathbf{x} > 0 \iff x_1 \oplus x_2 = 1
  $$  
- **Curse of Dimensionality**: Recognition in high-dimensional spaces ($\mathbb{R}^{N}$) required exponential samples.  

---

### **Conclusion**  
The AI winter emerged from **fundamental gaps** in computational theory (NP-hardness), hardware (limited FLOPS), and statistical learning (no backpropagation). These were later addressed by CNNs (exploiting Hubel-Wiesel hierarchy), GPUs, and large datasets (ImageNet).

### **1. Rapid Serial Visual Perception (RSVP): Temporal Dynamics of Recognition**

#### **Concept & Definition**
RSVP refers to the rapid presentation of visual stimuli (e.g., images at 10 Hz) to study **temporal limits** of human recognition. It reveals **attentional bottlenecks** (e.g., the "attentional blink") and neural mechanisms of **temporal integration**.

#### **Mathematical Framework**
- **Stimulus Presentation**: Let $S(t)$ denote the stimulus sequence:  
  $$
  S(t) = \sum_{k=1}^N \delta(t - k\Delta t)
  $$  
  where $\Delta t$ is the inter-stimulus interval (e.g., 100 ms) and $\delta$ is the Dirac delta.  

- **Neural Response**: Modeled as a **temporal convolution** of stimuli with a neural kernel $\kappa(t)$:  
  $$
  R(t) = \int_{-\infty}^\infty S(\tau) \cdot \kappa(t - \tau) \, d\tau
  $$  
  The kernel $\kappa(t)$ often includes excitatory/inhibitory phases:  
  $$
  \kappa(t) = A \cdot e^{-t/\tau_e} - B \cdot e^{-t/\tau_i} \quad (t \geq 0)
  $$  
  where $\tau_e$, $\tau_i$ are time constants for excitation/inhibition.  

- **Attentional Blink**: The probability of missing a target stimulus at lag $l$ follows:  
  $$
  P_{\text{miss}}(l) = 1 - \exp\left(-\frac{(l - t_0)^2}{2\sigma^2}\right)
  $$  
  where $t_0$ is the blink onset (~200 ms) and $\sigma$ (~50 ms) is its duration.  

#### **Technical Significance**
RSVP quantifies the **temporal resolution** of visual recognition (~10 Hz), influencing real-time vision systems like **event cameras**.

---

### **2. Neural Correlates of Object & Scene Recognition**

#### **Concept & Definition**
Object recognition involves hierarchical processing in the ventral visual stream (V1 → V2 → V4 → IT). **Scene recognition** engages the **parahippocampal place area (PPA)** and **retrosplenial cortex (RSC)**. Neural activity is measured via **fMRI** (blood oxygenation) or **single-unit recordings**.

#### **Mathematical Framework**
- **Population Coding**: Let $\mathbf{r} \in \mathbb{R}^N$ be the firing rates of $N$ neurons. The likelihood of a stimulus $s$ is:  
  $$
  P(\mathbf{r} | s) = \prod_{i=1}^N \frac{(f_i(s)\Delta t)^{r_i}}{r_i!} e^{-f_i(s)\Delta t}
  $$  
  where $f_i(s)$ is the tuning curve of neuron $i$.  

- **Decoding via Maximum Likelihood**:  
  $$
  \hat{s} = \arg\max_s \sum_{i=1}^N r_i \log f_i(s) - f_i(s)\Delta t
  $$  

- **fMRI BOLD Signal**: The hemodynamic response $h(t)$ is modeled as:  
  $$
  h(t) = \left( \frac{t}{\tau} \right)^\alpha e^{-(t - \tau)/\beta} \quad (t \geq 0)
  $$  
  where $\tau$, $\alpha$, $\beta$ are fitted parameters.  

#### **Technical Challenges**
- **Noise Floor**: fMRI has low temporal resolution (~2 sec) vs. neural dynamics (~10 ms).  
- **Curse of Dimensionality**: Decoding from $N$ neurons requires $O(e^N)$ samples.  

---

### **3. Recognition via Grouping (1990s): Gestalt Principles Formalized**

#### **Concept & Definition**
Grouping uses **Gestalt principles** (proximity, similarity, continuity) to segment objects from backgrounds. Algorithms like **normalized cuts** and **spectral clustering** emerged.

#### **Mathematical Framework**
- **Affinity Matrix**: For an image with $n$ pixels, define $W \in \mathbb{R}^{n \times n}$ where:  
  $$
  W_{ij} = \exp\left(-\frac{\| \mathbf{f}_i - \mathbf{f}_j \|^2}{2\sigma^2}\right) \cdot \exp\left(-\frac{\| \mathbf{x}_i - \mathbf{x}_j \|^2}{2\eta^2}\right)
  $$  
  $\mathbf{f}_i$: feature vector (color, texture), $\mathbf{x}_i$: spatial position.  

- **Normalized Cut**: Partition graph $G=(V, E)$ into $A$ and $B$ by minimizing:  
  $$
  \text{Ncut}(A, B) = \frac{\text{cut}(A, B)}{\text{assoc}(A, V)} + \frac{\text{cut}(A, B)}{\text{assoc}(B, V)}
  $$  
  where $\text{cut}(A, B) = \sum_{i \in A, j \in B} W_{ij}$, $\text{assoc}(A, V) = \sum_{i \in A, j \in V} W_{ij}$.  

- **Spectral Clustering**: Solve the generalized eigenvalue problem:  
  $$
  (D - W) \mathbf{v} = \lambda D \mathbf{v}
  $$  
  where $D$ is the degree matrix ($D_{ii} = \sum_j W_{ij}$). The second smallest eigenvector partitions the graph.  

#### **Technical Limitations**
- **O(n³) Complexity**: Eigen decomposition is intractable for large $n$.  
- **Parameter Sensitivity**: $\sigma$, $\eta$ heavily affect results.  

---

### **4. Recognition via Matching (2000s): SIFT & Bag-of-Words**

#### **Concept & Definition**
Matching-based recognition uses **local invariant features** (e.g., SIFT, SURF) and **bag-of-words (BoW)** models. This dominated pre-deep-learning eras.

#### **Mathematical Framework**
- **SIFT Descriptor**: For a keypoint at $(x, y)$, compute gradient magnitudes $m(x,y)$ and orientations $\theta(x,y)$ in a 16×16 patch. Bin orientations into 8 bins per 4×4 subregion, yielding a 128-D vector.  

- **Scale-Invariant Detection**:  
  The Difference of Gaussian (DoG) pyramid:  
  $$
  D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)
  $$  
  Keypoints are local maxima/minima in $D(x, y, \sigma)$ across scales.  

- **Bag-of-Words**:  
  1. **Codebook Learning**: Cluster SIFT descriptors into $K$ centroids using k-means:  
     $$
     \min_{\mathbf{c}_1, \dots, \mathbf{c}_K} \sum_{i=1}^N \min_k \| \mathbf{f}_i - \mathbf{c}_k \|^2
     $$  
  2. **Image Representation**: Histogram $h$ of codeword frequencies:  
     $$
     h_k = \sum_{i=1}^M \delta(\text{NN}(\mathbf{f}_i) = k)
     $$  
     where $\text{NN}(\mathbf{f}_i)$ is the nearest centroid to $\mathbf{f}_i$.  

- **Spatial Pyramid Matching**: Add spatial bins to BoW:  
  $$
  \text{Sim}(I_1, I_2) = \sum_{l=0}^L \frac{1}{2^{L-l}} \sum_{m=1}^{4^l} \chi^2(h_{1,l,m}, h_{2,l,m})
  $$  
  where $\chi^2$ is the chi-squared distance.  

#### **Technical Challenges**
- **Semantic Gap**: BoW lacks part relationships.  
- **Viewpoint Sensitivity**: SIFT fails under extreme affine distortion.  

---

### **Key Transitions & Limitations**  
- **Grouping → Matching**: Grouping struggled with **occlusion**; matching lacked **semantic coherence**.  
- **AI Winter Exit**: The 2010s saw **CNNs** (e.g., AlexNet) unify hierarchical feature learning (Hubel-Wiesel) and invariance (SIFT) via backpropagation:  
$$
  \frac{\partial \mathcal{L}}{\partial W_{ij}^{(l)}} = \sum_{k} \frac{\partial \mathcal{L}}{\partial z_k^{(l+1)}} \cdot a_j^{(l)} \cdot \sigma'(z_i^{(l)})
$$  
where $z^{(l)}$, $a^{(l)}$ are pre-/post-activations at layer $l$.  

- **Invariance Theory**: CNNs approximate **Lie group transformations** (translation, rotation) via weight sharing:  
  $$
  f(g \cdot x) \approx g \cdot f(x)
  $$  
  where $g$ is a group element.  

This progression resolved the AI winter by addressing **temporal** (RSVP), **neural** (population coding), and **algorithmic** (BoW → CNN) limitations.


### **1. Face Detection: Viola-Jones Framework (2001)**  
#### **Concept & Definition**  
Face detection involves locating faces in images via **Haar-like features**, **AdaBoost**, and **cascade classifiers**. The Viola-Jones algorithm introduced **integral images** for real-time computation.  

#### **Mathematical Framework**  
- **Integral Image**: For an image $I(x,y)$, the integral image $II(x,y)$ is:  
  $$  
  II(x, y) = \sum_{x' \leq x, y' \leq y} I(x', y')  
  $$  
  Enables rapid computation of rectangular features (e.g., edge, line, center-surround).  

- **Haar-like Features**: A feature $f_j$ is a weighted sum of pixels:  
  $$  
  f_j = \sum_{(x,y) \in R_1} w_1 I(x,y) + \sum_{(x,y) \in R_2} w_2 I(x,y)  
  $$  
  where $R_1$, $R_2$ are adjacent rectangles, $w_1 = +1$, $w_2 = -1$.  

- **AdaBoost Training**: Selects weak classifiers $h_t(x) \in \{-1, +1\}$ to minimize error $\epsilon_t$:  
  $$  
  \epsilon_t = \sum_{i=1}^N D_t(i) \cdot \mathbb{I}[h_t(x_i) \neq y_i]  
  $$  
  where $D_t(i)$ is the sample weight distribution at iteration $t$. Final strong classifier:  
  $$  
  H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right), \quad \alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)  
  $$  

- **Cascade Classifier**: A degenerate decision tree with stages $s=1,2,...,S$:  
  $$  
  \text{Accept iff } \prod_{s=1}^S \mathbb{I}\left[\sum_{t=1}^{T_s} \alpha_{s,t} h_{s,t}(x) \geq \tau_s\right] = 1  
  $$  
  Early stages reject ~50% of non-faces with minimal computation.  

#### **Technical Limitations**  
- **Limited Invariance**: Fails under extreme pose/lighting due to rigid Haar templates.  
- **Manual Feature Engineering**: No learning of optimal features.  

---

### **2. PASCAL Visual Object Challenge (2005–2012): Benchmarking Detection**  
#### **Concept & Definition**  
PASCAL VOC standardized object detection metrics: **mean Average Precision (mAP)**. Tasks included classification, detection, segmentation.  

#### **Mathematical Framework**  
- **Precision-Recall Curve**: For class $c$, sort detections by confidence. Compute:  
  $$  
  \text{Precision}(k) = \frac{\text{TP}(k)}{\text{TP}(k) + \text{FP}(k)}, \quad \text{Recall}(k) = \frac{\text{TP}(k)}{N_{\text{gt}}}  
  $$  
  where $k$ is the $k$-th detection, $N_{\text{gt}}$ is ground-truth count.  

- **Average Precision (AP)**: Interpolated area under the PR curve:  
  $$  
  AP_c = \frac{1}{11} \sum_{r \in \{0,0.1,...,1\}} \max_{\tilde{r} \geq r} \text{Precision}(\tilde{r})  
  $$  
  mAP averages $AP_c$ over all classes.  

- **Intersection-over-Union (IoU)**: Localization accuracy:  
  $$  
  \text{IoU} = \frac{B_{\text{pred}} \cap B_{\text{gt}}}{B_{\text{pred}} \cup B_{\text{gt}}}  
  $$  
  A detection is a TP if $\text{IoU} \geq 0.5$.  

#### **Technical Impact**  
- Drove adoption of **sliding window** + **HOG/SVM** (Dalal & Triggs, 2005) and later **Faster R-CNN**.  

---

### **3. Minsky and Papert (1969): Perceptron Limitations**  
#### **Concept & Definition**  
Proved that single-layer perceptrons cannot solve non-linearly separable tasks (e.g., XOR). Led to the first **AI winter** by exposing theoretical limits.  

#### **Mathematical Framework**  
- **Perceptron Update Rule**: For input $\mathbf{x}$, weights $\mathbf{w}$, output $y = \text{sign}(\mathbf{w}^T \mathbf{x})$:  
  $$  
  \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta (y_{\text{true}} - y_{\text{pred}}) \mathbf{x}  
  $$  

- **XOR Proof**: Define XOR as $f(0,0)=0$, $f(1,1)=0$, $f(0,1)=1$, $f(1,0)=1$. Assume weights $w_1, w_2$, bias $b$:  
  $$  
  \begin{cases}  
  w_1 \cdot 0 + w_2 \cdot 0 + b \leq 0 \\  
  w_1 \cdot 1 + w_2 \cdot 1 + b \leq 0 \\  
  w_1 \cdot 0 + w_2 \cdot 1 + b > 0 \\  
  w_1 \cdot 1 + w_2 \cdot 0 + b > 0  
  \end{cases}  
  $$  
  Adding first two inequalities: $2(w_1 + w_2) + 2b \leq 0$ → $w_1 + w_2 + b \leq 0$.  
  Adding last two: $w_1 + w_2 + 2b > 0$. Contradiction.  

#### **Technical Significance**  
- Motivated multi-layer networks (MLPs) and backpropagation.  

---

### **4. Neocognitron (Fukushima, 1980): CNN Precursor**  
#### **Concept & Definition**  
Hierarchical model with **S-cells** (simple, feature detectors) and **C-cells** (complex, invariance to shifts). Directly inspired by Hubel-Wiesel.  

#### **Mathematical Framework**  
- **S-cell Activation**: For layer $l$, cell $i$, input from layer $l-1$:  
  $$  
  s_i^{(l)} = \varphi\left(\sum_j w_{ij}^{(l)} c_j^{(l-1)} - \theta_s^{(l)}\right)  
  $$  
  where $\varphi$ is a threshold function (e.g., $\varphi(x) = \max(0, x)$), $\theta_s$ is a threshold.  

- **C-cell Activation**: Max-pooling over S-cells:  
  $$  
  c_k^{(l)} = \max_{i \in \mathcal{R}_k} s_i^{(l)}  
  $$  
  $\mathcal{R}_k$ is a local receptive field.  

- **Unsupervised Learning**: Weights $w_{ij}$ updated via competitive learning:  
  $$  
  \Delta w_{ij} \propto c_j^{(l-1)} \cdot (s_i^{(l)} - w_{ij})  
  $$  

#### **Technical Limitations**  
- No end-to-end training (pre-dated backprop).  
- Handcrafted receptive field sizes.  

---

### **5. Backpropagation (Rumelhart, Hinton, Williams, 1986)**  
#### **Concept & Definition**  
Algorithm for computing gradients in neural networks via **chain rule**, enabling training of multi-layer networks.  

#### **Mathematical Framework**  
- **Loss Function**: For output $\mathbf{y}$, target $\mathbf{t}$, e.g., MSE:  
  $$  
  \mathcal{L} = \frac{1}{2} \sum_{i=1}^n (y_i - t_i)^2  
  $$  

- **Weight Update**: For weight $w_{jk}^{(l)}$ between neuron $k$ in layer $l-1$ and neuron $j$ in layer $l$:  
  $$  
  \Delta w_{jk}^{(l)} = -\eta \frac{\partial \mathcal{L}}{\partial w_{jk}^{(l)}}  
  $$  
  where $\eta$ is the learning rate.  

- **Chain Rule**: Compute gradients recursively:  
  1. Output layer ($L$):  
     $$  
     \delta_j^{(L)} = (y_j - t_j) \cdot \varphi'\left(z_j^{(L)}\right)  
     $$  
  2. Hidden layer ($l$):  
     $$  
     \delta_j^{(l)} = \varphi'\left(z_j^{(l)}\right) \sum_k \delta_k^{(l+1)} w_{kj}^{(l+1)}  
     $$  
  3. Gradient for $w_{jk}^{(l)}$:  
     $$  
     \frac{\partial \mathcal{L}}{\partial w_{jk}^{(l)}} = \delta_j^{(l)} \cdot a_k^{(l-1)}  
     $$  
  where $z_j^{(l)} = \sum_k w_{jk}^{(l)} a_k^{(l-1)}$, $a_j^{(l)} = \varphi(z_j^{(l)})$.  

#### **Technical Significance**  
- Enabled deep networks but initially limited by **vanishing gradients** (saturating activations like sigmoid).  

---

### **Key Connections**  
- **Neocognitron → CNN**: Fukushima’s S/C-cells became convolutional/pooling layers.  
- **Backprop + CNNs**: LeCun (1989) combined backprop with convolutional weight sharing:  
  $$  
  \frac{\partial \mathcal{L}}{\partial w_{pq}} = \sum_{i,j} \delta_{ij}^{(l)} \cdot a_{i+p, j+q}^{(l-1)}  
  $$  
- **Viola-Jones → Deep Learning**: Cascades inspired **attention mechanisms**; AdaBoost influenced **ensemble methods**.  

Theoretical foundations (Minsky), architectures (Neocognitron), and algorithms (backprop) coalesced in the 2010s to end the AI winter via **deep learning**.

### **1. Convolutional Networks: LeCun et al., 1998 (LeNet-5)**  
#### **Concept & Definition**  
LeNet-5 was the first practical convolutional neural network (CNN), designed for handwritten digit recognition. It introduced **convolutional layers**, **subsampling (pooling)**, and **hierarchical feature learning**, formalizing Hubel-Wiesel’s biological insights into a computational framework.  

---

#### **Mathematical Framework**  
1. **Convolutional Layer**:  
   - Let input be a 2D map $I \in \mathbb{R}^{H \times W}$, kernel $K \in \mathbb{R}^{k \times k}$, stride $s$, and bias $b$.  
   - Output feature map $F$ is computed as:  
     $$  
     F(x, y) = \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} I(x \cdot s + i, y \cdot s + j) \cdot K(i, j) + b  
     $$  
     LeNet-5 used $k=5$, $s=1$, and 6 kernels in the first layer.  

   - **Weight Sharing**: Kernels are replicated across spatial locations, reducing parameters vs. fully connected layers.  

2. **Subsampling (Pooling) Layer**:  
   - Average pooling with $2 \times 2$ regions, stride 2:  
     $$  
     P(x, y) = \frac{1}{4} \sum_{i=0}^1 \sum_{j=0}^1 F(2x + i, 2y + j)  
     $$  
   - Followed by a sigmoid activation:  
     $$  
     A(x, y) = \frac{1}{1 + e^{-P(x, y)}}  
     $$  

3. **Fully Connected Layers**:  
   - Final layers flatten pooled features into a vector $\mathbf{z} \in \mathbb{R}^d$ and compute:  
     $$  
     \mathbf{h} = \sigma(W \mathbf{z} + \mathbf{b})  
     $$  
     where $\sigma$ is sigmoid, $W \in \mathbb{R}^{m \times d}$, and $m=120$ in LeNet-5.  

4. **Loss Function**:  
   - **Cross-Entropy Loss** for digit classification (10 classes):  
     $$  
     \mathcal{L} = -\sum_{c=1}^{10} t_c \log(p_c)  
     $$  
     where $t_c$ is the target label, and $p_c = \text{softmax}(h_c)$.  

---

#### **Technical Innovations**  
- **Gradient-Based Learning**: Backpropagation through convolutional layers via **chain rule**:  
  $$  
  \frac{\partial \mathcal{L}}{\partial K(i,j)} = \sum_{x,y} \frac{\partial \mathcal{L}}{\partial F(x,y)} \cdot I(x+i, y+j)  
  $$  
- **Parameter Efficiency**: Only 60k parameters (vs. millions in MLPs).  

---

### **2. 2000s: The Emergence of “Deep Learning”**  
#### **Concept & Definition**  
The term “deep learning” was coined to describe neural networks with **>3 hidden layers**, enabled by **unsupervised pre-training** (e.g., RBMs) and **computational advances** (GPUs).  

---

#### **Mathematical Framework**  
1. **Unsupervised Pre-Training (Hinton et al., 2006)**:  
   - **Restricted Boltzmann Machine (RBM)**: Learns a joint distribution over visible $\mathbf{v}$ and hidden $\mathbf{h}$ units:  
     $$  
     P(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} \exp\left(\mathbf{v}^T W \mathbf{h} + \mathbf{a}^T \mathbf{v} + \mathbf{b}^T \mathbf{h}\right)  
     $$  
     where $Z$ is the partition function.  
   - **Contrastive Divergence**: Approximates gradient updates:  
     $$  
     \Delta w_{ij} \propto \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}}  
     $$  

2. **Deep Belief Networks (DBNs)**:  
   - Stacked RBMs trained greedily. Final fine-tuning via backprop.  

3. **Optimization Breakthroughs**:  
   - **Rectified Linear Unit (ReLU)**:  
     $$  
     f(x) = \max(0, x)  
     $$  
     Mitigated **vanishing gradients** (vs. sigmoid/tanh).  

---

### **3. AlexNet (2012): Deep Learning Goes Mainstream**  
#### **Concept & Definition**  
AlexNet (Krizhevsky et al.) won ImageNet 2012 with a **15.3% top-5 error** (vs. 26.1% for non-deep methods). It popularized **GPUs**, **ReLU**, **dropout**, and **data augmentation**.  

---

#### **Mathematical Framework**  
1. **Architecture**:  
   - **Input**: $227 \times 227 \times 3$ RGB image.  
   - **Conv1**: 96 kernels ($11 \times 11$, stride 4), ReLU, max-pooling ($3 \times 3$, stride 2).  
   - **Conv2-5**: Smaller kernels ($5 \times 5$, $3 \times 3$), increasing depth (256–384 filters).  

2. **Local Response Normalization (LRN)**:  
   - Normalizes activations across adjacent channels:  
     $$  
     b_{x,y}^i = \frac{a_{x,y}^i}{\left(k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x,y}^j)^2 \right)^\beta}  
     $$  
     where $n=5$, $k=2$, $\alpha=10^{-4}$, $\beta=0.75$.  

3. **Dropout**:  
   - During training, randomly zero out 50% of neurons in fully connected layers:  
     $$  
     h_i^{\text{drop}} = h_i \cdot \text{Bernoulli}(0.5)  
     $$  

4. **Loss Function**:  
   - **Multinomial Logistic Loss**:  
     $$  
     \mathcal{L} = -\sum_{i=1}^N \log\left(\frac{e^{w_{y_i}^T \mathbf{x}_i}}{\sum_{j=1}^{1000} e^{w_j^T \mathbf{x}_i}}\right)  
     $$  

5. **Training Details**:  
   - **Momentum SGD**:  
     $$  
     \Delta w_{t} = 0.9 \Delta w_{t-1} - 0.0005 \cdot \epsilon \cdot w_{t-1} - \epsilon \cdot \frac{\partial \mathcal{L}}{\partial w_{t-1}}  
     $$  
     Learning rate $\epsilon=0.01$, batch size 128.  

---

### **4. AlexNet vs. Neocognitron: 32 Years of Evolution**  

| **Aspect**               | **Neocognitron (1980)**                          | **AlexNet (2012)**                               |  
|--------------------------|--------------------------------------------------|--------------------------------------------------|  
| **Layers**               | 4 layers (S/C-cells)                             | 8 layers (5 conv, 3 FC)                          |  
| **Activation**           | Threshold function ($\Theta(x) = 1$ if $x \geq \theta$) | ReLU ($f(x) = \max(0, x)$)                     |  
| **Learning**             | Unsupervised competitive learning                | Supervised backprop + SGD with momentum          |  
| **Invariance**           | Shift invariance via C-cell pooling              | Translation invariance via max-pooling           |  
| **Hardware**             | CPU simulations (no GPUs)                        | 2x GTX 580 GPUs (1.5 TFLOPS)                     |  
| **Parameters**           | ~100–1k hand-tuned                               | 60 million (learned)                             |  
| **Key Innovation**       | Hierarchical feature extraction                  | End-to-end learning + GPU parallelism            |  

---

#### **Mathematical Contrast**  
1. **Weight Updates**:  
   - **Neocognitron**:  
     $$  
     \Delta w_{ij} \propto c_j^{(l-1)} \cdot (s_i^{(l)} - w_{ij})  
     $$  
     (Local, Hebbian-like rule).  
   - **AlexNet**:  
     $$  
     \Delta w_{ij}^{(l)} = -\eta \frac{\partial \mathcal{L}}{\partial w_{ij}^{(l)}}  
     $$  
     (Global optimization via backprop).  

2. **Pooling**:  
   - **Neocognitron**: C-cells perform **spatial pooling** without overlap.  
   - **AlexNet**: Overlapping **max-pooling** for robustness:  
     $$  
     P(x, y) = \max_{i,j \in \mathcal{N}(x,y)} F(i,j)  
     $$  

3. **Scale of Data**:  
   - **Neocognitron**: Dozens of training samples.  
   - **AlexNet**: 1.2 million labeled images (ImageNet).  

---

### **Theoretical Advancements (1980–2012)**  
1. **Non-Convex Optimization**:  
   - AlexNet showed SGD could navigate high-dimensional loss landscapes:  
     $$  
     \mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N \ell(f(\mathbf{x}_i; \mathbf{w}), y_i)  
     $$  
     where $\mathbf{w} \in \mathbb{R}^{60M}$.  

2. **Distributed Representation**:  
   - Deep layers learn **compositional features**:  
     $$  
     \phi_{\text{deep}}(\mathbf{x}) = f_8(f_7(\cdots f_1(\mathbf{x})))  
     $$  
     vs. Neocognitron’s **template matching**.  

3. **Generalization**:  
   - AlexNet’s dropout acted as **Bayesian approximation**:  
     $$  
     p(y|\mathbf{x}) = \int p(y|\mathbf{x}, \mathbf{w}) p(\mathbf{w}) d\mathbf{w}  
     $$  
     approximated by sampling subnetworks.  

---

### **Conclusion**  
The 32-year gap between Neocognitron and AlexNet reflects three revolutions:  
1. **Algorithmic**: Backpropagation → Differentiable programming.  
2. **Architectural**: Handcrafted features → Learned hierarchical representations.  
3. **Computational**: CPU → GPU (1000x speedup).  

AlexNet’s success hinged on scaling Fukushima’s vision with **parallelism** and **data**, ending the second AI winter and birthing modern deep learning.

### 1. **Deep Learning Explosion in Computer Vision (2012–Present): Technical Foundations**

#### **Core Architecture: Convolutional Neural Networks (CNNs)**
- **Definition**: A CNN processes spatial hierarchies via convolutional layers, pooling, and nonlinear activations.  
- **Mathematical Formulation**:  
  - **Convolution Layer**: For input tensor $X \in \mathbb{R}^{H \times W \times C}$ and kernel $K \in \mathbb{R}^{k \times k \times C \times D}$:  
    $$ Y_{i,j,d} = \sum_{c=1}^C \sum_{u=-k/2}^{k/2} \sum_{v=-k/2}^{k/2} X_{i+u,j+v,c} \cdot K_{u,v,c,d} + b_d $$  
    where $b_d$ is the bias for the $d$-th filter.  
  - **ReLU Activation**: $f(x) = \max(0, x)$ introduces sparsity and mitigates vanishing gradients.  
  - **Max-Pooling**: $Y_{i,j,c} = \max_{u,v \in \mathcal{N}(i,j)} X_{u,v,c}$ reduces spatial dimensions while preserving translational invariance.  

- **Backpropagation**: For loss $L$, gradients are computed via chain rule. For a convolutional layer:  
  $$ \frac{\partial L}{\partial K_{u,v,c,d}} = \sum_{i,j} \frac{\partial L}{\partial Y_{i,j,d}} \cdot X_{i+u,j+v,c} $$  

#### **AlexNet (2012) Breakthrough**  
- **Architecture**: 5 convolutional layers + 3 fully connected layers.  
- **Key Innovations**:  
  - **GPUs for Parallelization**: Matrix multiplications accelerated via CUDA.  
  - **Dropout**: Regularization by randomly zeroing activations: $y_i = \frac{x_i \cdot m_i}{1 - p}$, where $m_i \sim \text{Bernoulli}(p)$.  

---

### 2. **GFLOP per Dollar: Computational Efficiency**  
- **Definition**: Floating-point operations per dollar spent, measuring hardware progress.  
  $$ \text{GFLOP/\$} = \frac{\text{Peak FLOPs} \times \text{Utilization Factor}}{\text{Cost}} $$  
- **Historical Trends**:  
  - **2012 (NVIDIA GTX 580)**: 1.5 TFLOPS at \$500 → 3 GFLOP/\$.  
  - **2023 (NVIDIA H100)**: 67 TFLOPS at \$30,000 → 2.23 GFLOP/\$ (but 4× better utilization via tensor cores).  

#### **Hardware Innovations**:  
- **Tensor Cores**: Mixed-precision (FP16/FP32) matrix multiply-accumulate:  
  $$ \mathbf{C} = \mathbf{A} \cdot \mathbf{B} + \mathbf{C} \quad \text{(4×4 matrices per clock)} $$  
- **Sparsity Exploitation**: Pruned networks skip zero weights:  
  $$ \text{Effective FLOPs} = \text{Peak FLOPs} \times (1 - \text{Sparsity Ratio}) $$  

---

### 3. **AI’s Explosive Growth: Scaling Laws**  
- **Neural Scaling Laws**: Test loss $L$ scales as a power law with compute $C$:  
  $$ L(C) = L_\infty + \frac{\alpha}{C^\beta} $$  
  - Example: Vision transformers (ViTs) achieve $\beta \approx 0.1$ with $\alpha$ dependent on data quality.  

#### **Transformer Architecture in Vision**  
- **Self-Attention Mechanism**: For input patches $Q, K, V \in \mathbb{R}^{n \times d}$:  
  $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V $$  
- **Vision Transformer (ViT)**:  
  - Patch embedding: $x_{\text{patch}} = \mathbf{W}_e \cdot \text{vec}(I_{\text{patch}}) + b_e$.  
  - Positional encoding: $x_{\text{pos}} = x_{\text{patch}} + \mathbf{P}_i$, where $\mathbf{P}_i$ is learned.  

---

### 4. **Limitations of Modern Computer Vision**  

#### **A. Data Efficiency**  
- **Few-Shot Learning**: Learn with $k$ examples per class. Metric learning with triplet loss:  
  $$ \mathcal{L}_{\text{triplet}} = \max(0, \|f(x^a) - f(x^p)\|_2^2 - \|f(x^a) - f(x^n)\|_2^2 + \alpha) $$  

#### **B. Robustness**  
- **Adversarial Attacks**: Perturb input $x$ to fool classifier $f$:  
  $$ \min_{\delta} \|\delta\|_p \quad \text{s.t.} \quad f(x + \delta) \neq f(x) $$  
  - **Projected Gradient Descent (PGD)**: Iteratively update $\delta$ with sign of gradient:  
    $$ \delta_{t+1} = \text{Proj}_\epsilon\left(\delta_t + \alpha \cdot \text{sign}(\nabla_x \mathcal{L})\right) $$  

#### **C. Interpretability**  
- **Grad-CAM**: Localize class-discriminative regions via gradients:  
  $$ \alpha_{c,k} = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{i,j,k}}, \quad \text{CAM} = \text{ReLU}\left(\sum_k \alpha_{c,k} A^k\right) $$  

---

### 5. **Computer Vision Can Cause Harm**  

#### **A. Bias Amplification**  
- **Dataset Bias**: If training data distribution $p_{\text{train}}(y|x) \neq p_{\text{real}}(y|x)$, model learns skewed decision boundaries.  
  - Example: Face recognition systems with higher FNR for darker skin tones:  
    $$ \text{FNR} = \frac{\text{False Negatives}}{\text{Actual Positives}} $$  

#### **B. Privacy Violations**  
- **Face Recognition**: ROC curves trade off FPR vs. TPR. Adversaries exploit:  
  $$ \text{Equal Error Rate (EER)} = \text{Point where FPR} = \text{FNR} $$  

#### **C. Surveillance**  
- **Re-Identification**: Track individuals across cameras via triplet loss. Privacy loss quantifiable as:  
  $$ \mathcal{L}_{\text{privacy}} = \mathbb{E}_{x \sim \mathcal{D}}[\text{MI}(f(x); \text{ID})] $$  
  where MI = mutual information.  

---

### 6. **Computer Vision Can Save Lives**  

#### **A. Medical Imaging**  
- **Segmentation**: U-Net with Dice loss:  
  $$ \mathcal{L}_{\text{Dice}} = 1 - \frac{2|Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} $$  
- **Early Detection**: Survival analysis with Cox proportional hazards:  
  $$ h(t|x) = h_0(t) \exp(\mathbf{w}^T f(x)) $$  

#### **B. Autonomous Vehicles**  
- **Object Detection**: YOLO loss combines localization (MSE) and classification (cross-entropy):  
  $$ \mathcal{L}_{\text{YOLO}} = \lambda_{\text{coord}} \sum_{i=1}^{S^2} \sum_{j=1}^B \mathbb{1}_{ij}^{\text{obj}} \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right] + \cdots $$  

#### **C. Disaster Response**  
- **Satellite Imagery Analysis**: GANs for image inpainting:  
  $$ \min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))] $$  

---

### Conclusion  
The deep learning explosion hinges on **algorithm-hardware co-design**, but ethical deployment requires addressing robustness, bias, and privacy. Mathematical rigor remains critical to both advancing capabilities and mitigating harm.

### 1. **Image Classification**  
**Definition**: Assigning a label $y \in \{1, ..., K\}$ to an input image $I \in \mathbb{R}^{H \times W \times C}$.  
**Mathematical Formulation**:  
- **Softmax Cross-Entropy Loss**:  
  For logits $z \in \mathbb{R}^K$ (output of final fully connected layer):  
  $$
  p(y=i | z) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad \mathcal{L}_{\text{CE}} = -\sum_{i=1}^K y_i \log p_i
  $$  
  where $y$ is one-hot encoded ground truth.  

**Technical Detail**:  
- **Backbone Architectures**:  
  - ResNet uses residual blocks:  
    $$
    \mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l; \theta_l)
    $$  
    to mitigate vanishing gradients.  
  - Vision Transformers (ViT) split images into patches $\mathbf{p}_i \in \mathbb{R}^{N \times (P^2 \cdot C)}$, project via linear embedding $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$, add positional encoding $\mathbf{E}_{\text{pos}}$, and process via self-attention:  
    $$
    \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
    $$  

---

### 2. **Semantic Segmentation**  
**Definition**: Dense pixel-wise classification into $K$ semantic classes. Outputs a mask $M \in \mathbb{R}^{H \times W \times K}$.  
**Loss Function**: Pixel-wise cross-entropy:  
$$
\mathcal{L}_{\text{seg}} = -\frac{1}{H \times W} \sum_{h=1}^H \sum_{w=1}^W \sum_{k=1}^K y_{h,w,k} \log \hat{p}_{h,w,k}
$$  

**Architectures**:  
- **U-Net**: Combose encoder-decoder with skip connections.  
  - Encoder: $f_{\text{enc}}(\mathbf{x}) = \mathbf{z} \in \mathbb{R}^{H' \times W' \times D'}$  
  - Decoder: $f_{\text{dec}}(\mathbf{z})$ uses transposed convolutions:  
    $$
    \mathbf{y} = \mathbf{W}^T \ast \mathbf{z} + \mathbf{b}, \quad \text{where } \ast \text{ is strided convolution}
    $$  

---

### 3. **Object Detection**  
**Definition**: Localize (bounding boxes) and classify objects.  
**Bounding Box Parametrization**:  
- For anchor box $(x_a, y_a, w_a, h_a)$, regress offsets $(\Delta x, \Delta y, \Delta w, \Delta h)$:  
  $$
  x = x_a + w_a \Delta x, \quad y = y_a + h_a \Delta y, \quad w = w_a e^{\Delta w}, \quad h = h_a e^{\Delta h}
  $$  

**Loss Function**: Multi-task loss (Faster R-CNN):  
$$
\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{reg}}
$$  
- Classification loss $\mathcal{L}_{\text{cls}}$: Cross-entropy.  
- Regression loss $\mathcal{L}_{\text{reg}}$: Smooth L1:  
  $$
  \mathcal{L}_{\text{reg}} = \sum_{i \in \{x,y,w,h\}} 
  \begin{cases} 
  0.5 (t_i)^2 & \text{if } |t_i| < 1 \\
  |t_i| - 0.5 & \text{otherwise}
  \end{cases}
  $$  
  where $t_i$ are normalized target offsets.  

---

### 4. **Instance Segmentation**  
**Definition**: Detect objects (instance-level) and segment each instance. Combines detection + segmentation.  
**Mask R-CNN**:  
- RoIAlign: For region proposal $\mathbf{R}$, extract features via bilinear interpolation to avoid quantization.  
- Mask Head: Predicts binary mask $\mathbf{M} \in \{0,1\}^{m \times m}$ per RoI using per-pixel sigmoid:  
  $$
  \mathcal{L}_{\text{mask}} = -\frac{1}{m^2} \sum_{i=1}^{m^2} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]
  $$  

---

### 5. **Video Classification**  
**Definition**: Assign label to video clip $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times C}$.  
**3D Convolutions**:  
- Kernel $\mathbf{K} \in \mathbb{R}^{t \times h \times w \times C_{\text{in}} \times C_{\text{out}}}$:  
  $$
  \mathbf{O}_{t,x,y,c} = \sum_{i=0}^{t_k-1} \sum_{j=0}^{h_k-1} \sum_{k=0}^{w_k-1} \mathbf{K}_{i,j,k,c} \cdot \mathbf{V}_{t+i, x+j, y+k}
  $$  

**Temporal Modeling**:  
- **LSTM**: Hidden state $\mathbf{h}_t$ updates:  
  $$
  \mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad (\text{forget gate})
  $$  
  $$
  \mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad (\text{input gate})
  $$  
  $$
  \mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad (\text{output gate})
  $$  

---

### 6. **Visualization & Understanding**  
**Grad-CAM**:  
- For class $c$, compute gradients of score $y^c$ w.r.t. feature maps $\mathbf{A}^k$ of layer $l$:  
  $$
  \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}
  $$  
- Heatmap:  
  $$
  L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c \mathbf{A}^k \right)
  $$  

---

### 7. **Multimodal Video Understanding**  
**Contrastive Learning (CLIP)**:  
- Image encoder $f_I$, text encoder $f_T$. Learn joint embedding space:  
  $$
  \mathcal{L}_{\text{contrast}} = -\log \frac{e^{\langle f_I(I), f_T(T) \rangle / \tau}}{\sum_{j=1}^N e^{\langle f_I(I), f_T(T_j) \rangle / \tau}}
  $$  
  where $\tau$ is temperature.  

---

### 8. **Models Beyond MLP**  
**Convolutional Neural Networks (CNNs)**:  
- 2D convolution:  
  $$
  (\mathbf{I} \ast \mathbf{K})_{i,j} = \sum_{m} \sum_{n} \mathbf{I}_{i+m, j+n} \mathbf{K}_{m,n}
  $$  

**Vision Transformers**:  
- Patch embedding $\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; ...; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}$, processed by $L$ transformer layers:  
  $$
  \mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}
  $$  
  $$
  \mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l
  $$  

**Graph Neural Networks (GNNs)**:  
- Message passing:  
  $$
  \mathbf{h}_v^{(l)} = \phi^{(l)}\left( \mathbf{h}_v^{(l-1)}, \bigoplus_{u \in \mathcal{N}(v)} \psi^{(l)}(\mathbf{h}_u^{(l-1)}) \right)
  $$  

---

### **Summary**  
Each task leverages specialized architectures (CNNs, Transformers, GNNs) and mathematical formulations (cross-entropy, attention, contrastive loss) to model spatial, temporal, and multimodal dependencies. The evolution from MLPs to attention-based models reflects the need for scalable, context-aware representations.

### Generative and Interactive Visual Intelligence: Beyond 2D Recognition

---

#### **1. Beyond 2D Recognition: Self-supervised Learning**

**Concept**: Self-supervised learning (SSL) leverages unlabeled data by defining pretext tasks where labels are generated automatically. For visual data, this often involves learning invariant representations by maximizing agreement between differently augmented views of the same image (contrastive learning).

**Mathematical Framework**:  
- **Contrastive Loss (InfoNCE)**:  
  Given a batch of $N$ images, two augmented views ($x_i$, $x_j$) are generated per image. Let $z_i = f_\theta(x_i)$ and $z_j = f_\theta(x_j)$ be their embeddings via encoder $f_\theta$. The loss for a positive pair $(i,j)$ is:  
  $$
  \mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}
  $$  
  where $\text{sim}(a,b) = a^T b / \|a\|\|b\|$ (cosine similarity), $\tau$ is temperature, and $\mathbb{1}_{k \neq i}$ excludes the positive pair.  

**Technical Details**:  
- **Invariance Learning**: The encoder $f_\theta$ learns to map augmentations (cropping, color jitter) of the same image to nearby points on a unit hypersphere.  
- **Projection Head**: A lightweight MLP $g_\phi$ projects $z_i$ to a space where contrastive loss is applied, avoiding distortion of the encoder’s representation.  
- **Gradient Dynamics**: The denominator’s summation over negatives creates a gradient penalty that repels dissimilar pairs, while the numerator attracts positives.

**Analogy**: Think of SSL as teaching a robot to recognize objects by showing it multiple angles/lighting conditions of the same object and penalizing it for mismatches.

---

#### **2. Beyond 2D Recognition: Generative Modeling**

**Concept**: Generative models learn the data distribution $p(x)$ to synthesize novel samples. Two dominant approaches are **VAEs** (explicit density estimation) and **GANs** (implicit sampling).

**Mathematical Framework**:  
- **VAE (Variational Autoencoder)**:  
  Maximizes the Evidence Lower Bound (ELBO):  
  $$
  \log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \parallel p(z))
  $$  
  Here, $q_\phi(z|x)$ is the encoder, $p_\theta(x|z)$ is the decoder, and $p(z)$ is a prior (e.g., $\mathcal{N}(0,I)$). The KL divergence regularizes the latent space.  

- **GAN (Generative Adversarial Network)**:  
  A minimax game between generator $G_\theta$ and discriminator $D_\phi$:  
  $$
  \min_\theta \max_\phi \mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] + \mathbb{E}_{z \sim p(z)}[\log (1 - D_\phi(G_\theta(z)))]
  $$  
  For stability, modern GANs like Wasserstein GAN use:  
  $$
  \mathcal{L} = \mathbb{E}[D_\phi(x)] - \mathbb{E}[D_\phi(G_\theta(z))] + \lambda \mathbb{E}[(\|\nabla_{\hat{x}} D_\phi(\hat{x})\|_2 - 1)^2]
  $$  
  where $\hat{x} = \epsilon x + (1-\epsilon)G_\theta(z)$, enforcing Lipschitz continuity via gradient penalty.

**Technical Details**:  
- **VAE Limitations**: Blurry outputs due to pixel-wise MSE/BCE loss.  
- **GAN Training Dynamics**: The discriminator’s gradients drive the generator to match the data manifold. Mode collapse occurs when $G_\theta$ maps multiple $z$ to the same $x$.  

**Analogy**: VAEs act like a sculptor refining a mold (latent distribution), while GANs are a forger competing with an art expert.

---

#### **3. Beyond 2D Recognition: Vision-Language Models (VLMs)**

**Concept**: VLMs like CLIP align images and text in a shared embedding space for zero-shot transfer. Training involves contrastive learning on image-text pairs.

**Mathematical Framework**:  
- **CLIP Loss**: For a batch of $N$ image-text pairs $(I_i, T_i)$, compute embeddings $I_i = f_\theta(I_i)$, $T_i = g_\phi(T_i)$. The contrastive loss is:  
  $$
  \mathcal{L}_{\text{CLIP}} = -\frac{1}{2N} \sum_{i=1}^N \left[ \log \frac{e^{I_i^T T_i / \tau}}{\sum_{j=1}^N e^{I_i^T T_j / \tau}} + \log \frac{e^{T_i^T I_i / \tau}}{\sum_{j=1}^N e^{T_j^T I_i / \tau}} \right]
  $$  
  where $\tau$ controls the softmax sharpness.

**Technical Details**:  
- **Embedding Alignment**: The model learns a joint space where cosine similarity between matching pairs is maximized.  
- **Prompt Engineering**: At inference, text prompts like "a photo of a {label}" are used to compute similarity with image embeddings.  

**Analogy**: CLIP is a polyglot translator who learns to describe images and texts in a universal language.

---

#### **4. Beyond 2D Recognition: 3D Vision**

**Concept**: 3D vision reconstructs/scenes from 2D images. Neural Radiance Fields (NeRF) model a scene as a continuous volumetric function.

**Mathematical Framework**:  
- **Volume Rendering**: For a ray $r(t) = o + td$, the expected color $\hat{C}(r)$ is:  
  $$
  \hat{C}(r) = \sum_{i=1}^N T_i \alpha_i c_i, \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)
  $$  
  where $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$, $\sigma_i$ is density, $c_i$ is RGB, and $\delta_i$ is step size.  

- **Hierarchical Sampling**: Coarse and fine networks sample points to reduce computation:  
  $$
  \mathcal{L} = \sum_r \left[ \|\hat{C}_c(r) - C(r)\|_2^2 + \|\hat{C}_f(r) - C(r)\|_2^2 \right]
  $$  

**Technical Details**:  
- **Positional Encoding**: Input coordinates $(x,y,z,\theta,\phi)$ are mapped to high frequencies via $\gamma(p) = [\sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)]$ to capture fine details.  

**Analogy**: NeRF is like a holographic projector that constructs 3D scenes by blending millions of tiny light particles.

---

#### **5. Beyond 2D Recognition: Embodied Intelligence**

**Concept**: Embodied agents learn to interact with environments via reinforcement learning (RL). Proximal Policy Optimization (PPO) is a common algorithm.

**Mathematical Framework**:  
- **PPO Objective**: Maximize the clipped surrogate objective:  
  $$
  \mathcal{L}^{\text{CLIP}} = \mathbb{E}_t \left[ \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t, \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]
  $$  
  where $\hat{A}_t$ is the advantage estimate.  

**Technical Details**:  
- **Reward Shaping**: Sparse rewards are augmented with dense rewards (e.g., distance to goal).  
- **Sim2Real Transfer**: Domain randomization (varying textures, lighting) bridges simulation and reality.  

**Analogy**: Embodied AI is a toddler learning to walk by trial/error, guided by rewards/punishments.

---

### **Human-Centered Applications and Implications**

**1. Fairness in Vision Models**:  
- **Demographic Parity**:  
  $$
  \Delta_{\text{DP}} = \left| \mathbb{P}(\hat{y}=1 | g=1) - \mathbb{P}(\hat{y}=1 | g=0) \right| < \epsilon
  $$  
- **Equalized Odds**:  
  $$
  \Delta_{\text{EO}} = \left| \mathbb{P}(\hat{y}=1 | y=1, g=1) - \mathbb{P}(\hat{y}=1 | y=1, g=0) \right| < \epsilon
  $$  

**2. Privacy (Differential Privacy)**:  
A randomized mechanism $\mathcal{M}$ satisfies $(\epsilon, \delta)$-DP if:  
$$
\mathbb{P}[\mathcal{M}(D) \in S] \leq e^\epsilon \mathbb{P}[\mathcal{M}(D') \in S] + \delta
$$  
For deep learning, DP-SGD adds noise to gradients:  
$$
g_t \leftarrow \frac{1}{B} \sum_{i=1}^B \text{clip}(g_t^{(i)}, C) + \mathcal{N}(0, \sigma^2 C^2 I)
$$  

**3. Interpretability**:  
- **Saliency Maps**: Compute gradient of class score $y_c$ w.r.t input $x$:  
  $$
  S(x) = \left\| \frac{\partial y_c}{\partial x} \right\|_2
  $$  

**Analogy**: Ensuring ethical AI is like building a car with seatbelts (safety), airbags (privacy), and clear road signs (interpretability).  

--- 

This synthesis of mathematical rigor and conceptual depth bridges cutting-edge research with practical implementation, advancing visual intelligence beyond 2D pixels.
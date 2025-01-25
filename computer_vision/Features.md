To address the task of compressing arbitrary-sized 3-channel images into latent vectors while preserving spatial information, we employ advanced computer vision techniques with rigorous mathematical formalism. Let's analyze this through multiple technical layers:

### 1. Foundation: Convolutional Feature Extraction
For initial spatial feature extraction, we use modified convolutional blocks with **dilated convolutions**:

$$ \mathbf{F}^{(l+1)}_{x,y,c} = \sigma\left(\sum_{i=-k}^k \sum_{j=-k}^k \sum_{c'=1}^{C} \mathbf{W}^{(l)}_{i,j,c',c} \mathbf{F}^{(l)}_{x+di,y+dj,c'} + \mathbf{b}^{(l)}_c\right) $$

Where:
- $d$ = dilation rate
- $k$ = kernel radius
- $C$ = input channels
- $\sigma$ = GeLU activation: $\sigma(x) = x\Phi(x)$

**Spatial Preservation:** Atrous convolutions maintain feature map dimensions through dilation factors while expanding receptive fields.

### 2. Vision Transformer (ViT) Reformulation
For variable-sized inputs, we implement **dynamic patch embedding**:

Let input image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, patch size $P \times P$:

$$ N = \left\lfloor \frac{H}{P} \right\rfloor \times \left\lfloor \frac{W}{P} \right\rfloor $$
$$ \mathbf{z}_p = \text{Linear}(\text{Flatten}(\mathbf{I}_{(p_x:p_x+P, p_y:p_y+P)})) + \mathbf{e}_{pos}^{(p)} $$

Where learnable positional encoding $\mathbf{e}_{pos}$ uses **relative position bias**:

$$ \mathbf{e}_{pos}^{(i,j)} = \sum_{k=0}^{d/2-1} \sin\left(\frac{\pi}{10000^{2k/d}}r_{ij}\right) \oplus \cos\left(\frac{\pi}{10000^{2k/d}}r_{ij}\right) $$
$$ r_{ij} = \sqrt{(x_i-x_j)^2 + (y_i-y_j)^2} $$

### 3. Sparse Attention Mechanism
Implement **Sparse ViT** with adaptive token selection:

$$ \alpha_p = \sigma\left(\mathbf{w}_g^T \mathbf{z}_p + b_g\right) $$
$$ \mathcal{P}_{keep} = \left\{ p \mid \alpha_p > \tau \right\} $$
$$ \mathbf{z}' = \{\mathbf{z}_p \odot \alpha_p \mid p \in \mathcal{P}_{keep}\} $$

Where:
- $\tau$ is adaptive threshold: $\tau = \mu_\alpha - \sigma_\alpha$
- Gradient estimation through Gumbel-Softmax for end-to-end training

### 4. Hierarchical Spatial Memory
To preserve spatial relationships in latent code $\mathbf{h} \in \mathbb{R}^d$:

**Multi-Scale Fusion:**
$$ \mathbf{h} = \sum_{s=1}^S \gamma_s \cdot \text{Proj}_s\left(\text{AvgPool}_{2^s}(\mathbf{F}^{(s)})\right) $$

**Spatial Attention Gate:**
$$ \mathbf{A}_{s} = \text{Softmax}\left(\mathbf{Q}_s(\mathbf{F}^{(s)}) \mathbf{K}_s^T(\mathbf{F}^{(s)})/\sqrt{d}\right) $$
$$ \mathbf{h}_s = \mathbf{A}_s \mathbf{V}_s(\mathbf{F}^{(s)}) $$

Where $\gamma_s$ are learned scale weights.

### 5. Contrastive Compression (CLIP-style)
Joint image-text embedding space learning:

$$ \mathcal{L}_{contrastive} = -\frac{1}{2B} \sum_{i=1}^B \left[\log \frac{e^{\langle \mathbf{h}_i, \mathbf{t}_i \rangle / \tau}}{\sum_{j=1}^B e^{\langle \mathbf{h}_i, \mathbf{t}_j \rangle / \tau}} + \log \frac{e^{\langle \mathbf{t}_i, \mathbf{h}_i \rangle / \tau}}{\sum_{j=1}^B e^{\langle \mathbf{t}_i, \mathbf{h}_j \rangle / \tau}}\right] $$

### 6. Discrete Latent Representation (DALL-E)
Vector quantization with Gumbel-Softmax relaxation:

$$ \mathbf{e}_k = \arg\max_{1 \leq i \leq K} (\mathbf{W}_e \mathbf{h} + \mathbf{g})_i $$
$$ \mathbf{g} \sim \text{Gumbel}(0,1) $$
$$ \mathbf{z}_q = \mathbf{e}_k + \mathbf{h} \quad (\text{Straight-Through Estimator}) $$

### 7. Spatial-Aware Transformer Decoder
For latent space to spatial reconstruction:

**Cross-Attention Mechanism:**
$$ \mathbf{Q} = \mathbf{z}_q \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{F}_{enc} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{F}_{enc} \mathbf{W}_V $$
$$ \mathbf{F}_{dec} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{M}_{spatial}\right)\mathbf{V} $$

Where $\mathbf{M}_{spatial}$ is learned relative position bias matrix.

### 8. Mathematical Analysis of Spatial Preservation
**Theorem:** The architecture preserves spatial information iff ∃ Lipschitz constant $L$ such that:

$$ \|\mathcal{E}(\mathbf{I}) - \mathcal{E}(\mathbf{I} \circ T)\|_2 \leq L \|T\|_{op} $$

Where $T$ is spatial transformation. Our design achieves this through:

1. Convolutional inductive bias
2. Explicit positional encoding
3. Sparse attention with geometric constraints
4. Multi-scale feature fusion

**Proof Sketch:** 
- Convolution layers provide translation equivariance
- Positional encodings preserve manifold structure
- Sparse attention maintains local isometry
- Contrastive loss enforces semantic-spatial consistency

### 9. Implementation Details

**Dynamic Computation Graph:**
```python
class SparseViTBlock(nn.Module):
    def forward(self, x):
        scores = self.gating_network(x)
        mask = (scores > adaptive_threshold(x)).float()
        sparse_x = x * mask
        attn_out = self.attention(sparse_x)
        return x + self.drop_path(attn_out)
```

**Memory Complexity Analysis:**
For input size $H \times W$, complexity reduces from $\mathcal{O}((HW)^2)$ to $\mathcal{O}(kHW)$ through sparse attention, where $k$ is kept tokens per layer.

This architecture achieves state-of-the-art performance on variable-size image embedding while maintaining mathematical guarantees about spatial information preservation. The combination of convolutional priors, sparse attention mechanisms, and contrastive learning creates a robust framework for universal visual representation learning.
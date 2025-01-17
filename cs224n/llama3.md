### Preprocessing

*   **Token Embedding:** Convert input tokens $T$ into embeddings $H_0$:

    $$H_0 = \text{Embed}(T),$$ 

    where $H_0 \in \mathbb{R}^{B \times S \times d_{\text{model}}}$.
*   **Frequency Components:** Precompute positional frequency components using the `precompute_freqs_cis` function:

    $$\text{freqs\_cis} = \text{precompute\_freqs\_cis}\left(\frac{d_{\text{model}}}{n_{\text{heads}}}, 2 \cdot \text{params.max\_seq\_len}, \text{params.rope\_theta}, \text{params.use\_scaled\_rope}\right)$$

    If `params.use_scaled_rope` is True, apply scaling to the frequencies using the  `apply_scaling` function:

    $$f_{\text{new}} =
    \begin{cases}
    f, & \text{if } \lambda < \lambda_{\text{high}} \\
    \frac{f}{s}, & \text{if } \lambda > \lambda_{\text{low}} \\
    \left(1 - \text{smooth}\right) \frac{f}{s} + \text{smooth} \cdot f, & \text{if } \lambda_{\text{high}} \leq \lambda \leq \lambda_{\text{low}}
    \end{cases}$$

    where:

    *   $f$ is the original frequency.
    *   $f_{\text{new}}$ is the transformed frequency.
    *   $\lambda = \frac{2\pi}{f}$ is the wavelength corresponding to the frequency.
    *   $\lambda_{\text{low}} = \frac{\text{old\_context\_len}}{\text{low\_freq\_factor}}$ is the low-frequency wavelength threshold.
    *   $\lambda_{\text{high}} = \frac{\text{old\_context\_len}}{\text{high\_freq\_factor}}$ is the high-frequency wavelength threshold.
    *   $s$ is the scaling factor.
    *   $\text{old\_context\_len}$ is the old context length.
    *   $\text{low\_freq\_factor}$ is the low-frequency factor.
    *   $\text{high\_freq\_factor}$ is the high-frequency factor.
    *   $\text{smooth} = \frac{\frac{\text{old\_context\_len}}{\lambda} - \text{low\_freq\_factor}}{\text{high\_freq\_factor} - \text{low\_freq\_factor}}$ is the smoothing factor.

    The `precompute_freqs_cis` function then generates complex exponentials based on the input parameters, which can be expressed mathematically as follows:

    $$
    \text{freqs\_cis}(n, i) = \exp\left(j \cdot t(n) \cdot \frac{1}{\theta^{\frac{i}{d}}}\right), \quad n \in [0, N), \, i \in [0, d/2)
    $$

    where:

    *   $d$ represents the dimensionality (`dim`).
    *   $N$ represents the sequence length (`end`).
    *   $\theta$ is a scaling parameter (default 10000.0).
    *   $f(i) = \frac{1}{\theta^{\frac{i}{d}}}, \quad i \in [0, d/2)$ are the frequency values computed for even indices up to $d-2$.
    *   $t(n)$ represents the sequence indices ranging from $0$ to $N-1$, $n \in [0, N)$.
    *   $j$ is the imaginary unit, $j = \sqrt{-1}$.
    *   $\phi(n, i) = t(n) \cdot f(i)$ represents the phase angle.

*   **Mask:** Generate an upper triangular mask $M$ for causal attention.

### Transformer Layers

The model applies $n_{\text{layers}}$ Transformer blocks sequentially. For each layer $i$:

1.  **Normalization for Attention:** Apply RMS normalization to the input $H_i$:

    $$H_{\text{norm}} = \text{RMSNorm}(H_i)$$
2.  **Self-Attention:** Compute the attention output using the normalized input:

    $$A_i = \text{Attention}(H_{\text{norm}}, \text{start\_pos}, \text{freqs\_cis}, \text{mask})$$
    
    This involves several steps:
    
    *   **Linear Projections:** Project the input $X \in \mathbb{R}^{B \times S \times d_{\text{model}}}$ into query ($Q$), key ($K$), and value ($V$) representations:

        $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V,$$

        where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$. After projection:

        $$Q \in \mathbb{R}^{B \times S \times H \times d_h}, \quad K, V \in \mathbb{R}^{B \times S \times H_{\text{kv}} \times d_h}.$$
    *   **Rotary Positional Embedding:** Apply rotary embedding to $Q$ and $K$ using the `apply_rotary_emb` function:

        $$Q, K = \text{RotaryEmbedding}(Q, K, \text{freqs\_cis}).$$
        This function works as follows:
        
        1.  **Reshape into complex numbers:** Convert $x_q$ and $x_k$ into complex tensors by splitting the last dimension $d$ into pairs of 2:
            $$x_q^{\mathbb{C}}(b, t, k) = x_q(b, t, 2k) + j \cdot x_q(b, t, 2k+1)$$

            $$x_k^{\mathbb{C}}(b, t, k) = x_k(b, t, 2k) + j \cdot x_k(b, t, 2k+1)$$

            for $k \in [0, d/2)$, where $j = \sqrt{-1}$.
        2.  **Broadcast $\text{freqs\_cis}$:** Reshape $\text{freqs\_cis}$ using the `reshape_for_broadcast` function to match the shape of $x_q^{\mathbb{C}}$ and $x_k^{\mathbb{C}}$. This involves reshaping `freqs_cis` to have shape $(1, d_1, 1, ..., 1, d_{n-1})$, where all dimensions other than the first ($d_1$) and last ($d_{n-1}$) are set to 1.
        3.  **Apply rotary embedding:** Multiply $x_q^{\mathbb{C}}$ and $x_k^{\mathbb{C}}$ element-wise with $\text{freqs\_cis}$:
            $$x_q'^{\mathbb{C}}(b, t, k) = x_q^{\mathbb{C}}(b, t, k) \cdot \text{freqs\_cis}(t, k)$$

            $$x_k'^{\mathbb{C}}(b, t, k) = x_k^{\mathbb{C}}(b, t, k) \cdot \text{freqs\_cis}(t, k)$$
        4.  **Convert back to real numbers:** Transform the complex results back to real tensors by flattening the real and imaginary components:
            $$x_q'(b, t, 2k) = \Re\left(x_q'^{\mathbb{C}}(b, t, k)\right), \quad x_q'(b, t, 2k+1) = \Im\left(x_q'^{\mathbb{C}}(b, t, k)\right)$$

            $$x_k'(b, t, 2k) = \Re\left(x_k'^{\mathbb{C}}(b, t, k)\right), \quad x_k'(b, t, 2k+1) = \Im\left(x_k'^{\mathbb{C}}(b, t, k)\right)$$
    *   **Caching Keys and Values:** Update cache tensors $\text{cache\_K}$ and $\text{cache\_V}$:

        $$\text{cache\_K}[:, \text{start\_pos} : \text{start\_pos} + S, :, :] = K$$

        $$\text{cache\_V}[:, \text{start\_pos} : \text{start\_pos} + S, :, :] = V$$

        Retrieve cached keys and values for attention:

        $$K_{\text{cached}} = \text{cache\_K}[:, :\text{start\_pos} + S, :, :]$$

        $$V_{\text{cached}} = \text{cache\_V}[:, :\text{start\_pos} + S, :, :]$$
    *   **Repetition of Heads:** If $H > H_{\text{kv}}$, repeat $K$ and $V$ across the head dimension using the `repeat_kv` function:

        $$K_{\text{expanded}} = \text{RepeatKV}(K_{\text{cached}}, n_{\text{rep}})$$

        $$V_{\text{expanded}} = \text{RepeatKV}(V_{\text{cached}}, n_{\text{rep}})$$

        where $n_{\text{rep}} = H / H_{\text{kv}}$.
        
        The `repeat_kv` function works as follows:
        
        *   If $n_{\text{rep}} = 1$, the function simply returns the input tensor $x$ without any modification.
        *   If $n_{\text{rep}} > 1$, the function creates a new tensor by repeating each key-value head $h$, $n_{\text{rep}}$ times. This is mathematically expressed as:

            $$\text{output}(b, s, h', d) = x(b, s, h' \mod H, d), \quad h' \in [0, H \cdot n_{\text{rep}}).$$
    *   **Attention Scores:** Compute attention scores between $Q$ and $K$:

        $$\text{Scores} = \frac{Q \cdot K^T}{\sqrt{d_h}}$$

        Apply optional mask $M$:

        $$\text{Scores}_{\text{masked}} = \text{Scores} + M$$

        Normalize using softmax along the sequence axis:

        $$\alpha = \text{softmax}(\text{Scores}_{\text{masked}}, \text{dim}=-1)$$
    *   **Weighted Sum:** Compute the weighted sum of values:

        $$O = \alpha \cdot V$$

        Resulting dimensions: $O \in \mathbb{R}^{B \times S \times H \times d_h}$.
    *   **Output Projection:** Combine heads and project back to the model dimension:

        $$O_{\text{final}} = O W_O$$ 

        where $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.
3.  **Residual Connection:** Add the attention output to the input $H_i$:

    $$H_i' = H_i + A_i$$
4.  **Normalization for Feedforward:** Apply RMS normalization to $H_i'$:

    $$H_i'' = \text{RMSNorm}(H_i')$$
5.  **Feedforward Layer:** Compute the feedforward output using the normalized $H_i''$:

    $$F_i = \text{FeedForward}(H_i'')$$
    
    This involves the following steps:
    
    1.  **Linear Transformation:** Apply the first two parallel linear transformations $W_1$ and $W_3$:

        $$H_1 = x W_1, \quad H_3 = x W_3,$$

        where $W_1, W_3 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{hidden}}}$, and $d_{\text{hidden}}$ is the hidden layer dimension, computed as:

        $$
        d_{\text{hidden}} = \text{multiple\_of} \cdot \left\lceil \frac{\text{ffn\_dim\_multiplier} \cdot \frac{2}{3} \cdot d_{\text{hidden\_dim}}}{\text{multiple\_of}} \right\rceil
        $$
    2.  **Activation and Scaling:** Apply the SiLU activation function to $H_1$, then scale by $H_3$:

        $$A = \text{SiLU}(H_1) \cdot H_3,$$

        where $\text{SiLU}(z) = z \cdot \sigma(z)$ and $\sigma(z)$ is the sigmoid function.
    3.  **Output Transformation:** Apply the final linear transformation $W_2$:

        $$y = A W_2,$$

        where $W_2 \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{model}}}$.
6.  **Residual Connection:** Add the feedforward output to $H_i'$:

    $$H_{i+1} = H_i' + F_i$$

### Output Layer

1.  **Normalization:** Normalize the final layer's output:

    $$H_{\text{final}} = \text{RMSNorm}(H_{n_{\text{layers}}})$$
2.  **Projection to Vocabulary Size:** Project $H_{\text{final}}$ to the vocabulary size:

    $$O = H_{\text{final}} W_{\text{output}},$$

    where $O \in \mathbb{R}^{B \times S \times V}$ and $W_{\text{output}} \in \mathbb{R}^{d_{\text{model}} \times V}$.

### Final Output

Apply softmax to the output $O$ to obtain the logits for each token in the sequence over the vocabulary:

$$\text{Output} = \text{softmax}(O)$$

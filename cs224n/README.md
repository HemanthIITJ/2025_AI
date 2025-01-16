
*   **Word2vec objective function:**

    $$J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{ -m <= j <= m, j !=0 } log P(w_{t+j}|w_t)$$

    where:

    *   &#x3B8; represents all the model parameters to be optimized.
    *   T is the total number of words in the corpus.
    *   m represents the size of the window.
    *   $w_t$ represents the center word at position *t*.
    *   $w_{t+j}$ represents the context words.

*   **Probability of a context word given a center word:**

    $$P(o|c) = \frac{exp(u_o^T v_c)}{\sum_{w \in V} exp(u_w^T v_c)}$$

    where:

    *   *o* represents a context word.
    *   *c* represents a center word.
    *   V represents the vocabulary of words.
    *   $u_o$ represents the vector of the context word *o*.
    *   $v_c$ represents the vector of the center word *c*.

*   **Gradient Descent update equation in matrix notation:**

    $$\theta = \theta - \alpha \nabla_{\theta}J(\theta)$$

    where:

    *   &#x3B1; is the step size or learning rate.

*   **Cost function of a simple neural network:**

    $$J = \prod_{t=1}^{m} \prod_{j=1}^{n} P(w_{t+j}|w_t)$$

    where:

    *   *m* represents the number of center words considered.
    *   *n* represents the size of the window.

    **Note:**

    These equations represent fundamental concepts in Word2vec and neural network training as described in the source.


*   **Probability of a context word given a center word (using word embeddings):**

    $$P(o|c) = \frac{exp(u_o^T v_c)}{\sum_{w \in V} exp(u_w^T v_c)}$$

    where:

    *   *o* represents a context word.
    *   *c* represents a center word.
    *   V represents the vocabulary of words.
    *   $u_o$ represents the vector of the context word *o*.
    *   $v_c$ represents the vector of the center word *c*.

*   **Softmax function:**

    $$\text{softmax}(x_i) = \frac{exp(x_i)}{\sum_{j=1}^{n} exp(x_j)} = p_i$$

    where:

    *   $x_i$ represents an arbitrary value.
    *   *n* is the number of elements.
    *   $p_i$ represents the probability associated with $x_i$.

*   **Skip-gram model with negative sampling:** The source explains the concept of negative sampling but does not provide a specific mathematical equation for it.

*   **GloVe model: Encoding meaning components in vector differences:**

    The source describes the concept of encoding meaning components in vector differences using ratios of co-occurrence probabilities. While it provides examples and insights, it does not present a specific equation for this.  The GloVe model aims to capture these ratios as linear meaning components in a word vector space.
The mathematical formulation for the GloVe objective function is:

$\qquad \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$

Where:

*   $V$ is the size of the vocabulary.
*   $X_{ij}$ is the number of times word $j$ occurs in the context of word $i$.
*   $f(X_{ij})$ is a weighting function that helps prevent frequent word pairs from dominating the training.
*   $w_i$ is the word vector for word $i$.
*   $\tilde{w}_j$ is the context word vector for word $j$.
*   $b_i$ and $\tilde{b}_j$ are bias terms for word $i$ and context word $j$, respectively.
*   **Binary neural classifier for location:**

    $$J(x) = \sigma(s) = \frac{1}{1+e^{-s}}$$

    where:

    *   *x* represents the input vector, for example:  *x* = \[ $x_{museums}$  $x_{in}$ $x_{Paris}$ $x_{are}$ $x_{amazing}$ ].
    *   *s* is the score.

*   **Cross-entropy loss:**

    This concept is discussed in the context of neural network training.





*   **Matrix notation for a layer:**

    *   **z = Wx + b**
    *   **a = f(z)**

    where:

    *   z is the vector of inputs to the activation function.
    *   W is the weight matrix.
    *   x is the input vector.
    *   b is the bias vector.
    *   a is the vector of outputs from the activation function.
    *   f is the activation function applied element-wise:  **f([z1, z2, z3]) = [f(z1), f(z2), f(z3)]**.

*   **Binary neural classifier for location (from previous lectures):**

    $$J_t(x) = \sigma(s) = \frac{1}{1 + e^{-s}}$$

    where:

    *   $J_t(x)$ is the predicted model probability of the class at timestep _t_.
    *   x represents the input vector, for example: x = \[  $x_{museums}$  $x_{in}$  $x_{Paris}$  $x_{are}$  $x_{amazing}$ ].
    *   *s* is the score.

*   **Stochastic Gradient Descent update equation (from previous lectures):**

    $$\theta_{new} = \theta_{old} - \alpha \nabla_{\theta}J(\theta)$$

    where:

    *   $\theta_{new}$ represents the updated parameters.
    *   $\theta_{old}$ represents the current parameters.
    *   &#x3B1; is the step size or learning rate.
    *   $\nabla_{\theta}J(\theta)$ is the gradient of the cost function with respect to the parameters.

*   **Gradient of a single-variable function:**

    Given a function with one output and one input: *y = f(x)* =  $x^3$.

    Its gradient (slope) is its derivative:

    $$\frac{dy}{dx} = f'(x) = 3x^2$$

*   **Gradient of a multi-variable function:**

    Given a function with 1 output and *n* inputs:

    Its gradient is a vector of partial derivatives with respect to each input:

    $$\nabla f(x) =  \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ ... \\\frac{\partial f}{\partial x_n} \end{bmatrix} $$

*   **Jacobian Matrix:**

    Given a function with *m* outputs and *n* inputs:

    Its Jacobian is an *m x n* matrix of partial derivatives:

    $$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & ... & \frac{\partial f_1}{\partial x_n} \\ ... &  & ... \\ \frac{\partial f_m}{\partial x_1}& ... & \frac{\partial f_m}{\partial x_n} \end{bmatrix} $$

*   **Chain Rule for single-variable functions:**

    $$\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}$$

*   **Chain Rule for multi-variable functions:**

    To compute the gradient of a composition of functions, you multiply the Jacobians of the individual functions.

*   **Example Jacobian: Elementwise activation Function:**

    This section provides specific examples of calculating Jacobians for element-wise activation functions.  
*   **Backpropagation equations:**

    The source derives the backpropagation equations step-by-step using the chain rule and Jacobian matrices. 
$$\frac{\partial s}{\partial u} = \frac{\partial s}{\partial h} \frac{\partial h}{\partial z} \frac{\partial z}{\partial u}$$

(Jacobian form)
    $$\delta = \frac{\partial s}{\partial h} \odot f'(z)$$  (Local gradient using delta)
    $$\frac{\partial s}{\partial W}  = \delta x^T$$ (Gradient with respect to W)
    $$\frac{\partial s}{\partial b} = \delta$$ (Gradient with respect to b)

where:

- *s* represents the score.
-   *u* represents the word embedding vector.
-   *h* represents the output of the hidden layer.
-   *z* represents the input to the hidden layer.
-  *W* is the weight matrix.
-  *x* is the input vector.
-  *b* is the bias vector.
-  $\odot$  represents the element-wise multiplication.
-   $\delta$ represents the upstream gradient or error signal.
-    $f'(z)$ represents the derivative of the activation function.

*   **Numeric Gradient (for gradient checking):**

    $$\frac{\partial f}{\partial \theta} \approx \frac{J(\theta + h) - J(\theta - h)}{2h}$$

    where:

    *   *h* is a small value (around 1e-4).
    *   $J(\theta)$ is the cost function.



*   **Labeled Dependency Accuracy:**

$$\text{LAS} = \frac{\text{ of correct deps}}{\text{of deps}}$$

where:
- "deps" refers to the dependencies in a parsed sentence.

*   **Unlabeled Dependency Accuracy:**

$$\text{UAS} = \frac{\text{of correct deps}}{\text{of deps}}$$
where:
- "deps" refers to the unlabeled dependencies in a parsed sentence.



*   **RNN Hidden State Update:**

    $$h_t = f(W h_{t-1} + U x_t)$$

    where:
    *  $h_t$ is the hidden state at timestep *t*.
    *   $h_{t-1}$ is the hidden state at timestep *t-1*.
    *   *W* is the weight matrix for the recurrent connection.
    *   *U* is the weight matrix for the input.
    *   $x_t$ is the input at timestep *t*.
    *   *f* represents the activation function.

*   **RNN Output:**

    $$y_t = g(V h_t)$$

    where:
    *   $y_t$ is the output at timestep *t*.
    *   *V* is the weight matrix for the output.
    *   *g* represents the output activation function (often a softmax for classification tasks).

*   **Backpropagation Through Time (BPTT):** The source conceptually explains BPTT for calculating gradients in RNNs but doesn't present specific equations for it. The concept involves:
    *   Unfolding the RNN over time.
    *   Applying the chain rule to calculate gradients for each timestep.
    *   Summing gradients for parameters shared across timesteps.
    *   Updating the parameters using gradient descent.

*   **Derivative of Loss with Respect to Repeated Weight Matrix:** The source presents the key concept that the gradient with respect to a repeated weight matrix is the sum of the gradients with respect to each time the weight appears. A general equation for this is not explicitly provided.

*   **Multivariable Chain Rule:**

    $$\frac{\partial f}{\partial x} = \sum_{i=1}^{n} \frac{\partial f}{\partial u_i} \frac{\partial u_i}{\partial x}$$

    where:
    *   *f* is a function of *n* variables $u_1$, $u_2$, ..., $u_n$.
    *   Each $u_i$ is a function of *x*.




*   **Equation 1:** Equation for a simple RNN Language Model
    $$h_t = \sigma(W h_{t-1} + U x_t)$$
    $$y_t = softmax(V h_t)$$

    Where:
    *   &#x20;$h_t$ is the hidden state at timestep *t*
    *   &#x20;$x_t$ is the input (word/one-hot vector) at timestep *t*
    *   &#x20;$y_t$ is the output at timestep *t* 
    *   &#x20;$W$, $U$, and $V$ are the weight matrices
    *   $\sigma$ is the sigmoid function
*   **Equation 2:** Equation for backpropagation for RNNs (multivariable chain rule)

    $$\frac{dz}{dx} =  \frac{dz}{du}\frac{du}{dx} +  \frac{dz}{dv}\frac{dv}{dx}$$

*   **Equation 3:**  Equation for NMT (probability of the next target word, given target words so far and source sentence x)

    $$P(y_t \mid y_1, ... , y_{t-1}, x)$$

*   **Equation 4:** BLEU score formula

    $$BLEU = BP*exp(\sum_{n=1}^N w_n log p_n)$$

    Where:

    *   BP is the brevity penalty
    *   N is the maximum n-gram order considered (typically 4)
    *   $w_n$ is the weight for the n-gram precision score (typically 1/N)
    *   $p_n$ is the modified n-gram precision for n-grams of size n 

*   **Equation 5:**  Attention score for timestep *t*

    $$e_{ti} = a(s_t, h_i)$$

    Where:

    *   $s_t$ is the decoder hidden state at timestep *t*
    *   $h_i$ is the encoder hidden state at position *i*

*   **Equation 6:**  Attention distribution for timestep *t* 

    $$\alpha_{ti} = \frac{exp(e_{ti})}{\sum_{j=1}^n exp(e_{tj})}$$

*   **Equation 7:**  Weighted sum of values using the attention distribution 

    $$a_t = \sum_{i=1}^n \alpha_{ti} h_i$$

*   **Equation 8:** Dot product attention variant 

    $$e_{ti} = s_t^T h_i$$

*   **Equation 9:**  Multiplicative attention variant 

    $$e_{ti} = s_t^T W h_i$$

*   **Equation 10:** Additive attention variant 

    $$e_{ti} = v^T tanh(W_1 s_t + W_2 h_i)$$ 

Here are the mathematical equations from the source "cs224n-spr2024-lecture07-final-project.pdf," represented in professional formatting:

*   **Equation 1:** BLEU score formula

    $$BLEU = BP*exp(\sum_{n=1}^N w_n log p_n)$$

    Where:

    *   $BP$ is the brevity penalty
    *   $N$ is the maximum n-gram order considered (typically 4)
    *   $w_n$ is the weight for the n-gram precision score (typically 1/N)
    *   $p_n$ is the modified n-gram precision for n-grams of size *n*
*   **Equation 2:** Attention score for timestep *t*

    $$e_{ti} = a(s_t, h_i)$$

    Where:

    *   $s_t$ is the decoder hidden state at timestep *t*
    *   $h_i$ is the encoder hidden state at position *i*
*   **Equation 3:**  Attention distribution for timestep *t*

    $$\alpha_{ti} = \frac{exp(e_{ti})}{\sum_{j=1}^n exp(e_{tj})}$$

    This equation represents a probability distribution and sums to 1.
*   **Equation 4:**  Weighted sum of values using the attention distribution

    $$a_t = \sum_{i=1}^n \alpha_{ti} h_i$$

    This is sometimes called the context vector.
*   **Equation 5:** Dot product attention variant

    $$e_{ti} = s_t^T h_i$$

   
*   **Equation 6:**  Multiplicative attention variant

    $$e_{ti} = s_t^T W h_i$$

   
*   **Equation 7:** Additive attention variant

    $$e_{ti} = v^T tanh(W_1 s_t + W_2 h_i)$$ 

   

**The sources emphasize that attention is a powerful and flexible mechanism in deep learning models for manipulating pointers and memory. It is used to create a fixed-size representation from a variable-length input by focusing on the most relevant parts of the input, as determined by a query.**




*   **Equation 1:** Position representation vectors using sinusoids

    $$
    p_i = 
    \begin{cases}
    cos(i / 10000^{2 * j / d}), & \text{if }i \text{ is even}  \\
    sin(i / 10000^{2 * j / d}), & \text{if }i \text{ is odd}
    \end{cases}
    $$

    Where:

    *   *i* is the index in the sequence
    *   *j* represents the dimension 
*   **Equation 2:** Self-Attention with Relative Position Encodings

    $$z_i =  \sum_{j=1}^n \alpha_{ij} (V_j +  W_{ij}^V)$$

    Where:

    *   &#x20;$z_i$ is the output for the *i*-th word
    *   $V_j$ represents the value vector of the *j*-th word
    *   $W_{ij}^V$ is a matrix that stores relative position embeddings
    *   $\alpha_{ij}$ are the attention weights, calculated similarly to standard self-attention

        $$\alpha_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^n exp(e_{ik})}$$
        $$e_{ij} = \frac{Q_i^T (K_j + W_{ij}^K)}{\sqrt{d_k}}$$
        
    *   $Q_i$ represents the query vector for the *i*-th word 
    *   $K_j$ represents the key vector for the *j*-th word
    *   $W_{ij}^K$ represents a matrix that stores the relative position embeddings.
    *   $d_k$ is the dimensionality of the key vectors.

*   **Equation 3:** Multi-Head Attention

    $$output^l =  softmax\left( \frac{XQ^l {K^l}^T X^T}{\sqrt{d_k}} \right) * XV^l, \text{ where }  output^l \in R^{d / h}$$

    Where:

    *   $Q^l$, $K^l$, $V^l \in R^{d \times d/h}$, where *h* is the number of attention heads, and *l* ranges from 1 to *h*.
    *   Each attention head performs attention independently.
    *   The outputs of all the heads are combined:

        $$output = Y[output^1; ... ; output^h], \text{ where } \in R^{d \times d} $$

The sources highlight that the Transformer architecture, with its self-attention mechanism, has revolutionized NLP. **Residual connections, Layer Normalization, and Scaled Dot-Product Attention are crucial techniques for training stable and efficient Transformer models.**

The sources also discuss **extensions to self-attention like multi-head attention and relative position encodings, which enhance the model's ability to capture complex relationships between words in a sentence**.  

The conversation history illustrates how attention is a fundamental mechanism for sequence-to-sequence models, and the Transformer architecture employs several variations of this mechanism.

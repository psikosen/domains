# Sparse Autoencoder: The File Allocation Table

## 1. Conceptual Mapping: SAE ↔ File System

| File System Concept | SAE Equivalent | Mathematical Object |
|---------------------|----------------|---------------------|
| **File** | Activated feature | Sparse vector h where h_i ≈ 1 |
| **Directory** | Feature cluster | Subspace of correlated activations |
| **Inode** | Dictionary element | Column of W_dec (d_i ∈ ℝ^d) |
| **Block allocation** | Activation pattern | Support set {i : h_i > 0} |
| **Disk sector** | Dictionary index | Integer i ∈ {1...D} |
| **Free space** | Unused features | Indices where max_samples(h_i) < ε |

---

## 2. Architecture Specification

### 2.1 Core Parameters

```python
# Base latent space (from LLM)
d = 4096                    # LLM hidden dimension

# Sparse dictionary
D = 1_000_000              # Number of "slots" (inodes)
k = 10                     # Average sparsity (files per block)

# Training
batch_size = 256
learning_rate = 1e-4
lambda_sparse = 0.01       # L1 penalty coefficient
```

### 2.2 Network Architecture

```python
class SparseAutoencoder:
    """
    Maps dense LLM states to sparse, interpretable features.

    Think of it as a compression algorithm that:
    - Expands to larger space (d → D)
    - But keeps only k << D entries active
    - Each active entry = one "concept file"
    """

    def __init__(self, d_input=4096, d_sparse=1_000_000):
        self.d = d_input
        self.D = d_sparse

        # Encoder: ℝ^d → ℝ^D
        self.W_enc = glorot_uniform((d_sparse, d_input))
        self.b_enc = zeros(d_sparse)

        # Decoder: ℝ^D → ℝ^d
        # Constraint: W_dec = W_enc.T (tied weights for efficiency)
        self.W_dec = self.W_enc.T
        self.b_dec = zeros(d_input)

    def encode(self, x):
        """
        Input: dense LLM state x ∈ ℝ^d
        Output: sparse code h ∈ ℝ^D with ||h||_0 ≈ k
        """
        # Linear projection + bias
        z = self.W_enc @ x + self.b_enc  # ℝ^D

        # ReLU activation (enforces non-negativity + sparsity)
        h = ReLU(z)

        # Optional: top-k sparsification
        # h = keep_top_k(h, k=10)

        return h

    def decode(self, h):
        """
        Input: sparse code h ∈ ℝ^D
        Output: reconstructed state x̂ ∈ ℝ^d
        """
        x_hat = self.W_dec @ h + self.b_dec
        return x_hat

    def forward(self, x):
        """Full encode-decode cycle"""
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
```

### 2.3 Loss Function with Auxiliary Terms

```python
def loss(x, x_hat, h, lambda_sparse=0.01, lambda_diversity=0.001):
    """
    Multi-objective loss for SAE training.
    """
    # 1. Reconstruction: how well do we recover input?
    L_recon = torch.norm(x - x_hat, p=2) ** 2

    # 2. Sparsity: encourage few active features
    L_sparse = torch.norm(h, p=1)

    # 3. Diversity: prevent feature collapse (all h_i learn same thing)
    # Encourage W_enc columns to be orthogonal
    W_enc_normalized = F.normalize(W_enc, dim=1)
    gram = W_enc_normalized @ W_enc_normalized.T
    L_diversity = torch.norm(gram - torch.eye(D), p='fro')

    # 4. Dead feature penalty: encourage all features to activate sometimes
    # (computed over batch)
    activation_rate = (h > 0).float().mean(dim=0)  # shape: (D,)
    target_rate = k / D  # expect k active out of D
    L_dead = torch.norm(activation_rate - target_rate, p=2)

    return L_recon + lambda_sparse * L_sparse + lambda_diversity * L_diversity + 0.1 * L_dead
```

---

## 3. The "File Allocation Table" (FAT)

### 3.1 Feature Registry

Each of D features has metadata:

```python
class FeatureRegistry:
    """
    Tracks what each sparse feature represents.
    Like /etc/fstab or the inode table in ext4.
    """

    def __init__(self, D=1_000_000):
        self.D = D
        # Feature metadata
        self.allocated = [False] * D       # Is this feature in use?
        self.labels = [None] * D           # Human-readable label
        self.activation_freq = [0.0] * D   # How often does it fire?
        self.creation_time = [None] * D    # When was it first learned?
        self.last_access = [None] * D      # Last time it was accessed

        # Feature clustering (directory structure)
        self.parent_dir = [None] * D       # Which cluster does this belong to?
        self.children = {}                 # Hierarchical structure

    def allocate(self, feature_id, label=None):
        """Mark feature as in-use (like allocating an inode)"""
        self.allocated[feature_id] = True
        self.labels[feature_id] = label
        self.creation_time[feature_id] = time.now()

    def free(self, feature_id):
        """Release feature (like freeing a block)"""
        self.allocated[feature_id] = False

    def find_free_slot(self):
        """Find next available feature (like finding free inode)"""
        for i in range(self.D):
            if not self.allocated[i]:
                return i
        raise MemoryError("No free features! Sparse space exhausted.")

    def defragment(self):
        """
        Consolidate sparse features that activate together.
        Like disk defragmentation.
        """
        # Compute correlation matrix of activations
        # Merge highly correlated features
        pass  # TODO
```

### 3.2 Addressing Scheme

```python
class LatentFileSystem:
    """
    Combines SAE with registry for full filesystem.
    """

    def __init__(self):
        self.sae = SparseAutoencoder(d_input=4096, d_sparse=1_000_000)
        self.registry = FeatureRegistry(D=1_000_000)

    def write(self, concept_name, dense_state):
        """
        Store a new concept in latent space.

        Example:
            fs.write("Rust borrow checker", embedding_of_rust_concept)
        """
        # Encode to sparse representation
        sparse_code = self.sae.encode(dense_state)

        # Find which features activated
        active_features = torch.where(sparse_code > threshold)[0]

        # Allocate if not already allocated
        for feat_id in active_features:
            if not self.registry.allocated[feat_id]:
                self.registry.allocate(feat_id, label=concept_name)

        # Store the pattern
        self.memory[concept_name] = {
            'sparse_code': sparse_code,
            'active_features': active_features,
            'dense_backup': dense_state  # In case of retrieval failure
        }

    def read(self, query):
        """
        Retrieve concept by name or by similarity.

        Args:
            query: string (exact match) or vector (similarity search)
        """
        if isinstance(query, str):
            # Direct lookup
            return self.memory.get(query)
        else:
            # Similarity search
            query_sparse = self.sae.encode(query)

            # Find most similar stored pattern
            best_match = None
            best_score = -inf
            for name, data in self.memory.items():
                score = cosine_similarity(query_sparse, data['sparse_code'])
                if score > best_score:
                    best_score = score
                    best_match = name

            return self.memory[best_match]

    def ls(self, directory=None):
        """
        List all concepts in a given cluster.
        Like 'ls' command.
        """
        if directory is None:
            # List all allocated features
            return [self.registry.labels[i] for i in range(self.registry.D)
                    if self.registry.allocated[i]]
        else:
            # List features in specific cluster
            return [self.registry.labels[child]
                    for child in self.registry.children.get(directory, [])]

    def du(self):
        """
        Disk usage: how much of sparse space is allocated?
        """
        allocated = sum(self.registry.allocated)
        return f"{allocated} / {self.registry.D} features allocated ({100*allocated/self.registry.D:.2f}%)"
```

---

## 4. Hierarchical Feature Clustering (Directory Tree)

### 4.1 Automatic Clustering via Co-activation

```python
def build_directory_tree(activation_history, threshold=0.5):
    """
    Build hierarchical structure by clustering frequently co-activating features.

    Args:
        activation_history: (num_samples, D) binary matrix
        threshold: correlation threshold for grouping

    Returns:
        Tree structure representing concept hierarchy
    """
    # Compute feature co-activation matrix
    # C[i,j] = how often features i and j activate together
    C = (activation_history.T @ activation_history) / num_samples

    # Hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster

    # Convert correlation to distance
    distance_matrix = 1 - C

    # Perform clustering
    Z = linkage(distance_matrix, method='ward')
    cluster_ids = fcluster(Z, t=threshold, criterion='distance')

    # Build tree
    tree = {}
    for feat_id, cluster_id in enumerate(cluster_ids):
        if cluster_id not in tree:
            tree[cluster_id] = []
        tree[cluster_id].append(feat_id)

    return tree
```

### 4.2 Example Hierarchy

```
/                           (root)
├── /programming            (cluster 1)
│   ├── /programming/python (cluster 1.1)
│   │   ├── feature_42      "list comprehension"
│   │   ├── feature_137     "decorator syntax"
│   │   └── feature_891     "async/await"
│   ├── /programming/rust   (cluster 1.2)
│   │   ├── feature_234     "borrow checker"
│   │   ├── feature_567     "lifetime annotations"
│   │   └── feature_912     "trait bounds"
│   └── /programming/cpp    (cluster 1.3)
│       └── feature_445     "RAII pattern"
├── /math                   (cluster 2)
│   ├── /math/linear_algebra
│   │   ├── feature_88      "eigenvalue"
│   │   └── feature_199     "singular value decomposition"
│   └── /math/calculus
│       └── feature_303     "Riemann integral"
└── /philosophy             (cluster 3)
    └── feature_666         "Cartesian dualism"
```

---

## 5. Efficient Indexing Structures

### 5.1 The Challenge

With D = 1,000,000 features, linear search is too slow:
```
O(D) comparisons per query
```

### 5.2 Locality-Sensitive Hashing (LSH)

```python
class LSHIndex:
    """
    Fast approximate nearest neighbor search for sparse codes.
    """

    def __init__(self, D, num_tables=10, hash_size=20):
        self.D = D
        self.num_tables = num_tables
        self.hash_size = hash_size

        # Random projection matrices
        self.hash_functions = [
            randn(hash_size, D) for _ in range(num_tables)
        ]

        # Hash tables
        self.tables = [defaultdict(list) for _ in range(num_tables)]

    def insert(self, feature_id, sparse_code):
        """Add sparse code to index"""
        for i, h_func in enumerate(self.hash_functions):
            # Compute hash
            hash_val = tuple((h_func @ sparse_code > 0).astype(int))

            # Store in table
            self.tables[i][hash_val].append(feature_id)

    def query(self, sparse_code, k=10):
        """
        Find k nearest neighbors.

        Complexity: O(log D) average case (vs O(D) brute force)
        """
        candidates = set()

        # Query all hash tables
        for i, h_func in enumerate(self.hash_functions):
            hash_val = tuple((h_func @ sparse_code > 0).astype(int))
            candidates.update(self.tables[i][hash_val])

        # Refine candidates with exact similarity
        scores = []
        for feat_id in candidates:
            score = cosine_similarity(sparse_code, self.get_code(feat_id))
            scores.append((score, feat_id))

        # Return top-k
        scores.sort(reverse=True)
        return [feat_id for _, feat_id in scores[:k]]
```

---

## 6. Training Protocol

### 6.1 Two-Stage Training

**Stage 1: Unsupervised Pre-training**
```python
# Train SAE on random LLM hidden states
for epoch in range(100):
    for batch in dataloader(llm_activations):
        x_hat, h = sae.forward(batch)
        loss = compute_loss(batch, x_hat, h)
        loss.backward()
        optimizer.step()
```

**Stage 2: Interpretability Refinement**
```python
# Fine-tune with human labels to align features with concepts
# Use techniques from "Anthropic's dictionary learning" paper

for concept, examples in labeled_dataset:
    # Get sparse codes for all examples
    sparse_codes = [sae.encode(ex) for ex in examples]

    # Find most consistently active feature
    avg_activation = mean(sparse_codes, axis=0)
    top_feature = argmax(avg_activation)

    # Label it
    registry.allocate(top_feature, label=concept)
```

### 6.2 Curriculum: From General to Specific

```
Week 1: Train on broad categories (programming, math, philosophy)
Week 2: Train on subcategories (Python, Rust, C++)
Week 3: Train on specific patterns (list comprehension, borrow checker)
Week 4: Train on nuances (when to use Vec vs slice in Rust)
```

---

## 7. Maintenance Operations

### 7.1 Garbage Collection

```python
def garbage_collect(registry, activation_history, threshold=1e-6):
    """
    Free features that are never activated (dead code).
    """
    for feat_id in range(registry.D):
        if registry.allocated[feat_id]:
            usage = activation_history[:, feat_id].mean()
            if usage < threshold:
                print(f"Freeing dead feature {feat_id}: {registry.labels[feat_id]}")
                registry.free(feat_id)
```

### 7.2 Defragmentation

```python
def defragment(sae, registry):
    """
    Consolidate related features to improve cache locality.
    """
    # Re-order W_enc columns so that frequently co-activating features are adjacent
    # This improves memory access patterns during inference

    # Build adjacency matrix of co-activations
    # Use traveling salesman heuristic to find good ordering
    # Permute W_enc columns
    pass  # Non-trivial optimization problem
```

---

## 8. Performance Metrics

### 8.1 Key Metrics

```python
def evaluate_sae(sae, test_data):
    """
    Evaluate SAE quality.
    """
    metrics = {}

    # 1. Reconstruction fidelity
    x_hat, h = sae.forward(test_data)
    metrics['mse'] = ((test_data - x_hat) ** 2).mean()
    metrics['cosine_sim'] = cosine_similarity(test_data, x_hat).mean()

    # 2. Sparsity
    metrics['avg_sparsity'] = (h > 0).sum(dim=1).float().mean()
    metrics['sparsity_std'] = (h > 0).sum(dim=1).float().std()

    # 3. Feature utilization
    metrics['dead_features'] = (h.max(dim=0)[0] == 0).sum()
    metrics['utilization'] = 1 - metrics['dead_features'] / sae.D

    # 4. Interpretability (requires human eval)
    # Sample 100 random features, show top activating examples
    # Ask humans: "Does this feature represent a coherent concept?"
    # metrics['interpretability_score'] = human_eval()

    return metrics
```

### 8.2 Expected Performance

```
MSE: < 0.01 (good reconstruction)
Cosine similarity: > 0.95 (preserves direction)
Avg sparsity: 10-20 active features
Utilization: > 80% (few dead features)
Interpretability: > 70% features have clear meaning
```

---

## Summary: SAE as a File System

| Operation | File System | SAE |
|-----------|-------------|-----|
| **Create file** | `touch foo.txt` | Allocate new feature index |
| **Write file** | `echo "data" > foo.txt` | Activate feature with high value |
| **Read file** | `cat foo.txt` | Decode sparse code to dense state |
| **List files** | `ls /dir` | List features in cluster |
| **Disk usage** | `du -h` | Count allocated features |
| **Defrag** | `defrag C:` | Re-order W_enc for locality |
| **Garbage collect** | `rm unused` | Free dead features |

**Next:** Use this SAE to implement Read/Write heads for the Latent OS.

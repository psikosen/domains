# Testing Strategy for Latent Space OS

## Overview

This directory contains a comprehensive test suite to validate the mathematical foundations and implementation of the Latent Space OS architecture **before** building the full system.

**Philosophy:** Test the math first, then build with confidence.

---

## Test Suite Structure

### Level 1: Mathematical Validation (`test_math_validation.py`)

Tests that validate theoretical claims from `mathematical_foundations.md`.

| Test | What It Validates | Key Metric | Pass Criteria |
|------|------------------|------------|---------------|
| **Holographic Capacity** | Can store N vectors in d dims | Retrieval accuracy | >90% for N < d/log(N) |
| **Sparse Autoencoder** | SAE reconstruction quality | MSE, sparsity | MSE < 0.01, k=10-20 |
| **Content-Addressable Memory** | Similarity-based retrieval | Precision | >95% |
| **Vector Arithmetic** | Semantic analogies work | Cosine similarity | >0.7 |
| **Attention Read/Write** | Memory operations preserve info | Retrieval similarity | >90% |
| **Gradient Stability** | Convergence to target | Energy decrease | Monotonic decrease |
| **Hierarchical Indexing** | Speed improvement | Query time | O(√N) vs O(N) |

**Run:** `python test_math_validation.py`

**Expected Runtime:** ~2 minutes

---

### Level 2: Component Integration (`test_component_integration.py`)

Tests how components work together as a system.

| Test | What It Validates | Integration Point | Pass Criteria |
|------|------------------|-------------------|---------------|
| **Read-Write Consistency** | Write then read returns same | Read ↔ Write | >80% similarity |
| **Multi-Concept Storage** | Multiple items without interference | Memory ↔ Heads | >70% avg retrieval |
| **Atomic Updates** | Read-Modify-Write works | All components | Modified > Initial |
| **Thought Chains** | Iterative refinement converges | Think ↔ Memory | Positive improvement |
| **Memory Persistence** | Save/load preserves state | Serialization | >80% after reload |
| **Kernel Protection** | Writes to kernel blocked | Safety layer | 100% blocking |
| **Sparse Activation** | Related concepts share features | SAE ↔ Concepts | More overlap for related |

**Run:** `python test_component_integration.py`

**Expected Runtime:** ~3 minutes

---

### Level 3: Full Prototype (`prototype.py`)

End-to-end system test with real LLM.

**Experiments:**
1. Memory persistence across queries
2. Multi-step reasoning chains
3. Learning from examples
4. Session save/restore

**Run:** `python prototype.py`

**Expected Runtime:** ~10 minutes (downloads GPT-2 first time)

---

## Quick Start

### Run All Tests

```bash
# Make executable
chmod +x run_tests.sh

# Run complete test suite
./run_tests.sh
```

This will:
1. Check dependencies
2. Run mathematical validation (7 tests)
3. Run component integration (7 tests)
4. Print comprehensive summary

### Run Individual Test Files

```bash
# Mathematical validation only
python test_math_validation.py

# Component integration only
python test_component_integration.py

# Full prototype
python prototype.py
```

---

## What Each Test Proves

### Mathematical Validation Tests

#### 1. Holographic Capacity Test
**Proves:** The claim that we can store d/log(N) vectors in d-dimensional space.

**How:**
- Generate N random unit vectors
- Store as superposition (sum)
- Attempt retrieval via cosine similarity
- Measure accuracy vs N

**Expected Result:**
```
N=10   → 100% accuracy ✓
N=50   → 100% accuracy ✓
N=100  → 100% accuracy ✓
N=200  →  95% accuracy ✓
N=500  →  80% accuracy (approaching capacity limit)
```

#### 2. Sparse Autoencoder Test
**Proves:** SAE can compress to sparse representation with low reconstruction error.

**How:**
- Train simple SAE (256 → 1000 → 256)
- Measure reconstruction MSE
- Count active features (sparsity)
- Check cosine similarity

**Expected Result:**
```
Reconstruction MSE: < 0.01 ✓
Average sparsity: 10-20 features ✓
Cosine similarity: > 0.95 ✓
```

#### 3. Content-Addressable Memory Test
**Proves:** Can retrieve items by similarity with high precision.

**How:**
- Store 1000 random vectors
- Query with noisy version of each
- Measure retrieval precision

**Expected Result:**
```
Precision: > 95% ✓
(Even with 10% noise added to queries)
```

#### 4. Vector Arithmetic Test
**Proves:** Semantic relationships encoded in vector space.

**How:**
- Create structured vectors (king, queen, man, woman)
- Test analogy: king - man + woman ≈ queen
- Measure similarity

**Expected Result:**
```
Similarity to queen: > 0.7 ✓
Much higher than similarity to random vector
```

#### 5. Attention Read/Write Test
**Proves:** Attention mechanism preserves information.

**How:**
- Write fact to memory using attention
- Read back using same query
- Measure retrieval similarity

**Expected Result:**
```
Retrieval similarity: > 0.9 ✓
Attention peaked on correct slot
```

#### 6. Gradient Stability Test
**Proves:** Gradient descent converges to stable states.

**How:**
- Define energy function (distance to target)
- Run gradient descent
- Check energy decreases monotonically

**Expected Result:**
```
Energy decreases: ✓
Converges to target: ✓
Lyapunov stability satisfied
```

#### 7. Hierarchical Indexing Test
**Proves:** Two-stage search faster than linear.

**How:**
- Build clustered index
- Compare search time: linear vs hierarchical
- Measure speedup

**Expected Result:**
```
N=1000  → 2x speedup
N=5000  → 5x speedup
N=10000 → 10x speedup
```

---

### Component Integration Tests

#### 1. Read-Write Consistency
**Validates:** Memory operations are reversible.

**Critical For:** Ensuring information isn't lost during storage.

#### 2. Multi-Concept Storage
**Validates:** Can handle multiple items without catastrophic interference.

**Critical For:** Scaling beyond toy examples.

#### 3. Read-Modify-Write
**Validates:** Can update existing memories atomically.

**Critical For:** Learning and refinement.

#### 4. Thought Chains
**Validates:** Iterative refinement converges.

**Critical For:** Multi-step reasoning.

#### 5. Memory Persistence
**Validates:** State survives serialization.

**Critical For:** Sessions and deployment.

#### 6. Kernel Protection
**Validates:** Safety mechanisms work.

**Critical For:** Production use.

#### 7. Sparse Activation
**Validates:** SAE features capture semantic similarity.

**Critical For:** Interpretability.

---

## Success Criteria

### Phase 1: Mathematical Validation ✓

All 7 tests must pass with metrics above thresholds.

- [x] Holographic capacity follows d/log(N) bound
- [x] SAE reconstruction error < 0.01
- [x] Content retrieval precision > 95%
- [x] Vector arithmetic preserves analogies
- [x] Attention R/W preserves information
- [x] Gradient descent converges
- [x] Hierarchical indexing provides speedup

### Phase 2: Component Integration ✓

All 7 integration tests must pass.

- [x] Read-write cycle consistency
- [x] Multi-concept storage without interference
- [x] Atomic updates work correctly
- [x] Thought chains improve over time
- [x] Memory persists across save/load
- [x] Kernel protection blocks writes
- [x] Sparse features cluster semantically

### Phase 3: Full Prototype

4 experiments show expected behavior.

- [ ] Facts persist across queries
- [ ] Multi-hop reasoning works
- [ ] Learns patterns from examples
- [ ] Session state preserved

---

## Interpreting Results

### Green Flags ✓

- **High retrieval accuracy** (>80%) → Memory working
- **Positive improvement** in thought chains → Reasoning working
- **Low reconstruction error** (<0.01) → SAE working
- **Speedup with hierarchical search** → Indexing working

### Yellow Flags ⚠

- **Moderate retrieval** (60-80%) → Tune parameters
- **Slow convergence** (>50 steps) → Adjust learning rate
- **High sparsity** (>30 features) → Increase L1 penalty

### Red Flags ✗

- **Low retrieval** (<60%) → Architecture issue
- **Divergence** (energy increases) → Instability
- **Zero improvement** in chains → Not learning
- **Kernel writes succeed** → Safety broken

---

## Debugging Guide

### Test Fails: Holographic Capacity

**Symptom:** Accuracy drops below 90% for small N

**Likely Cause:** Vectors not normalized or dimension too low

**Fix:**
```python
vectors = F.normalize(vectors, dim=1)  # Ensure unit vectors
d = 1024  # Increase dimension if needed
```

### Test Fails: SAE Reconstruction

**Symptom:** MSE > 0.01 or sparsity wrong

**Likely Cause:** Training not converged or wrong λ

**Fix:**
```python
# Train longer
epochs = 1000  # Increase from 500

# Tune sparsity penalty
lambda_sparse = 0.01  # Increase to enforce more sparsity
```

### Test Fails: Thought Chain

**Symptom:** No improvement or divergence

**Likely Cause:** Learning rate too high or target not in memory

**Fix:**
```python
learning_rate = 0.01  # Decrease from 0.1

# Ensure target is stored
memory = write_head(target, memory)
```

---

## Performance Benchmarks

### Expected Runtimes (CPU)

| Test Suite | Runtime | Tests | Operations |
|-----------|---------|-------|-----------|
| Math Validation | ~2 min | 7 | 100K+ ops |
| Integration | ~3 min | 7 | 50K+ ops |
| Full Prototype | ~10 min | 4 | 1K+ LLM calls |

### Expected Runtimes (GPU)

| Test Suite | Runtime | Speedup |
|-----------|---------|---------|
| Math Validation | ~30 sec | 4x |
| Integration | ~45 sec | 4x |
| Full Prototype | ~3 min | 3x |

---

## Next Steps After Tests Pass

1. **Analyze Results**
   - Review all metrics
   - Identify bottlenecks
   - Document findings

2. **Tune Hyperparameters**
   - Memory size
   - Sparsity target
   - Learning rates
   - Attention temperature

3. **Scale Up**
   - Larger memory (100 → 1000 slots)
   - Larger SAE (1K → 100K features)
   - Bigger LLM (GPT-2 → LLaMA)

4. **Add Features**
   - Hierarchical memory (Phase 3)
   - Continuous thought (Phase 4)
   - Safety monitoring (Phase 5)

---

## Continuous Integration

To run tests automatically:

```bash
# Add to CI pipeline
.github/workflows/test.yml:

name: Test Latent OS
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install torch numpy
      - run: ./run_tests.sh
```

---

## Troubleshooting

### Import Errors

```bash
pip install torch numpy matplotlib
```

### Memory Errors

Reduce test sizes:
```python
# In test files
N = 100  # Instead of 1000
d = 128  # Instead of 512
```

### Slow Tests

Use smaller models:
```python
# In prototype.py
model = 'distilgpt2'  # Instead of 'gpt2'
```

---

## References

- Mathematical foundations: `mathematical_foundations.md`
- System architecture: `system_integration.md`
- Implementation guide: `implementation_roadmap.md`

---

**Remember:** These tests validate the theory. If they pass, the architecture is sound. Build with confidence! 🚀

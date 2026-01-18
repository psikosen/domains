"""
prototype.py

Minimal working prototype of Latent Space OS.

This is the actual code you can run to start experimenting.

Usage:
    python prototype.py

Requirements:
    pip install torch transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer
from typing import List, Tuple, Optional
import numpy as np


# =============================================================================
# CORE COMPONENTS
# =============================================================================

class ReadHead(nn.Module):
    """
    Content-addressable memory read.

    Given current state, retrieves relevant information from memory.
    """

    def __init__(self, hidden_dim: int, memory_size: int):
        super().__init__()
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, state: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (hidden_dim,) - current mental state
            memory: (memory_size, hidden_dim) - stored knowledge

        Returns:
            read_value: (hidden_dim,) - retrieved information
            attention: (memory_size,) - where we read from
        """
        # Generate query
        query = self.query_projection(state)  # (hidden_dim,)

        # Compute similarity to all memory slots
        similarity = torch.matmul(query, memory.T)  # (memory_size,)

        # Softmax to get attention weights
        attention = F.softmax(similarity / self.temperature, dim=0)  # (memory_size,)

        # Weighted sum of memory
        read_value = torch.matmul(attention, memory)  # (hidden_dim,)

        return read_value, attention


class WriteHead(nn.Module):
    """
    Memory write operation.

    Updates memory based on current state.
    """

    def __init__(self, hidden_dim: int, memory_size: int):
        super().__init__()
        # What to write
        self.content_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Where to write
        self.address_net = nn.Linear(hidden_dim, memory_size)

        # How much to write (blend strength)
        self.strength_net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (hidden_dim,)
            memory: (memory_size, hidden_dim)

        Returns:
            updated_memory: (memory_size, hidden_dim)
        """
        # Generate write content
        content = self.content_net(state)  # (hidden_dim,)

        # Generate write address (soft)
        write_attention = F.softmax(self.address_net(state), dim=0)  # (memory_size,)

        # Generate write strength
        strength = self.strength_net(state)  # (1,)

        # Perform write (blend new content with existing memory)
        erase = 1 - strength * write_attention.unsqueeze(-1)  # (memory_size, 1)
        add = strength * write_attention.unsqueeze(-1) * content.unsqueeze(0)  # (memory_size, hidden_dim)

        updated_memory = memory * erase + add

        return updated_memory


class MinimalLatentOS:
    """
    Simplest possible Latent OS implementation.

    Features:
    - Fixed-size dense memory
    - Read and write heads
    - Multi-step reasoning in latent space
    """

    def __init__(
        self,
        memory_size: int = 100,
        hidden_dim: int = 768,
        max_think_steps: int = 5,
        device: str = 'cpu'
    ):
        self.device = device
        self.max_think_steps = max_think_steps

        # Language model for encoding/decoding
        print("Loading language model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2Model.from_pretrained('gpt2').to(device)
        self.lm.eval()  # Freeze LM
        self.hidden_dim = self.lm.config.hidden_size

        # Memory
        self.memory = nn.Parameter(
            torch.randn(memory_size, self.hidden_dim, device=device) * 0.01
        )

        # Read/Write heads
        self.read_head = ReadHead(self.hidden_dim, memory_size).to(device)
        self.write_head = WriteHead(self.hidden_dim, memory_size).to(device)

        # Thinking module (simple transformer layer)
        self.think_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
            device=device
        )

        print(f"✓ Latent OS initialized:")
        print(f"  - Memory: {memory_size} slots × {self.hidden_dim} dims")
        print(f"  - Max thinking steps: {max_think_steps}")

    def encode(self, text: str) -> torch.Tensor:
        """
        Text → Latent vector

        Uses last token's hidden state as representation.
        """
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.lm(**tokens)

        # Use last token's hidden state
        hidden_state = outputs.last_hidden_state[0, -1, :]  # (hidden_dim,)

        return hidden_state

    def decode(self, state: torch.Tensor, max_length: int = 20) -> str:
        """
        Latent vector → Text

        Simplified: Just finds nearest token in vocabulary.
        (In production, would use full autoregressive decoding)
        """
        # Project state to vocabulary space
        with torch.no_grad():
            # Use LM's output projection
            logits = torch.matmul(state, self.lm.wte.weight.T)  # (vocab_size,)

            # Sample top token
            token_id = logits.argmax().item()

        return self.tokenizer.decode([token_id])

    def think_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        One step of reasoning.

        Process:
        1. Read from memory
        2. Combine with current state
        3. Think (transform)
        4. Write back to memory
        """
        # 1. Read
        read_value, read_attention = self.read_head(state, self.memory.data)

        # 2. Combine
        combined = state + read_value

        # 3. Think
        thought = self.think_layer(combined.unsqueeze(0)).squeeze(0)

        # 4. Write
        self.memory.data = self.write_head(thought, self.memory.data)

        return thought

    def query(self, text: str, verbose: bool = False) -> str:
        """
        Main interface: text in, text out.

        Args:
            text: Natural language query
            verbose: Print thinking process

        Returns:
            answer: Natural language answer
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {text}")
            print(f"{'='*60}")

        # Encode to latent space
        state = self.encode(text)
        if verbose:
            print(f"Encoded to latent space: ||state|| = {state.norm():.2f}")

        # Think
        for step in range(self.max_think_steps):
            prev_norm = state.norm().item()
            state = self.think_step(state)
            new_norm = state.norm().item()

            if verbose:
                print(f"  Step {step+1}: ||state|| = {new_norm:.2f} (Δ = {new_norm - prev_norm:+.2f})")

        # Decode back to text
        answer = self.decode(state)

        if verbose:
            print(f"Answer: {answer}")
            print(f"{'='*60}\n")

        return answer

    def save(self, path: str):
        """Save OS state to disk."""
        state = {
            'memory': self.memory.data,
            'read_head': self.read_head.state_dict(),
            'write_head': self.write_head.state_dict(),
            'think_layer': self.think_layer.state_dict(),
        }
        torch.save(state, path)
        print(f"💾 Saved to {path}")

    def load(self, path: str):
        """Load OS state from disk."""
        state = torch.load(path, map_location=self.device)
        self.memory.data = state['memory']
        self.read_head.load_state_dict(state['read_head'])
        self.write_head.load_state_dict(state['write_head'])
        self.think_layer.load_state_dict(state['think_layer'])
        print(f"📂 Loaded from {path}")


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_1_memory_persistence():
    """
    Test: Does information persist in memory across queries?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Memory Persistence")
    print("="*60)

    os = MinimalLatentOS(memory_size=50, max_think_steps=3)

    # Teach it facts
    print("\n--- Teaching Phase ---")
    facts = [
        "The capital of France is Paris",
        "The sky is blue",
        "Water freezes at 0 degrees Celsius",
    ]

    for fact in facts:
        _ = os.query(fact, verbose=True)

    # Query phase
    print("\n--- Query Phase ---")
    questions = [
        "What is the capital of France?",
        "What color is the sky?",
        "At what temperature does water freeze?",
    ]

    for question in questions:
        answer = os.query(question, verbose=True)


def experiment_2_reasoning_chains():
    """
    Test: Can it perform multi-step reasoning?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Reasoning Chains")
    print("="*60)

    os = MinimalLatentOS(memory_size=100, max_think_steps=10)

    # Store intermediate facts
    print("\n--- Storing Facts ---")
    facts = [
        "Alice is taller than Bob",
        "Bob is taller than Charlie",
    ]

    for fact in facts:
        _ = os.query(fact, verbose=False)

    # Transitive reasoning
    print("\n--- Transitive Inference ---")
    question = "Who is tallest: Alice, Bob, or Charlie?"
    answer = os.query(question, verbose=True)


def experiment_3_learning_from_examples():
    """
    Test: Can it learn patterns from examples?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Learning from Examples")
    print("="*60)

    os = MinimalLatentOS(memory_size=100, max_think_steps=5)

    # Teach pattern: country → capital
    print("\n--- Training Phase ---")
    examples = [
        "France capital Paris",
        "Germany capital Berlin",
        "Italy capital Rome",
        "Spain capital Madrid",
    ]

    for example in examples:
        _ = os.query(example, verbose=False)

    # Test generalization
    print("\n--- Test Phase ---")
    test_query = "What is the capital of France?"
    answer = os.query(test_query, verbose=True)


def experiment_4_persistence_across_sessions():
    """
    Test: Can state be saved and restored?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Session Persistence")
    print("="*60)

    # Session 1: Teach
    print("\n--- Session 1: Teaching ---")
    os1 = MinimalLatentOS(memory_size=50, max_think_steps=3)

    facts = [
        "Python is a programming language",
        "Rust is memory safe",
    ]

    for fact in facts:
        _ = os1.query(fact, verbose=False)

    # Save
    os1.save('latent_os_checkpoint.pt')

    # Session 2: Restore and query
    print("\n--- Session 2: Querying (After Restart) ---")
    os2 = MinimalLatentOS(memory_size=50, max_think_steps=3)
    os2.load('latent_os_checkpoint.pt')

    question = "What is Python?"
    answer = os2.query(question, verbose=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Run all experiments.
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║              LATENT SPACE OS - PROTOTYPE                ║
    ║                                                          ║
    ║  "Treating latent space as a mutable operating system"  ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Run experiments
    experiment_1_memory_persistence()
    experiment_2_reasoning_chains()
    experiment_3_learning_from_examples()
    experiment_4_persistence_across_sessions()

    print("\n" + "="*60)
    print("All experiments complete! 🎉")
    print("="*60)
    print("""
Next steps:
1. Analyze results - did memory persist?
2. Add sparse autoencoder for scale
3. Implement hierarchical memory for speed
4. Add continuous thought chains
5. Add safety monitoring

See implementation_roadmap.md for details.
    """)


if __name__ == "__main__":
    main()

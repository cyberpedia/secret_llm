#!/usr/bin/env python3
"""
SecretLLM - Autonomous LLM Agent from Scratch
Based on "Fundamentals of Building Autonomous LLM Agents" (arXiv:2510.09244)
"""

import os
import json
import math
import random
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 512          
    n_layers: int = 8       
    n_heads: int = 8        
    n_kv_heads: int = 4     
    hidden_dim: int = 2048  
    max_seq_len: int = 2048
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Training
    batch_size: int = 4
    learning_rate: float = 3e-4
    max_steps: int = 100000
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    
    # Agentic features
    use_tools: bool = True
    memory_size: int = 1000

# ============================================================================
# TOKENIZER (Byte-Pair Encoding from scratch)
# ============================================================================

class BPETokenizer:
    """Simple BPE tokenizer implementation"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []
        
        # Initialize with bytes
        for i in range(256):
            self.vocab[bytes([i])] = i
        
        self.special_tokens = {
            '<|endoftext|>': vocab_size - 1,
            '<|im_start|>': vocab_size - 2,
            '<|im_end|>': vocab_size - 3,
            '<|tool|>': vocab_size - 4,
            '<|action|>': vocab_size - 5,
            '<|observation|>': vocab_size - 6,
            '<|thought|>': vocab_size - 7,
        }
        
        for token, idx in self.special_tokens.items():
            self.vocab[token.encode()] = idx
            self.inverse_vocab[idx] = token.encode()
    
    def train(self, texts: List[str], num_merges: int = None):
        """Train BPE on provided texts"""
        if num_merges is None:
            num_merges = self.vocab_size - 256 - len(self.special_tokens)
        
        word_freqs = defaultdict(int)
        for text in texts:
            words = text.encode('utf-8')
            word = tuple(bytes([b]) for b in words)
            word_freqs[word] += 1
        
        for i in range(num_merges):
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j+1])] += freq
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]
            new_idx = len(self.vocab)
            
            if new_idx >= self.vocab_size - len(self.special_tokens):
                break
                
            self.vocab[new_token] = new_idx
            self.merges.append(best_pair)
            
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j+1]) == best_pair:
                        new_word.append(new_token)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs
        
        for token, idx in self.vocab.items():
            self.inverse_vocab[idx] = token
        
        print(f"Trained tokenizer with {len(self.vocab)} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        for special, idx in sorted(self.special_tokens.items(), key=lambda x: -len(x[0])):
            text = text.replace(special, f"\x00{idx}\x00")
        
        parts = text.split("\x00")
        result = []
        
        for part in parts:
            if part.isdigit():
                result.append(int(part))
            else:
                word = tuple(bytes([b]) for b in part.encode('utf-8'))
                for merge in self.merges:
                    new_word = []
                    i = 0
                    while i < len(word):
                        if i < len(word) - 1 and (word[i], word[i+1]) == merge:
                            new_word.append(merge[0] + merge[1])
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = tuple(new_word)
                result.extend([self.vocab.get(t, self.vocab_size-1) for t in word])
        
        return result
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        bytes_list = []
        for idx in ids:
            if idx in self.inverse_vocab:
                token = self.inverse_vocab[idx]
                if isinstance(token, bytes):
                    bytes_list.append(token)
                else:
                    bytes_list.append(str(token).encode())
        
        try:
            return b''.join(bytes_list).decode('utf-8', errors='replace')
        except:
            return ''.join([self.inverse_vocab.get(i, b'?').decode('utf-8', errors='replace') 
                          for i in ids])
    
    def save(self, path: str):
        """Save tokenizer to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'vocab': {k.hex(): v for k, v in self.vocab.items() if isinstance(k, bytes)},
            'special_tokens': self.special_tokens,
            'merges': [(p[0].hex(), p[1].hex()) for p in self.merges]
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load tokenizer from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab = {bytes.fromhex(k): v for k, v in data['vocab'].items()}
        self.special_tokens = data['special_tokens']
        self.merges = [(bytes.fromhex(p[0]), bytes.fromhex(p[1])) for p in data['merges']]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.inverse_vocab.update({v: k.encode() for k, v in self.special_tokens.items()})

# ============================================================================
# ROPE (Rotary Position Embeddings)
# ============================================================================

class RoPE:
    """Rotary Position Embeddings"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def get_rotary_embedding(self, seq_len: int, device: torch.device):
        """Generate rotary embeddings for sequence"""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
    
    def rotate_half(self, x):
        """Rotate half the hidden dims"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary embeddings to queries and keys"""
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

# ============================================================================
# ATTENTION (Grouped Query Attention with RoPE)
# ============================================================================

class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention and RoPE"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope_theta)
        self.dropout = config.dropout
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        cos, sin = self.rope.get_rotary_embedding(seqlen, x.device)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        
        xq, xk = self.rope.apply_rotary_pos_emb(xq, xk, cos, sin)
        
        keys = xk.repeat_interleave(self.n_rep, dim=2)
        values = xv.repeat_interleave(self.n_rep, dim=2)
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# ============================================================================
# FEED-FORWARD (SwiGLU)
# ============================================================================

class FeedForward(nn.Module):
    """SwiGLU feed-forward network"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# ============================================================================
# MAIN TRANSFORMER MODEL
# ============================================================================

class SecretLLM(nn.Module):
    """Complete LLM with agentic capabilities"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        self.output.weight = self.tok_embeddings.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool()
        mask = mask.to(tokens.device)
        mask = torch.where(mask, float('-inf'), 0.0)
        mask = mask[None, None, :, :]
        
        for layer in self.layers:
            h = layer(h, mask)
        
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    def generate(self, tokens: torch.Tensor, max_new: int = 256, 
                 temperature: float = 0.8, top_p: float = 0.9):
        """Generate text autoregressively"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new):
                logits, _ = self(tokens)
                logits = logits[:, -1, :] / temperature
                
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                tokens = torch.cat([tokens, next_token], dim=1)
                
                if next_token.item() == self.config.vocab_size - 1:
                    break
        
        return tokens

# ============================================================================
# AGENTIC SYSTEMS
# ============================================================================

class AgentMemory:
    """Long-term and short-term memory system"""
    
    def __init__(self, max_size: int = 1000):
        self.short_term = []
        self.long_term = []
        self.max_size = max_size
        self.embeddings = {}
        
    def add_short_term(self, item: str):
        self.short_term.append(item)
        if len(self.short_term) > 10:
            self.short_term.pop(0)
    
    def add_long_term(self, item: str, importance: float = 1.0):
        self.long_term.append((item, importance))
        if len(self.long_term) > self.max_size:
            self.long_term.sort(key=lambda x: x[1])
            self.long_term.pop(0)
    
    def get_context(self, query: str = None, top_k: int = 5) -> str:
        context = "Recent context:\n" + "\n".join(self.short_term[-5:])
        
        if query and self.long_term:
            relevant = [item for item, _ in self.long_term 
                       if any(word in item.lower() for word in query.lower().split())]
            if relevant:
                context += "\n\nRelevant memories:\n" + "\n".join(relevant[:top_k])
        
        return context
    
    def clear_short_term(self):
        self.short_term = []

class ToolExecutor:
    """Action system - executes tools and code"""
    
    def __init__(self):
        self.tools = {
            'python': self.execute_python,
            'search': self.mock_search,
            'file_read': self.read_file,
            'file_write': self.write_file,
            'calculate': self.calculate,
        }
        self.allowed_paths = ['/tmp', './data', './workspace']
        
    def execute_python(self, code: str) -> str:
        import io
        import sys
        
        safe_globals = {
            '__builtins__': {
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'str': str, 'int': int, 'float': float, 'list': list,
                'dict': dict, 'tuple': tuple, 'set': set, 'print': print,
                'open': open, 'isinstance': isinstance, 'type': type,
                'Exception': Exception, 'True': True, 'False': False,
                'None': None, 'round': round, 'pow': pow, 'divmod': divmod,
            }
        }
        
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            exec(code, safe_globals, {})
            output = stdout.getvalue()
            error = stderr.getvalue()
            return output if output else (error if error else "Executed successfully")
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
    
    def mock_search(self, query: str) -> str:
        return f"[Search results for: {query}]\n- Local knowledge base searched\n- No external connection available"
    
    def read_file(self, path: str) -> str:
        if not any(path.startswith(p) for p in self.allowed_paths):
            return f"Error: Path {path} not allowed. Use: {self.allowed_paths}"
        try:
            with open(path, 'r') as f:
                return f.read()[:10000]
        except Exception as e:
            return f"Error reading file: {e}"
    
    def write_file(self, path: str, content: str) -> str:
        if not any(path.startswith(p) for p in self.allowed_paths):
            return f"Error: Path {path} not allowed"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    def calculate(self, expression: str) -> str:
        try:
            if not re.match(r'^[\d\+\-\*\/\(\)\.\s\%\*\*]+$', expression):
                return "Error: Invalid characters in expression"
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"
    
    def execute(self, tool_name: str, params: Dict) -> str:
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        return self.tools[tool_name](**params)

class ReasoningEngine:
    """Chain-of-Thought and Tree-of-Thought reasoning"""
    
    def __init__(self, model: SecretLLM, tokenizer: BPETokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def chain_of_thought(self, problem: str, max_steps: int = 5) -> str:
        prompt = f"<|thought|>Let's solve this step by step:\n\nProblem: {problem}\n\nStep 1:"
        
        reasoning_steps = []
        for i in range(max_steps):
            tokens = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
            if torch.cuda.is_available():
                tokens = tokens.cuda()
            
            generated = self.model.generate(tokens, max_new=100, temperature=0.7)
            text = self.tokenizer.decode(generated[0].tolist())
            
            new_content = text[len(prompt):].split("\n")[0]
            reasoning_steps.append(new_content)
            
            prompt += new_content + f"\n\nStep {i+2}:"
            
            if "therefore" in new_content.lower() or "answer is" in new_content.lower():
                break
        
        return "\n".join(reasoning_steps)
    
    def reflect(self, action: str, observation: str) -> str:
        prompt = f"<|thought|>Reflection:\nAction taken: {action}\nObservation: {observation}\n\nWas this action correct? What could be improved?"
        
        tokens = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        
        generated = self.model.generate(tokens, max_new=150, temperature=0.7)
        return self.tokenizer.decode(generated[0].tolist())

class SecretAgent:
    """Complete autonomous agent"""
    
    def __init__(self, model: SecretLLM, tokenizer: BPETokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.memory = AgentMemory(config.memory_size)
        self.tools = ToolExecutor()
        self.reasoning = ReasoningEngine(model, tokenizer)
        
        self.conversation_history = []
        self.max_iterations = 10
        
    def perceive(self, input_data: str) -> Dict:
        self.memory.add_short_term(f"User: {input_data}")
        context = self.memory.get_context(input_data)
        
        return {
            'raw_input': input_data,
            'context': context,
        }
    
    def reason(self, perception: Dict) -> Dict:
        input_text = perception['raw_input']
        context = perception['context']
        
        needs_tools = any(keyword in input_text.lower() for keyword in 
                         ['calculate', 'compute', 'run', 'execute', 'file', 'search', 'code'])
        
        if needs_tools:
            plan = "use_tools"
        elif '?' in input_text or 'how' in input_text.lower() or 'why' in input_text.lower():
            plan = "chain_of_thought"
        else:
            plan = "direct_response"
        
        return {
            'plan': plan,
            'context': context,
            'input': input_text
        }
    
    def act(self, reasoning: Dict) -> Dict:
        plan = reasoning['plan']
        context = reasoning['context']
        input_text = reasoning['input']
        
        if plan == "use_tools":
            return self._act_with_tools(input_text, context)
        elif plan == "chain_of_thought":
            return self._act_with_reasoning(input_text, context)
        else:
            return self._act_direct(input_text, context)
    
    def _act_with_tools(self, input_text: str, context: str) -> Dict:
        tool_prompt = f"""<|im_start|>system
You are an AI assistant with access to tools. Available tools:
- python: Execute Python code
- calculate: Mathematical calculations  
- file_read: Read files from /tmp, ./data, ./workspace
- file_write: Write files to allowed paths
- search: Search local knowledge

Respond with tool calls in format: <|tool|>tool_name|param1=value1|param2=value2<|action|>
<|im_start|>user
{context}

User request: {input_text}
<|im_start|>assistant
I need to use a tool to help with this."""

        tokens = torch.tensor([self.tokenizer.encode(tool_prompt)], dtype=torch.long)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        
        generated = self.model.generate(tokens, max_new=100, temperature=0.7)
        response = self.tokenizer.decode(generated[0].tolist())
        
        observation = ""
        if "<|tool|>" in response:
            tool_part = response.split("<|tool|>")[1].split("<|action|>")[0]
            parts = tool_part.split("|")
            tool_name = parts[0]
            params = {}
            for part in parts[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
            
            observation = self.tools.execute(tool_name, params)
            reflection = self.reasoning.reflect(f"{tool_name}({params})", observation)
        else:
            observation = "No tool call detected"
            reflection = ""
        
        final_prompt = f"""{tool_prompt}{response}
<|observation|>{observation}
<|thought|>{reflection}
<|im_start|>assistant
Based on the tool result:"""

        tokens = torch.tensor([self.tokenizer.encode(final_prompt)], dtype=torch.long)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        
        generated = self.model.generate(tokens, max_new=300, temperature=0.8)
        final_response = self.tokenizer.decode(generated[0].tolist())
        
        return {
            'response': final_response.split("<|im_start|>assistant")[-1].strip(),
            'tool_used': tool_name if "<|tool|>" in response else None,
            'observation': observation
        }
    
    def _act_with_reasoning(self, input_text: str, context: str) -> Dict:
        reasoning = self.reasoning.chain_of_thought(input_text)
        
        prompt = f"""<|im_start|>system
You are a helpful AI assistant that explains reasoning clearly.
<|im_start|>user
{context}

Question: {input_text}

My reasoning process:
{reasoning}

Please provide the final answer based on this reasoning.
<|im_start|>assistant
"""
        
        tokens = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        
        generated = self.model.generate(tokens, max_new=300, temperature=0.8)
        response = self.tokenizer.decode(generated[0].tolist())
        
        return {
            'response': response.split("<|im_start|>assistant")[-1].strip(),
            'reasoning': reasoning
        }
    
    def _act_direct(self, input_text: str, context: str) -> Dict:
        prompt = f"""<|im_start|>system
You are a helpful AI assistant.
<|im_start|>user
{context}

{input_text}
<|im_start|>assistant
"""
        
        tokens = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        
        generated = self.model.generate(tokens, max_new=300, temperature=0.8)
        response = self.tokenizer.decode(generated[0].tolist())
        
        return {
            'response': response.split("<|im_start|>assistant")[-1].strip()
        }
    
    def run(self, user_input: str) -> str:
        perception = self.perceive(user_input)
        reasoning = self.reason(perception)
        result = self.act(reasoning)
        
        self.memory.add_short_term(f"Assistant: {result['response'][:200]}...")
        
        return result['response']
    
    def chat(self):
        print("🤖 Secret Agent LLM initialized")
        print("Commands: /reset (clear memory), /tools (list tools), /exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input == '/exit':
                    break
                elif user_input == '/reset':
                    self.memory.clear_short_term()
                    print("Memory cleared.")
                    continue
                elif user_input == '/tools':
                    print("Available tools:", list(self.tools.tools.keys()))
                    continue
                elif not user_input:
                    continue
                
                response = self.run(user_input)
                print(f"\nAgent: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")

# ============================================================================
# TRAINING INFRASTRUCTURE
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BPETokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens), max_length // 2):
                chunk = tokens[i:i + max_length]
                if len(chunk) > 10:
                    self.samples.append(chunk)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

class Trainer:
    def __init__(self, model: SecretLLM, config: ModelConfig, tokenizer: BPETokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n], 'weight_decay': 0.1},
            {'params': [p for n, p in model.named_parameters() if 'bias' in n or 'norm' in n], 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.95))
        
        self.scheduler = self.get_cosine_schedule_with_warmup(
            self.optimizer, config.warmup_steps, config.max_steps
        )
        
        self.step = 0
        self.best_loss = float('inf')
        
    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_step(self, batch_x, batch_y):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        
        self.optimizer.zero_grad()
        logits, loss = self.model(batch_x, batch_y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train(self, dataloader, save_dir: str = './checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.train()
        print(f"Training on {self.device}")
        print(f"Total steps: {self.config.max_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")
        print("-" * 50)
        
        running_loss = 0.0
        data_iter = iter(dataloader)
        
        for step in range(self.config.max_steps):
            try:
                batch_x, batch_y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_x, batch_y = next(data_iter)
            
            loss = self.train_step(batch_x, batch_y)
            running_loss += loss
            
            if step % 100 == 0:
                avg_loss = running_loss / 100
                lr = self.scheduler.get_last_lr()[0]
                print(f"Step {step}/{self.config.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                running_loss = 0.0
                
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint(f"{save_dir}/best_model.pt")
                
                if step % 1000 == 0:
                    self.save_checkpoint(f"{save_dir}/checkpoint_{step}.pt")
                    self.generate_sample()
        
        print("Training complete!")
        self.save_checkpoint(f"{save_dir}/final_model.pt")
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'config': self.config,
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        print(f"Loaded checkpoint from {path}")
    
    def generate_sample(self):
        self.model.eval()
        with torch.no_grad():
            prompt = "The secret to building AI is"
            tokens = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            generated = self.model.generate(tokens, max_new=50, temperature=0.8)
            text = self.tokenizer.decode(generated[0].tolist())
            print(f"Sample: {text[:100]}...")
        self.model.train()

class DataProcessor:
    @staticmethod
    def load_text_files(directory: str) -> List[str]:
        texts = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.txt', '.md', '.py', '.json', '.csv')):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            texts.append(f.read())
                    except Exception as e:
                        print(f"Error reading {path}: {e}")
        return texts
    
    @staticmethod
    def load_jsonl(file_path: str, text_key: str = 'text') -> List[str]:
        texts = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if isinstance(data, dict) and text_key in data:
                    texts.append(data[text_key])
                elif isinstance(data, str):
                    texts.append(data)
        return texts
    
    @staticmethod
    def create_conversation_format(system_msg: str, conversations: List[Dict]) -> List[str]:
        formatted = []
        for conv in conversations:
            text = f"<|im_start|>system\n{system_msg}<|im_start|>user\n{conv['input']}<|im_start|>assistant\n{conv['output']}<|endoftext|>"
            formatted.append(text)
        return formatted

# Make everything available at package level
__all__ = [
    'SecretLLM', 'ModelConfig', 'BPETokenizer', 'SecretAgent',
    'Trainer', 'TextDataset', 'DataProcessor', 'ToolExecutor',
    'AgentMemory', 'ReasoningEngine'
]

if __name__ == '__main__':
    print("SecretLLM Library")
    print("Import with: from secret_llm import SecretLLM, SecretAgent, ...")

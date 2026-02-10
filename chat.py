#!/usr/bin/env python3
"""
Chat with your trained SecretLLM Agent
Usage: python chat.py --model_dir ./model
"""

import argparse
import os
import torch
from secret_llm import SecretLLM, ModelConfig, BPETokenizer, SecretAgent

def main():
    parser = argparse.ArgumentParser(description='Chat with SecretLLM')
    parser.add_argument('--model_dir', type=str, default='./model', help='Model directory')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Checkpoint file')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.model_dir, args.checkpoint)
    tokenizer_path = os.path.join(args.model_dir, 'tokenizer.json')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    print("=" * 60)
    print("SECRET LLM AGENT")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    print("✓ Tokenizer loaded")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    if args.cpu:
        config.device = 'cpu'
    
    model = SecretLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if torch.cuda.is_available() and not args.cpu:
        model = model.cuda()
        print("✓ Model loaded (GPU)")
    else:
        print("✓ Model loaded (CPU)")
    
    # Initialize agent
    agent = SecretAgent(model, tokenizer, config)
    
    # Start chat
    agent.chat()

if __name__ == '__main__':
    main()

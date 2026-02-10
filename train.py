#!/usr/bin/env python3
"""
Train SecretLLM on your private data
Usage: python train.py --data_dir ./my_secret_data --output_dir ./model
"""

import argparse
import os
import torch
from secret_llm import (
    SecretLLM, ModelConfig, BPETokenizer, 
    Trainer, TextDataset, DataProcessor
)
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Train SecretLLM')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./model', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Tokenizer vocabulary size')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--steps', type=int, default=100000, help='Training steps')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Configuration
    config = ModelConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.layers,
        batch_size=args.batch_size,
        max_steps=args.steps
    )
    
    print("=" * 60)
    print("SECRET LLM TRAINING")
    print("=" * 60)
    
    # Load or train tokenizer
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.json')
    tokenizer = BPETokenizer(config.vocab_size)
    
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer.load(tokenizer_path)
    else:
        print("Training tokenizer on your data...")
        texts = DataProcessor.load_text_files(args.data_dir)
        if not texts:
            raise ValueError(f"No text files found in {args.data_dir}")
        print(f"Loaded {len(texts)} documents")
        tokenizer.train(texts)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    texts = DataProcessor.load_text_files(args.data_dir)
    dataset = TextDataset(texts, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Initialize model
    print("\nInitializing model...")
    model = SecretLLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Trainer
    trainer = Trainer(model, config, tokenizer)
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    trainer.train(dataloader, save_dir=args.output_dir)
    
    print("\n✓ Training complete!")
    print(f"Model saved to {args.output_dir}")

if __name__ == '__main__':
    main()

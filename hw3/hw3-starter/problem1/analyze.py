"""
Analysis and visualization of attention patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size
from attention import create_causal_mask

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def extract_attention_weights(model, dataloader, device, num_samples=100):
    """
    Extract attention weights from model for analysis.

    Args:
        model: Trained transformer model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to analyze

    Returns:
        Dictionary containing attention weights and sample data
    """
    model.eval()

    all_encoder_attentions = []
    all_decoder_self_attentions = []
    all_decoder_cross_attentions = []
    all_inputs = []
    all_targets = []

    samples_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            batch_size = inputs.size(0)

            # Modify model forward pass to return attention weights
            # This requires updating the model to store/return attention weights




            # For now, we'll need to hook into the attention layers
            encoder_attentions = []
            decoder_self_attentions = []
            decoder_cross_attentions = []

            # Register hooks to capture attention weights
            def make_hook(attention_list):
                def hook(module, input, output):
                    # output is (attention_output, attention_weights)
                    attention_list.append(output[1].detach().cpu())
                return hook

            # Register hooks on attention layers
            # You'll need to access model.encoder_layers[i].self_attn
            # and model.decoder_layers[i].self_attn, cross_attn
            hooks = []
            for layer in model.encoder_layers:
                h = layer.self_attn.register_forward_hook(
                    make_hook(encoder_attentions)
                )
                hooks.append(h)
            for layer in model.decoder_layers:
                h1 = layer.self_attn.register_forward_hook(
                    make_hook(decoder_self_attentions)
                )
                h2 = layer.cross_attn.register_forward_hook(
                    make_hook(decoder_cross_attentions)
                )
                hooks.extend([h1, h2])

            # Forward pass
            # Run model forward pass
            outputs = model(
                src=inputs,
                tgt=targets[:, :-1],
                tgt_mask=create_causal_mask(targets.size(1) - 1).to(device)
            )

            # Collect samples
            samples_to_take = min(batch_size, num_samples - samples_collected)
            all_inputs.extend(inputs[:samples_to_take].cpu().numpy())
            all_targets.extend(targets[:samples_to_take].cpu().numpy())

            # Collect attention weights from hooks
            for i in range(len(encoder_attentions)):
                all_encoder_attentions.append(
                    encoder_attentions[i][:samples_to_take].numpy()
                )
            for i in range(len(decoder_self_attentions)):
                all_decoder_self_attentions.append(
                    decoder_self_attentions[i][:samples_to_take].numpy()
                )
            for i in range(len(decoder_cross_attentions)):
                all_decoder_cross_attentions.append(
                    decoder_cross_attentions[i][:samples_to_take].numpy()
                )

            samples_collected += samples_to_take

    return {
        'encoder_attention': all_encoder_attentions,
        'decoder_self_attention': all_decoder_self_attentions,
        'decoder_cross_attention': all_decoder_cross_attentions,
        'inputs': all_inputs,
        'targets': all_targets
    }


def visualize_attention_pattern(attention_weights, input_tokens, output_tokens,
                               title="Attention Pattern", save_path=None):
    """
    Visualize attention weights as heatmap.

    Args:
        attention_weights: Attention weights [num_heads, out_len, in_len]
        input_tokens: Input token labels
        output_tokens: Output token labels
        title: Plot title
        save_path: Path to save figure
    """
    num_heads = attention_weights.shape[0]

    # Create figure with subplots for each head
    fig, axes = plt.subplots(
        2, (num_heads + 1) // 2,
        figsize=(5 * ((num_heads + 1) // 2), 8)
    )
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]

        # Plot heatmap
        sns.heatmap(
            attention_weights[head_idx],
            ax=ax,
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            vmin=0,
            vmax=1
        )

        ax.set_title(f'Head {head_idx + 1}')
        ax.set_xlabel('Input Position')
        ax.set_ylabel('Output Position')

    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    


def analyze_head_specialization(attention_data, output_dir, num_digits, num_encoder_layers=2, num_decoder_layers=2, num_heads=4):
    """
    Analyze what each attention head specializes in.

    Args:
        attention_data: Dictionary with attention weights and samples
        output_dir: Directory to save analysis results
        num_digits: Number of digits in the addition problem
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    head_stats = {}

    flat_encoder_attentions = attention_data['encoder_attention']
    flat_decoder_self_attentions = attention_data['decoder_self_attention']
    flat_decoder_cross_attentions = attention_data['decoder_cross_attention']

    def aggregate_by_layer(flat_list, num_layers):
        by_layer = [[] for _ in range(num_layers)]
        for idx, attention_batch in enumerate(flat_list):
            layer_idx = idx % num_layers
            by_layer[layer_idx].append(attention_batch)

        aggregated = []
        for layer_batches in by_layer:
            if layer_batches:
                aggregated.append(np.concatenate(layer_batches, axis=0))
            else:
                aggregated.append(None)
        return aggregated

    def compute_head_metrics(head_attention, seq_len_q, seq_len_k):
        operator_token_idx = num_digits
        if operator_token_idx < seq_len_k:
            avg_operator = float(head_attention[:, :, operator_token_idx].mean())
        else:
            avg_operator = 0.0

        diagonal = np.diagonal(head_attention, axis1=-2, axis2=-1)
        avg_diagonal = float(diagonal.mean()) if diagonal.size else 0.0

        entropy = float(
            -np.sum(head_attention * np.log(head_attention + 1e-10), axis=-1).mean()
        )

        carry_values = []
        for i in range(num_digits):
            pos1 = num_digits - 1 - i
            pos2 = 2 * num_digits - i
            if pos1 < seq_len_q and pos2 < seq_len_k:
                carry_values.append(head_attention[:, pos1, pos2])
            if pos2 < seq_len_q and pos1 < seq_len_k:
                carry_values.append(head_attention[:, pos2, pos1])

        if carry_values:
            carry_stack = np.stack(carry_values, axis=0)
            avg_carry = float(carry_stack.mean())
        else:
            avg_carry = 0.0

        return avg_operator, avg_diagonal, avg_carry, entropy

    attention_sources = [
        ("encoder", flat_encoder_attentions, num_encoder_layers),
        ("decoder_self", flat_decoder_self_attentions, num_decoder_layers),
        ("decoder_cross", flat_decoder_cross_attentions, num_decoder_layers),
    ]

    for attn_name, flat_list, num_layers in attention_sources:
        readable_name = attn_name.replace('_', ' ')
        if not flat_list:
            print(f"No {readable_name} attention data found.")
            continue

        print(f"Analyzing {readable_name} attention patterns...")
        per_layer_attentions = aggregate_by_layer(flat_list, num_layers)

        for layer_idx, layer_attentions in enumerate(per_layer_attentions):
            if layer_attentions is None:
                continue

            seq_len_q = layer_attentions.shape[2]
            seq_len_k = layer_attentions.shape[3]

            for head_idx in range(num_heads):
                head_attention = layer_attentions[:, head_idx, :, :]
                avg_operator, avg_diagonal, avg_carry, entropy = compute_head_metrics(
                    head_attention,
                    seq_len_q,
                    seq_len_k
                )

                stat_key = f'{attn_name}_layer_{layer_idx}_head_{head_idx}'
                head_stats[stat_key] = {
                    'avg_attention_to_operator': avg_operator,
                    'avg_attention_to_same_position': avg_diagonal,
                    'avg_attention_to_carry_positions': avg_carry,
                    'entropy': entropy
                }

    # Save analysis results
    with open(output_dir / 'head_analysis.json', 'w') as f:
        json.dump(head_stats, f, indent=2)

    return head_stats


def ablation_study(model, dataloader, device, output_dir):
    """
    Perform head ablation study.

    Test model performance when individual heads are disabled.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run on
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running head ablation study...")

    # Get baseline accuracy
    baseline_acc = evaluate_model(model, dataloader, device)
    print(f"Baseline accuracy: {baseline_acc:.2%}")

    ablation_results = {'baseline': baseline_acc}

    # TODO: For each layer and head:
    # 1. Temporarily zero out the head's output
    # 2. Evaluate model performance
    # 3. Restore the head
    # 4. Record the performance drop
    for layer_idx, layer in enumerate(model.encoder_layers):
        num_heads = layer.self_attn.num_heads
        for head_idx in range(num_heads):
            print(f"Ablating Layer {layer_idx} Head {head_idx}...")

            # Store original projection weights
            original_W_O = layer.self_attn.W_O.weight.data.clone()
            original_W_O_bias = layer.self_attn.W_O.bias.data.clone()

            # Zero out the head's output by modifying W_O
            with torch.no_grad():
                d_k = layer.self_attn.d_k
                start = head_idx * d_k
                end = (head_idx + 1) * d_k
                layer.self_attn.W_O.weight.data[:, start:end] = 0.0
                layer.self_attn.W_O.bias.data += 0.0  # No change to bias

            # Evaluate model
            ablated_acc = evaluate_model(model, dataloader, device)
            print(f"  Ablated accuracy: {ablated_acc:.2%}")

            # Restore original weights
            with torch.no_grad():
                layer.self_attn.W_O.weight.data = original_W_O
                layer.self_attn.W_O.bias.data = original_W_O_bias

            # Record results
            ablation_results[f'layer_{layer_idx}_head_{head_idx}'] = ablated_acc

    for layer_idx, layer in enumerate(model.decoder_layers):
        num_heads = layer.self_attn.num_heads
        for head_idx in range(num_heads):
            print(f"Ablating Layer {layer_idx} Self-Attention Head {head_idx}...")

            # Store original projection weights
            original_W_O = layer.self_attn.W_O.weight.data.clone()
            original_W_O_bias = layer.self_attn.W_O.bias.data.clone()

            # Zero out the head's output by modifying W_O
            with torch.no_grad():
                d_k = layer.self_attn.d_k
                start = head_idx * d_k
                end = (head_idx + 1) * d_k
                layer.self_attn.W_O.weight.data[:, start:end] = 0.0
                layer.self_attn.W_O.bias.data += 0.0  # No change to bias

            # Evaluate model
            ablated_acc = evaluate_model(model, dataloader, device)
            print(f"  Ablated accuracy: {ablated_acc:.2%}")

            # Restore original weights
            with torch.no_grad():
                layer.self_attn.W_O.weight.data = original_W_O
                layer.self_attn.W_O.bias.data = original_W_O_bias

            # Record results
            ablation_results[f'Self-Attention_layer_{layer_idx}_head_{head_idx}'] = ablated_acc

    for layer_idx, layer in enumerate(model.decoder_layers):
        num_heads = layer.cross_attn.num_heads
        for head_idx in range(num_heads):
            print(f"Ablating Layer {layer_idx} Cross-Attention Head {head_idx}...")

            # Store original projection weights
            original_W_O = layer.cross_attn.W_O.weight.data.clone()
            original_W_O_bias = layer.cross_attn.W_O.bias.data.clone()

            # Zero out the head's output by modifying W_O
            with torch.no_grad():
                d_k = layer.cross_attn.d_k
                start = head_idx * d_k
                end = (head_idx + 1) * d_k
                layer.cross_attn.W_O.weight.data[:, start:end] = 0.0
                layer.cross_attn.W_O.bias.data += 0.0  # No change to bias

            # Evaluate model
            ablated_acc = evaluate_model(model, dataloader, device)
            print(f"  Ablated accuracy: {ablated_acc:.2%}")

            # Restore original weights
            with torch.no_grad():
                layer.cross_attn.W_O.weight.data = original_W_O
                layer.cross_attn.W_O.bias.data = original_W_O_bias

            # Record results
            ablation_results[f'cross_layer_{layer_idx}_head_{head_idx}'] = ablated_acc       

    # Save ablation results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # Create visualization of head importance
    plot_head_importance(ablation_results, output_dir / 'head_importance.png')

    return ablation_results


def evaluate_model(model, dataloader, device):
    """
    Evaluate model accuracy.

    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to run on

    Returns:
        Accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Generate predictions
            # Compare with targets
            # Count correct sequences
            predictions = model.generate(inputs, max_len=targets.size(1))
            for pred, target in zip(predictions, targets):
                if torch.equal(pred, target):
                    correct += 1
                total += 1


    return correct / total


def plot_head_importance(ablation_results, save_path):
    """
    Visualize head importance from ablation study.

    Args:
        ablation_results: Dictionary of ablation results
        save_path: Path to save figure
    """
    # Extract performance drops for each head
    baseline = ablation_results['baseline']

    # Create bar plot showing accuracy drop when each head is removed

    drops = []
    for key, acc in ablation_results.items():
        if key == 'baseline':
            continue
        drops.append((key, baseline - acc))

    drops.sort(key=lambda item: item[1], reverse=True)
    head_labels = [key for key, _ in drops]
    head_labels = [label.replace('_', ' ') for label in head_labels]
    head_labels = [label.replace('layer', 'L').replace('head', 'H') for label in head_labels]
    head_labels = [label.replace('Self-Attention', 'Self').replace('cross', 'Cross') for label in head_labels]
    accuracy_drops = [value for _, value in drops]


   

    plt.figure(figsize=(12, 6))

    # Plot bars for each head
    plt.bar(head_labels, accuracy_drops, color='skyblue')
    plt.axhline(0, color='gray', linestyle='--')


    plt.xlabel('Head')
    plt.ylabel('Accuracy Drop')
    plt.title('Head Importance (Accuracy Drop When Removed)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    


def visualize_example_predictions(model, dataloader, device, output_dir, num_examples=5):
    """
    Visualize model predictions on example inputs.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        output_dir: Directory to save visualizations
        num_examples: Number of examples to visualize
    """
    output_dir = Path(output_dir)
    (output_dir / 'attention_patterns' / 'examples').mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_examples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Take first sample from batch
            input_seq = inputs[0:1]
            target_seq = targets[0]

            # Generate prediction
            # Use model.generate() to get prediction
            prediction = model.generate(input_seq, max_len=target_seq.size(0))

            # Convert to strings for visualization
            input_str = ' '.join(map(str, input_seq[0].cpu().numpy()))
            target_str = ''.join(map(str, target_seq.cpu().numpy()))
            pred_str = ''.join(map(str, prediction[0].cpu().numpy()))

            print(f"\nExample {batch_idx + 1}:")
            print(f"  Input:  {input_str}")
            print(f"  Target: {target_str}")
            print(f"  Pred:   {pred_str}")
            print(f"  Correct: {target_str == pred_str}")

            # Extract and visualize attention for this example
            with torch.enable_grad():
                attention_weights = model.get_attention_weights(input_seq, prediction)

            # Visualize and save encoder self-attention
            num_encoder_layers = attention_weights['encoder_attention'].shape[0]
            for layer_idx in range(num_encoder_layers):
                visualize_attention_pattern(
                    attention_weights['encoder_attention'][layer_idx].squeeze(0).cpu().numpy(),
                    input_tokens=[str(i) for i in input_seq[0].cpu().numpy()],
                    output_tokens=[str(i) for i in input_seq[0].cpu().numpy()],
                    title=f"Example {batch_idx + 1}: Encoder Layer {layer_idx} Self-Attention",
                    save_path=output_dir / 'attention_patterns' / 'examples' / f'example_{batch_idx}_encoder_layer_{layer_idx}_self_attn.png'
                )
            

            # Visualize and save decoder cross-attention
            num_decoder_layers = attention_weights['decoder_cross_attention'].shape[0]
            for layer_idx in range(num_decoder_layers):
                visualize_attention_pattern(
                    attention_weights['decoder_cross_attention'][layer_idx].squeeze(0).cpu().numpy(),
                    input_tokens=[str(i) for i in input_seq[0].cpu().numpy()],
                    output_tokens=[str(i) for i in prediction[0].cpu().numpy()],
                    title=f"Example {batch_idx + 1}: Decoder Layer {layer_idx} Cross-Attention",
                    save_path=output_dir / 'attention_patterns' / 'examples' / f'example_{batch_idx}_decoder_layer_{layer_idx}_cross_attn.png'
                )




def main():
    parser = argparse.ArgumentParser(description='Analyze attention patterns')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-digits', type=int, default=1, help='Number of digits in addition problem')

    args = parser.parse_args()

    # Load model
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512
    ).to(args.device)

    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")

    # Load data
    _, _, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'attention_patterns').mkdir(parents=True, exist_ok=True)
    (output_dir / 'head_analysis').mkdir(parents=True, exist_ok=True)

    # Extract attention weights
    print("Extracting attention weights...")
    attention_data = extract_attention_weights(
        model, test_loader, args.device, args.num_samples
    )

    # Analyze head specialization
    head_stats = analyze_head_specialization(
        attention_data,
        output_dir / 'head_analysis',
        args.num_digits,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4
    )

    # Run ablation study
    ablation_results = ablation_study(
       model, test_loader, args.device, output_dir / 'head_analysis'
    )

    # Visualize example predictions
    visualize_example_predictions(
        model, test_loader, args.device, output_dir, num_examples=5
    )


    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
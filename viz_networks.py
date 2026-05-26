"""
InternVLA-N1: Network Architecture & Internal Structures
========================================================
Detailed visualization of S1 and S2 network architectures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'input': '#3498DB',
    'output': '#27AE60',
    'hidden': '#9B59B6',
    'attention': '#F39C12',
    'conv': '#E74C3C',
    'fc': '#16A085',
    'skip': '#E91E63',
    'noise': '#795548',
    'env': '#34495E',
    'observation': '#3498DB',
    's2': '#E74C3C',
    'pixel_goal': '#F39C12',
    'latent': '#9B59B6',
    's1': '#27AE60',
    'trajectory': '#1ABC9C',
    'action': '#16A085',
}

def draw_layer(ax, x, y, w, h, color, name, details='', fontsize=7):
    """Draw a network layer box"""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.65, name, ha='center', va='center', 
           fontsize=fontsize, fontweight='bold', color='white')
    if details:
        ax.text(x + w/2, y + h*0.3, details, ha='center', va='center', 
               fontsize=5, color='white')

def draw_connection(ax, start, end, color='black', dashed=False):
    """Draw connection between layers"""
    style = ':' if dashed else '-'
    ax.plot([start[0], end[0]], [start[1], end[1]], 
           color=color, linestyle=style, linewidth=1.5)

# ============================================================================
# FIGURE 2: S2 Network Architecture (Language Planner)
# ============================================================================
def create_s2_architecture():
    """Detailed S2 network architecture"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('S2: LANGUAGE PLANNER - NETWORK ARCHITECTURE', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # =========================================================================
    # Panel A: Vision Encoder (CLIP/ViT)
    # =========================================================================
    ax = axes[0, 0]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_title('A. Vision Encoder (CLIP/ViT-based)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Input
    draw_layer(ax, 1, 9, 3, 2, COLORS['input'], 'RGB Image', '(B,224,224,3)')
    
    # Patch Embedding
    draw_layer(ax, 5, 9, 3, 2, COLORS['conv'], 'Patch Embed', '16×16 patches\n→ 768 features')
    draw_connection(ax, (4, 10), (5, 10))
    
    # Transformer Blocks
    for i in range(3):
        y_pos = 7 - i * 2
        draw_layer(ax, 9, y_pos, 2.5, 1.5, COLORS['hidden'], f'Transformer\nBlock {i+1}', '768-D')
        draw_layer(ax, 12, y_pos, 2, 1.5, COLORS['attention'], 'MH-Attention', '12 heads')
        draw_layer(ax, 9, y_pos - 1.5, 2.5, 1.2, COLORS['fc'], 'FFN', '3072→768')
    
    draw_connection(ax, (8, 10), (9, 9.75))
    draw_connection(ax, (11.5, 9), (12, 9), color=COLORS['attention'])
    draw_connection(ax, (14, 9), (9, 8.5), color=COLORS['skip'])
    
    # Output
    draw_layer(ax, 15, 9, 3, 2, COLORS['output'], 'Vision\nFeatures', '(B,196,768)')
    draw_connection(ax, (14, 10), (15, 10))
    
    # =========================================================================
    # Panel B: Text Encoder (LSTM/BERT)
    # =========================================================================
    ax = axes[0, 1]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_title('B. Text Encoder (LSTM/BERT)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Input
    draw_layer(ax, 1, 9, 3, 2, COLORS['input'], 'Instruction', '"Go to kitchen"')
    
    # Embedding
    draw_layer(ax, 5, 9, 2.5, 2, COLORS['fc'], 'Token Embed', 'Vocab→512')
    draw_connection(ax, (4, 10), (5, 10))
    
    # LSTM Layers
    for i in range(2):
        y_pos = 7 - i * 2.5
        draw_layer(ax, 8.5, y_pos, 3, 2, COLORS['hidden'], f'LSTM Layer {i+1}', 
                  'hidden=512')
        draw_layer(ax, 8.5, y_pos - 1.5, 3, 1.2, COLORS['fc'], 'Dropout', 'p=0.1')
    
    draw_connection(ax, (7.5, 10), (8.5, 8.5))
    
    # Output projection
    draw_layer(ax, 13, 8, 3, 2, COLORS['output'], 'Language\nFeatures', '(B,512)')
    draw_connection(ax, (11.5, 6), (13, 9))
    
    # =========================================================================
    # Panel C: Multimodal Fusion & Output Heads
    # =========================================================================
    ax = axes[1, 0]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_title('C. Multimodal Fusion & Output Heads', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Inputs
    draw_layer(ax, 1, 9, 2.5, 1.5, COLORS['input'], 'Vision\n(B,196,768)', fontsize=6)
    draw_layer(ax, 1, 6, 2.5, 1.5, COLORS['input'], 'Language\n(B,512)', fontsize=6)
    
    # Cross Attention Fusion
    draw_layer(ax, 5, 6, 4, 3, COLORS['attention'], 'Cross-Attention\nFusion', 
              'Q=Vision, K/V=Lang\n512-D output')
    draw_connection(ax, (4, 9.5), (5, 7.5), color=COLORS['input'])
    draw_connection(ax, (4, 6.5), (5, 7), color=COLORS['input'])
    
    # Output Heads
    head_y = [9, 6.5, 4]
    head_colors = ['#F39C12', COLORS['hidden'], COLORS['input']]
    head_names = ['Pixel Goal\nHead (2-D)', 'Latent Code\nHead (256-D)', 'Action\nHead (vocab)']
    
    for i, (y, color, name) in enumerate(zip(head_y, head_colors, head_names)):
        draw_layer(ax, 10.5, y, 3, 1.8, color, name, fontsize=6)
        draw_connection(ax, (9, 7.5), (10.5, y + 0.9), color=COLORS['attention'])
    
    # Decoders for each head
    for i, (y, color) in enumerate(zip([9, 6.5, 4], head_colors)):
        draw_layer(ax, 14.5, y, 2.5, 1.8, color, 'MLP Decoder', fontsize=6)
        draw_connection(ax, (13.5, y + 0.9), (14.5, y + 0.9))
    
    # =========================================================================
    # Panel D: Complete S2 Data Flow
    # =========================================================================
    ax = axes[1, 1]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_title('D. Complete S2 Data Flow', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Stages
    stages = [
        (1, 'rgb, depth', 'Sensors'),
        (4, 'features', 'Encoder'),
        (8, 'fused', 'Fusion'),
        (12, 'outputs', 'Heads'),
        (16, '(u,v), z', 'Output'),
    ]
    
    for x, data, label in stages:
        rect = FancyBboxPatch((x, 5), 2.5, 2, boxstyle="round,pad=0.05",
                              facecolor='#ECF0F1', edgecolor='#2C3E50', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.25, 6.2, label, ha='center', va='center', 
               fontsize=9, fontweight='bold')
        ax.text(x + 1.25, 5.7, data, ha='center', va='center', 
               fontsize=7, color='#7F8C8D')
        
        if x < 16:
            ax.annotate('', xy=(x + 2.7, 6), xytext=(x + 2.5, 6),
                       arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Dimension annotations
    dims = ['(224,224,3)\n(224,224,1)', '(196,768)\n(77,512)', '(B,512)', 
            '(B,2)\n(B,256)', 'u,v ∈ [0,224]\nz ∈ ℝ²⁵⁶']
    for (x, _, _), dim in zip(stages, dims):
        ax.text(x + 1.25, 3.5, dim, ha='center', va='center', 
               fontsize=6, color='#E74C3C', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig2_s2_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Generated: fig2_s2_architecture.png")

# ============================================================================
# FIGURE 3: S1 Network Architecture (Visual Navigation Policy)
# ============================================================================
def create_s1_architecture():
    """Detailed S1 network architecture"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('S1: VISUAL NAVIGATION POLICY - NETWORK ARCHITECTURE', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # =========================================================================
    # Panel A: Input Processing
    # =========================================================================
    ax = axes[0, 0]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_title('A. Input Processing (2-Frame History)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Inputs
    draw_layer(ax, 1, 9, 2.5, 2, COLORS['input'], 'RGB t-1\n(224,224,3)', fontsize=7)
    draw_layer(ax, 1, 6, 2.5, 2, COLORS['input'], 'RGB t\n(224,224,3)', fontsize=7)
    draw_layer(ax, 1, 3, 2.5, 2, COLORS['input'], 'Depth\n(224,224,1)', fontsize=7)
    
    # Encoders
    for i, (y, name) in enumerate([(9, 'RGB Enc'), (6, 'RGB Enc'), (3, 'Depth Enc')]):
        draw_layer(ax, 5, y, 2.5, 1.5, COLORS['conv'], name, 'ResNet-50\n→256 features')
        draw_connection(ax, (4, y + 1), (5, y + 0.75))
    
    # Concatenation
    draw_layer(ax, 8.5, 5, 2.5, 3, '#34495E', 'Concat\n& Flatten', '768-D')
    draw_connection(ax, (7.5, 9.5), (8.5, 6.5))
    draw_connection(ax, (7.5, 6.5), (8.5, 6.5))
    draw_connection(ax, (7.5, 3.5), (8.5, 5.5))
    
    # =========================================================================
    # Panel B: Latent Conditioning
    # =========================================================================
    ax = axes[0, 1]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_title('B. Latent Code Conditioning', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Latent input
    draw_layer(ax, 1, 8, 2.5, 1.5, COLORS['hidden'], 'Latent z\n(256-D)', fontsize=7)
    
    # MLP projection
    draw_layer(ax, 5, 7.5, 2.5, 2, COLORS['fc'], 'MLP\nProjection', '256→512')
    draw_connection(ax, (4, 8.5), (5, 8.5))
    
    # Cross attention
    draw_layer(ax, 9, 6, 4, 3, COLORS['attention'], 'Cross-Attention\nConditioning', 
              'Q=Vision features\nK,V=Latent z\nAdditive concat')
    draw_connection(ax, (7.5, 8.5), (9, 7.5))
    
    # Output
    draw_layer(ax, 14, 7, 3, 2, COLORS['output'], 'Conditioned\nFeatures', '(B,768)')
    draw_connection(ax, (13, 7.5), (14, 8))
    
    # =========================================================================
    # Panel C: Diffusion Process
    # =========================================================================
    ax = axes[1, 0]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_title('C. Diffusion Trajectory Prediction', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Forward process
    ax.text(3, 11, 'Forward Process (q)', fontsize=10, fontweight='bold')
    draw_layer(ax, 1, 9, 2.5, 1.5, COLORS['output'], 'Clean τ\n(N×3)', fontsize=7)
    
    for i, t in enumerate([0, 50, 100]):
        x_pos = 5 + i * 2.5
        draw_layer(ax, x_pos, 9, 2, 1.5, '#795548', f't={t}', f'ᾱ={1-t/100:.2f}')
        draw_connection(ax, (x_pos - 1.2, 9.75), (x_pos, 9.75))
        if i > 0:
            ax.text((x_pos - 1.2 + x_pos)/2, 10.2, '+ noise', fontsize=6, ha='center')
    
    # Reverse process
    ax.text(3, 7, 'Reverse Process (pθ)', fontsize=10, fontweight='bold')
    draw_layer(ax, 1, 5, 2.5, 1.5, COLORS['noise'], 'Noisy τ\n(N×3)', fontsize=7)
    
    # U-Net backbone
    draw_layer(ax, 5, 3, 10, 4, '#9C27B0', 'U-Net Backbone\nwith Cross-Attention', 
              '• Encoder: concat features\n• Decoder: skip connections\n• Time embedding: sinusoidal')
    
    draw_connection(ax, (4, 5.75), (5, 5))
    
    # Output
    draw_layer(ax, 16, 4, 2.5, 2, COLORS['output'], 'Noise ε̂\nPredicted', fontsize=7)
    draw_connection(ax, (15, 5), (16, 5))
    
    # Loss arrow
    ax.annotate('', xy=(9, 1.5), xytext=(9, 3),
               arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
    ax.text(10.5, 2.2, 'L = MSE(ε, ε̂)', fontsize=9, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#E74C3C'))
    
    # =========================================================================
    # Panel D: Trajectory Output Decoding
    # =========================================================================
    ax = axes[1, 1]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_title('D. Trajectory Output Decoding', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Noise prediction
    draw_layer(ax, 1, 8, 3, 1.5, COLORS['hidden'], 'ε̂ Prediction\n(N×3)', fontsize=7)
    
    # Denoise
    ax.text(6, 8.5, 'Denoise:', fontsize=9, fontweight='bold')
    ax.text(6, 8, 'τ₀ = (1/√āₜ)·(zₜ - √(1-āₜ)·ε̂)', fontsize=8, fontfamily='monospace')
    
    # Trajectory
    draw_layer(ax, 10, 7, 3.5, 2, COLORS['output'], 'Trajectory τ\nShape: (N, 3)', fontsize=7)
    draw_connection(ax, (4, 8.75), (10, 8))
    
    # Breakdown
    draw_layer(ax, 15, 9, 3, 1.2, COLORS['output'], 'dx (forward)', fontsize=6)
    draw_layer(ax, 15, 7, 3, 1.2, COLORS['output'], 'dy (lateral)', fontsize=6)
    draw_layer(ax, 15, 5, 3, 1.2, COLORS['output'], 'dθ (heading)', fontsize=6)
    
    for y in [9, 7, 5]:
        draw_connection(ax, (13.5, 8), (15, y + 0.6))
    
    # Visualization
    ax.text(2, 3, 'τ visualization:', fontsize=9, fontweight='bold')
    for i, (dx, dy, dt) in enumerate([(0.5, 0.2, 0.1), (0.4, 0.3, 0.2), (0.3, 0.1, -0.1)]):
        x = 5 + i * 2
        ax.arrow(x, 2, dx, dy, head_width=0.15, head_length=0.08, 
                fc=COLORS['output'], ec='black')
        ax.text(x, 1.2, f'Step {i+1}', fontsize=7, ha='center')
        ax.text(x, 0.5, f'({dx:.1f},{dy:.1f},{dt:.1f})', fontsize=5, ha='center')
    
    plt.tight_layout()
    plt.savefig('fig3_s1_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Generated: fig3_s1_architecture.png")

# ============================================================================
# FIGURE 4: Complete Model Network Diagram
# ============================================================================
def create_complete_network():
    """Complete network diagram showing all components"""
    
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    fig.suptitle('COMPLETE InternVLA-N1 DUAL-SYSTEM NETWORK DIAGRAM', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # =========================================================================
    # Row 1: Input to S2
    # =========================================================================
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 8)
    ax.set_title('Complete S2 (Language Planner) Processing Pipeline', fontsize=14, 
                fontweight='bold', pad=15)
    ax.axis('off')
    
    # Sensors
    draw_layer(ax, 0.5, 5, 2.5, 2, COLORS['input'], 'RGB\nSensor', '(H,W,3)')
    draw_layer(ax, 0.5, 2.5, 2.5, 2, COLORS['input'], 'Depth\nSensor', '(H,W,1)')
    draw_layer(ax, 0.5, 0, 2.5, 2, COLORS['input'], 'Language\nInput', 'T tokens')
    
    # Stage 1: Encoding
    ax.text(5, 7, 'STAGE 1: ENCODING', fontsize=11, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='#E8F5E9'))
    
    draw_layer(ax, 4, 5, 2.5, 2, COLORS['conv'], 'ResNet/\nViT Encoder', '→512-D')
    draw_layer(ax, 4, 2.5, 2.5, 2, COLORS['conv'], 'Depth\nEncoder', '→256-D')
    draw_layer(ax, 4, 0, 2.5, 2, COLORS['hidden'], 'Text\nEncoder', '→512-D')
    
    # Stage 2: Fusion
    ax.text(9.5, 7, 'STAGE 2: FUSION', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#E3F2FD'))
    
    draw_layer(ax, 8.5, 2, 4, 4, COLORS['attention'], 'Cross-Modal\nAttention\n(Q,K,V)', 'Vision↔Lang\n→512-D')
    
    # Stage 3: Output
    ax.text(15, 7, 'STAGE 3: OUTPUT HEADS', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFF3E0'))
    
    outputs = [
        (14, 5.5, COLORS['attention'], 'Pixel Goal\nHead', '(u,v)'),
        (14, 3, COLORS['hidden'], 'Latent Head', 'z∈ℝ²⁵⁶'),
        (14, 0.5, COLORS['output'], 'Action Head', 'vocab'),
    ]
    
    for x, y, color, name, detail in outputs:
        draw_layer(ax, x, y, 3, 1.8, color, name, detail)
        ax.annotate('', xy=(x, y + 0.9), xytext=(12.5, 4),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Arrows
    for (x1, y1), (x2, y2) in [((3, 6), (4, 6)), ((3, 3.5), (4, 3.5)), ((3, 1), (4, 1))]:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['input'], lw=2))
    ax.annotate('', xy=(8.5, 4), xytext=(6.5, 4),
               arrowprops=dict(arrowstyle='->', color=COLORS['conv'], lw=2))
    
    # =========================================================================
    # Row 2: S1 Processing
    # =========================================================================
    ax = fig.add_subplot(gs[1, :2])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.set_title('S1 (Visual Navigation Policy) - Trajectory Generation', fontsize=12, 
                fontweight='bold', pad=10)
    ax.axis('off')
    
    # Inputs to S1
    draw_layer(ax, 1, 7, 2.5, 2, COLORS['input'], 'Current\nRGB', fontsize=8)
    draw_layer(ax, 1, 4, 2.5, 2, COLORS['input'], 'Current\nDepth', fontsize=8)
    draw_layer(ax, 1, 1, 2.5, 2, COLORS['attention'], 'Pixel Goal\n(u,v)', fontsize=8)
    
    # History encoder
    draw_layer(ax, 5, 5, 3, 3, COLORS['conv'], 'History\nEncoder\n(2-frame)', 'ResNet→512')
    draw_layer(ax, 5, 1, 3, 2, COLORS['hidden'], 'Goal\nEncoder', 'MLP→256')
    
    # Conditioning
    draw_layer(ax, 9.5, 4, 3, 4, COLORS['attention'], 'Conditioning\nBlock\n(Cross-Attn)', 'z + goal')
    
    # Diffusion
    draw_layer(ax, 13.5, 3, 4, 5, '#9C27B0', 'Diffusion\nU-Net', 'T=100 steps\nε_θ prediction')
    
    # Output
    draw_layer(ax, 18, 5, 2, 2, COLORS['output'], 'τ\n(N,3)', 'Trajectory')
    
    # Arrows
    for y in [8, 5, 2]:
        ax.annotate('', xy=(5, 6.5), xytext=(3.5, y + 1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['input'], lw=1.5))
    ax.annotate('', xy=(8, 6.5), xytext=(8, 6.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['attention'], lw=1.5))
    ax.annotate('', xy=(12.5, 5.5), xytext=(12.5, 5.5),
               arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=2))
    ax.annotate('', xy=(17.5, 6), xytext=(17.5, 6),
               arrowprops=dict(arrowstyle='->', color=COLORS['output'], lw=2))
    
    # =========================================================================
    # Row 2: Execution & Environment
    # =========================================================================
    ax = fig.add_subplot(gs[1, 2:])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.set_title('Trajectory Execution & Environment Loop', fontsize=12, 
                fontweight='bold', pad=10)
    ax.axis('off')
    
    # Trajectory
    draw_layer(ax, 1, 6, 3, 2, COLORS['output'], 'τ\n(N,3)', 'Trajectory')
    
    # Executor
    draw_layer(ax, 5, 6, 3, 2, '#16A085', 'Action\nExecutor', 'SE(2) ⊕')
    
    # Environment
    draw_layer(ax, 9, 6, 3, 2, COLORS['env'], 'Environment\nSim', 'Step()')
    
    # State
    draw_layer(ax, 13, 6, 3, 2, COLORS['input'], 'New State\n(pose, obs)', '')
    
    # Loop back
    ax.annotate('', xy=(5, 5), xytext=(13, 4),
               arrowprops=dict(arrowstyle='->', color=COLORS['env'], lw=2,
                             connectionstyle='arc3,rad=-0.4'))
    ax.text(9, 3, 'Loop: max(N, gap) steps', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFEBEE'))
    
    # Arrows
    for start_end in [(4, 7, 5, 7), (8, 7, 9, 7), (12, 7, 13, 7)]:
        x1, y1, x2, y2 = start_end
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # =========================================================================
    # Row 3: Timing & Modes
    # =========================================================================
    for idx, (gap, mode) in enumerate([(1, 'SYNC'), (2, 'ASYNC-2'), (8, 'ASYNC-8')]):
        ax = fig.add_subplot(gs[2, idx])
        ax.set_title(f'{mode} (gap={gap})', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 4)
        
        for step in range(16):
            s2 = (step == 0) or (step % gap == 0)
            
            # S2
            color = COLORS['s2'] if s2 else '#BDC3C7'
            rect = patches.Rectangle((step, 2.5), 0.8, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            if s2:
                ax.text(step + 0.4, 3.1, 'S2', ha='center', va='center', 
                       fontsize=6, fontweight='bold', color='white')
            
            # S1 Execution
            rect = patches.Rectangle((step, 1), 0.8, 1, facecolor=COLORS['output'], 
                                     edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
        
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(['Step', 'S1 Exec', 'S2', 'Legend'])
        ax.set_xlabel('Time step')
    
    # Metrics
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    
    metrics = """
    MODE COMPARISON
    ═════════════════════════════
    
    SYNC (gap=1):
    ├─ S2 frequency: 100%
    ├─ Latency: ~80ms/step
    └─ Freshness: 1.0
    
    ASYNC (gap=2):
    ├─ S2 frequency: 50%
    ├─ Latency: ~40ms/step
    └─ Freshness: 0.7
    
    ASYNC (gap=8):
    ├─ S2 frequency: 12.5%
    ├─ Latency: ~15ms/step
    └─ Freshness: 0.3
    """
    ax.text(0.1, 0.95, metrics, transform=ax.transAxes, fontsize=9,
           fontfamily='monospace', verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#2C3E50'))
    
    plt.tight_layout()
    plt.savefig('fig4_complete_network.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Generated: fig4_complete_network.png")

if __name__ == '__main__':
    create_s2_architecture()
    create_s1_architecture()
    create_complete_network()
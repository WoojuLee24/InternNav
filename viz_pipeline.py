"""
InternVLA-N1: Complete End-to-End Data Flow Visualization
==========================================================
Detailed visualization of all intermediate inputs and outputs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow, Arc, Wedge
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# COLOR SCHEME
# ============================================================================
COLORS = {
    'observation': '#3498DB',    # Blue - observations
    's2': '#E74C3C',             # Red - System 2
    'pixel_goal': '#F39C12',      # Orange - pixel goal
    'latent': '#9B59B6',          # Purple - latent
    's1': '#27AE60',             # Green - System 1
    'trajectory': '#1ABC9C',      # Teal - trajectory
    'action': '#16A085',         # Dark teal - action
    'env': '#34495E',             # Dark gray - environment
    'arrow': '#2C3E50',           # Dark - arrows
    'highlight': '#E91E63',      # Pink - highlights
    'success': '#00C853',        # Green - success
}

def draw_box(ax, x, y, w, h, color, text, fontsize=8, text_color='white', edgecolor='black'):
    """Draw a styled box with text"""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor=color, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
           fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, start, end, color='black', style='->', label='', label_offset=(0, 0.3)):
    """Draw arrow with optional label"""
    ax.annotate('', xy=end, xytext=start,
               arrowprops=dict(arrowstyle=style, color=color, lw=2, 
                             connectionstyle='arc3,rad=0'))
    if label:
        mid_x = (start[0] + end[0]) / 2 + label_offset[0]
        mid_y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mid_x, mid_y, label, fontsize=7, color=color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor=color, alpha=0.9))

def draw_diamond(ax, x, y, size, color, text, fontsize=6):
    """Draw decision diamond"""
    diamond = patches.RegularPolygon((x, y), numVertices=4, radius=size,
                                     facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
           fontweight='bold', color='white')

# ============================================================================
# FIGURE 1: Complete Pipeline Overview
# ============================================================================
def create_pipeline_overview():
    """Create complete pipeline overview with all stages"""
    
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[0.8, 1.5, 1.2, 1.2, 1],
                          hspace=0.35)
    
    # =========================================================================
    # Section 1: High-Level Architecture
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 8)
    ax1.set_title('1. HIGH-LEVEL SYSTEM ARCHITECTURE', fontsize=14, fontweight='bold', 
                  pad=20, color='#2C3E50')
    ax1.axis('off')
    
    # Main components
    components = [
        (2, 4, 2.5, 1.5, COLORS['observation'], 'OBSERVATION\n(rgb, depth)\nShape: (224, 224, 4)', 'Sensors'),
        (6, 4, 2.5, 1.5, COLORS['s2'], 'S2: LANGUAGE\nPLANNER\n(LSTM/VLM)', 'Planning'),
        (10, 4, 2.5, 1.5, COLORS['s1'], 'S1: VISUAL NAV\nPOLICY\n(Diffusion/Trans)', 'Control'),
        (14, 4, 2.5, 1.5, COLORS['trajectory'], 'TRAJECTORY\ngenerator\n[dx,dy,dθ]×N', 'Output'),
        (18, 4, 2.5, 1.5, COLORS['action'], 'ACTION\nExecution\n(robot motors)', 'Actuation'),
    ]
    
    for x, y, w, h, color, text, label in components:
        draw_box(ax1, x, y, w, h, color, text, fontsize=9)
        ax1.text(x + w/2, y - 0.3, label, ha='center', va='top', fontsize=8, 
                color=color, fontweight='bold')
    
    # Arrows between main components
    for i in range(len(components) - 1):
        x1 = components[i][0] + components[i][2]
        x2 = components[i+1][0]
        y = 4.75
        draw_arrow(ax1, (x1, y), (x2, y), color=COLORS['arrow'])
        ax1.text((x1 + x2)/2, y + 0.3, f'Step {i+1}', fontsize=7, ha='center', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Key operations boxes
    ops = [
        (4, 2, 'Sensor Fusion\n(image encoding)'),
        (8, 2, 'Language Parsing\n+ Pixel Selection'),
        (12, 2, 'Trajectory\nPrediction'),
        (16, 2, 'Action\nDecoding'),
    ]
    
    for x, y, text in ops:
        draw_box(ax1, x, y, 1.5, 0.8, '#95A5A6', text, fontsize=7)
    
    # =========================================================================
    # Section 2: Detailed S2 Processing
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 12)
    ax2.set_title('2. S2: LANGUAGE PLANNER - DETAILED PROCESSING', fontsize=14, 
                  fontweight='bold', pad=15, color=COLORS['s2'])
    ax2.axis('off')
    
    # Input branch
    draw_box(ax2, 0.5, 9, 3, 1.5, COLORS['observation'], 'RGB Image\nShape: (224,224,3)\nuint8 → float32', fontsize=7)
    draw_box(ax2, 0.5, 6.5, 3, 1.5, COLORS['observation'], 'Depth Image\nShape: (224,224,1)\nfloat32, 0-10m', fontsize=7)
    draw_box(ax2, 0.5, 4, 3, 1.5, '#E74C3C', 'Instruction\n"Go to the\nkitchen"', fontsize=7)
    
    # Encoder
    draw_box(ax2, 4.5, 6, 3, 2, '#3498DB', 'Vision Encoder\n(CLIP/ViT)\nFeature: 768-D', fontsize=8)
    draw_box(ax2, 4.5, 3, 3, 2, '#9B59B6', 'Text Encoder\n(BERT/LSTM)\nFeature: 512-D', fontsize=8)
    
    # Fusion
    draw_box(ax2, 8.5, 4.5, 3, 3, '#8E44AD', 'Multimodal Fusion\n(Cross-Attention)\n512-D fusion', fontsize=8)
    
    # Output branches
    draw_box(ax2, 13, 8, 3.5, 1.8, COLORS['pixel_goal'], 'PIXEL GOAL\n(u, v) coordinates\nShape: (2,) | Range: [0,224]', fontsize=7)
    draw_box(ax2, 13, 5.5, 3.5, 1.8, COLORS['latent'], 'LATENT CODE\nz ∈ ℝⁿ\nn = 256 (typical)', fontsize=7)
    draw_box(ax2, 13, 3, 3.5, 1.8, '#E74C3C', 'Discret Action\n(if text action)\n[e.g., "turn left"]', fontsize=7)
    
    # Arrows with data flow labels
    draw_arrow(ax2, (3.5, 10), (4.5, 7), color=COLORS['observation'])
    draw_arrow(ax2, (3.5, 7.2), (4.5, 7), color=COLORS['observation'])
    draw_arrow(ax2, (3.5, 4.8), (4.5, 4), color=COLORS['s2'])
    draw_arrow(ax2, (7.5, 7), (8.5, 6), color='#3498DB')
    draw_arrow(ax2, (7.5, 4), (8.5, 4.5), color='#9B59B6')
    draw_arrow(ax2, (11.5, 6), (13, 8.8), color='#8E44AD')
    draw_arrow(ax2, (11.5, 6), (13, 6.4), color='#8E44AD')
    draw_arrow(ax2, (11.5, 4.5), (13, 3.9), color='#8E44AD')
    
    # Annotations
    ax2.text(1, 11.2, 'INPUTS', fontsize=10, fontweight='bold', color='#2C3E50')
    ax2.text(5, 11.2, 'ENCODING', fontsize=10, fontweight='bold', color='#2C3E50')
    ax2.text(9.5, 11.2, 'FUSION', fontsize=10, fontweight='bold', color='#2C3E50')
    ax2.text(14, 11.2, 'OUTPUTS', fontsize=10, fontweight='bold', color='#2C3E50')
    
    # =========================================================================
    # Section 3: S1 Processing
    # =========================================================================
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 24)
    ax3.set_ylim(0, 10)
    ax3.set_title('3. S1: VISUAL NAVIGATION POLICY - TRAJECTORY GENERATION', fontsize=14, 
                  fontweight='bold', pad=15, color=COLORS['s1'])
    ax3.axis('off')
    
    # Input branch
    draw_box(ax3, 0.5, 7, 2.5, 1.5, COLORS['observation'], 'Current RGB\n(224,224,3)', fontsize=7)
    draw_box(ax3, 0.5, 5, 2.5, 1.5, COLORS['observation'], 'Current Depth\n(224,224,1)', fontsize=7)
    draw_box(ax3, 0.5, 3, 2.5, 1.5, COLORS['pixel_goal'], 'Pixel Goal\n(u=150, v=80)', fontsize=7)
    draw_box(ax3, 0.5, 1, 2.5, 1.5, COLORS['latent'], 'Latent z\n(256-D)', fontsize=7)
    
    # S1 Network
    draw_box(ax3, 4, 4, 4, 4, '#27AE60', 'S1 Network\n(Diffusion Policy)\n\n• 2-frame history input\n• Cross-attention with z\n• Noise prediction\n• Trajectory output', fontsize=7)
    
    # Intermediate outputs
    draw_box(ax3, 9.5, 7, 3, 1.5, '#1ABC9C', 'Feature Maps\n(256×56×56)', fontsize=7)
    draw_box(ax3, 9.5, 5, 3, 1.5, '#16A085', 'Attention\nWeights', fontsize=7)
    draw_box(ax3, 9.5, 2.5, 3, 1.5, '#1ABC9C', 'Noise Pred\nε_θ', fontsize=7)
    
    # Trajectory
    draw_box(ax3, 14, 4, 4, 4, COLORS['trajectory'], 'TRAJECTORY τ\nShape: (N, 3)\n\nτ = [(dx₁,dy₁,dθ₁),\n     (dx₂,dy₂,dθ₂),\n     ...\n     (dx_N,dy_N,dθ_N)]', fontsize=7)
    
    # Annotations
    draw_arrow(ax3, (3, 7.8), (4, 6), color=COLORS['observation'])
    draw_arrow(ax3, (3, 5.8), (4, 6), color=COLORS['observation'])
    draw_arrow(ax3, (3, 3.8), (4, 4.5), color=COLORS['pixel_goal'])
    draw_arrow(ax3, (3, 1.8), (4, 4), color=COLORS['latent'])
    
    draw_arrow(ax3, (8, 6), (9.5, 7.8), color=COLORS['s1'])
    draw_arrow(ax3, (8, 6), (9.5, 5.8), color=COLORS['s1'])
    draw_arrow(ax3, (8, 4), (9.5, 3.3), color=COLORS['s1'])
    draw_arrow(ax3, (8, 5.5), (14, 6), color=COLORS['s1'])
    
    # =========================================================================
    # Section 4: Action Execution
    # =========================================================================
    ax4 = fig.add_subplot(gs[3])
    ax4.set_xlim(0, 24)
    ax4.set_ylim(0, 10)
    ax4.set_title('4. TRAJECTORY EXECUTION & ENVIRONMENT INTERACTION', fontsize=14, 
                  fontweight='bold', pad=15, color=COLORS['action'])
    ax4.axis('off')
    
    # Trajectory input
    draw_box(ax4, 0.5, 6, 3, 2, COLORS['trajectory'], 'Trajectory τ\n[(0.5,0.2,0.1),\n (0.4,0.3,0.2),\n (0.3,0.1,-0.1)]', fontsize=7)
    
    # Execution loop
    draw_box(ax4, 4.5, 6, 3, 2, COLORS['action'], 'Action δ\n(dx, dy, dθ)\n\nExtracted from τ\none step at time', fontsize=7)
    
    # SE(2) composition
    se2_text = "SE(2) Composition\n\npose_t+1 = pose_t + delta\n\nx_new = x + dx*cos - dy*sin\ny_new = y + dx*sin + dy*cos\ntheta_new = theta + dtheta"
    draw_box(ax4, 8.5, 4.5, 4, 3, '#16A085', se2_text, fontsize=6)
    
    # Environment
    draw_box(ax4, 14, 6, 3, 2, COLORS['env'], 'Environment\n\n• Step simulation\n• Collision check\n• State update', fontsize=7)
    
    # New observation
    draw_box(ax4, 18.5, 6, 3, 2, COLORS['observation'], 'New Obs\n\n• rgb updated\n• depth updated\n• pose updated', fontsize=7)
    
    # Feedback arrow
    ax4.annotate('', xy=(2.5, 6), xytext=(19, 5),
                arrowprops=dict(arrowstyle='->', color=COLORS['env'], lw=2,
                              connectionstyle='arc3,rad=-0.4'))
    ax4.text(10, 4, 'FEEDBACK LOOP', fontsize=10, fontweight='bold', 
             color=COLORS['env'], ha='center')
    
    # Arrows
    draw_arrow(ax4, (3.5, 7), (4.5, 7), color=COLORS['trajectory'])
    draw_arrow(ax4, (7.5, 7), (8.5, 6), color=COLORS['action'])
    draw_arrow(ax4, (12.5, 6), (14, 7), color='#16A085')
    draw_arrow(ax4, (17, 7), (18.5, 7), color=COLORS['env'])
    
    # =========================================================================
    # Section 5: Gap Modes Comparison
    # =========================================================================
    ax5 = fig.add_subplot(gs[4])
    ax5.set_xlim(0, 24)
    ax5.set_ylim(0, 8)
    ax5.set_title('5. ASYNC EXECUTION MODES COMPARISON', fontsize=14, 
                  fontweight='bold', pad=15, color='#2C3E50')
    ax5.axis('off')
    
    # Mode boxes
    modes = [
        (2, 'SYNC (gap=1)', COLORS['s2'], [
            '• S2 every step',
            '• Fresh pixel goal',
            '• Latency: 80ms',
            '• Max traj: 1-3'
        ]),
        (8, 'ASYNC (gap=2)', '#F39C12', [
            '• S2 every 2 steps',
            '• 50% S2 load',
            '• Latency: 40ms',
            '• Max traj: 2-4'
        ]),
        (15, 'ASYNC (gap=8)', COLORS['success'], [
            '• S2 every 8 steps',
            '• 12.5% S2 load',
            '• Latency: 15ms',
            '• Max traj: 4'
        ]),
    ]
    
    for x, title, color, items in modes:
        draw_box(ax5, x, 4.5, 5, 3, color, title, fontsize=9)
        for i, item in enumerate(items):
            ax5.text(x + 0.2, 4.0 - i*0.7, item, fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Timeline comparison
    for mode_idx, (gap, start_x) in enumerate([(1, 2), (2, 9), (8, 16)]):
        for step in range(8):
            if step % gap == 0 or step == 0:
                circle = Circle((start_x + step*0.4, 2.5), 0.15, 
                               facecolor=COLORS['s2'], edgecolor='black')
            else:
                circle = Circle((start_x + step*0.4, 2.5), 0.1, 
                               facecolor='#BDC3C7', edgecolor='gray')
            ax5.add_patch(circle)
    
    ax5.text(6, 2, 'S2 Triggers', fontsize=8, ha='center')
    ax5.text(13, 2, 'S2 Triggers', fontsize=8, ha='center')
    ax5.text(20, 2, 'S2 Triggers', fontsize=8, ha='center')
    
    ax5.text(4.5, 0.5, 'Step: 0  1  2  3  4  5  6  7', fontsize=8)
    ax5.text(11.5, 0.5, 'Step: 0     2     4     6     8', fontsize=8)
    ax5.text(18.5, 0.5, 'Step: 0              8', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fig1_pipeline_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Generated: fig1_pipeline_overview.png")

if __name__ == '__main__':
    create_pipeline_overview()
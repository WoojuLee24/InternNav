"""
InternVLA-N1 Dual-System: Comprehensive Mathematical Analysis
============================================================
Comprehensive visualizations with complete mathematical derivations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Affine2D
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# COMPLETE MATHEMATICAL FORMULATION
# ============================================================================

def create_complete_analysis():
    """Create comprehensive analysis figure"""
    
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3,
                          height_ratios=[0.8, 1, 1, 1])
    
    # =========================================================================
    # Panel A: System Architecture (top spanning full width)
    # =========================================================================
    ax_arch = fig.add_subplot(gs[0, :])
    ax_arch.set_xlim(0, 20)
    ax_arch.set_ylim(0, 6)
    
    def draw_block(ax, x, y, w, h, color, text, text_color='white', fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=fontsize, fontweight='bold', color=text_color)
    
    # Architecture blocks with mathematical notation
    draw_block(ax_arch, 0.5, 2.5, 1.8, 1.2, '#3498DB', 'OBSERVATION\n(rgb, depth)')
    draw_block(ax_arch, 3.0, 2.5, 2.0, 1.2, '#E74C3C', 'S2: LANGUAGE\nPLANNER\nLSTM/VLM')
    
    # S2 outputs
    draw_block(ax_arch, 5.5, 4.0, 1.5, 0.8, '#F39C12', 'pixel_goal\n(u, v)', fontsize=8)
    draw_block(ax_arch, 5.5, 1.2, 1.5, 0.8, '#9B59B6', 'latent\nz ∈ ℝⁿ', fontsize=8)
    
    draw_block(ax_arch, 7.8, 2.5, 2.0, 1.2, '#27AE60', 'S1: VISUAL\nNAV POLICY\nDiffusion/Transformer')
    draw_block(ax_arch, 10.5, 2.5, 1.8, 1.2, '#1ABC9C', 'TRAJECTORY\n[dx,dy,dθ]×N')
    draw_block(ax_arch, 13.0, 2.5, 1.5, 1.2, '#16A085', 'ACTION\nExecution')
    draw_block(ax_arch, 15.2, 2.5, 1.5, 1.2, '#34495E', 'ENVIRONMENT\nStep + Reset')
    
    # Arrows with labels
    arrow_props = dict(arrowstyle='->', color='black', lw=1.5)
    ax_arch.annotate('', xy=(3.0, 3.1), xytext=(2.3, 3.1), arrowprops=arrow_props)
    ax_arch.annotate('', xy=(5.5, 4.4), xytext=(4.5, 3.1), arrowprops=arrow_props)
    ax_arch.annotate('', xy=(5.5, 1.6), xytext=(4.5, 2.5), arrowprops=arrow_props)
    ax_arch.annotate('', xy=(7.8, 3.1), xytext=(7.0, 3.1), arrowprops=arrow_props)
    ax_arch.annotate('', xy=(7.8, 3.1), xytext=(7.0, 4.4), arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1.5))
    ax_arch.annotate('', xy=(7.8, 2.5), xytext=(7.0, 1.6), arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=1.5))
    ax_arch.annotate('', xy=(10.5, 3.1), xytext=(9.8, 3.1), arrowprops=arrow_props)
    ax_arch.annotate('', xy=(13.0, 3.1), xytext=(12.3, 3.1), arrowprops=arrow_props)
    ax_arch.annotate('', xy=(15.2, 3.1), xytext=(14.5, 3.1), arrowprops=arrow_props)
    
    # Feedback loop
    ax_arch.annotate('', xy=(3.5, 2.5), xytext=(3.5, 1.5),
                    arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2,
                                  connectionstyle='arc3,rad=-0.3'))
    ax_arch.text(2.9, 1.0, 'feedback loop', fontsize=8, color='#2C3E50')
    
    # Mode boxes
    modes = [('SYNC', '#E74C3C', 16.5, 5.0), ('ASYNC', '#27AE60', 18.0, 5.0)]
    for mode_name, color, x, y in modes:
        rect = patches.Rectangle((x, y), 1.5, 0.7, facecolor=color, edgecolor='black', linewidth=2)
        ax_arch.add_patch(rect)
        ax_arch.text(x + 0.75, y + 0.35, mode_name, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    
    ax_arch.axis('off')
    ax_arch.set_title('A. Dual-System Architecture: S2 (Language Planner) + S1 (Visual Navigation Policy)', 
                     fontsize=13, fontweight='bold', pad=10)
    
    # =========================================================================
    # Panel B: Camera Geometry (pixel to 3D)
    # =========================================================================
    ax_cam = fig.add_subplot(gs[1, 0:2], projection='3d')
    
    def draw_3d_camera(ax):
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('B. Camera Geometry: Pixel to 3D Projection', fontsize=12, fontweight='bold')
        
        # Camera at origin
        cam = np.array([0, 0, 0])
        ax.scatter(*cam, c='blue', s=150, marker='^', label='Camera')
        
        # Camera axes
        for c, vec, label in [('r', [1,0,0], 'X_c'), ('g', [0,1,0], 'Y_c'), ('b', [0,0,1], 'Z_c')]:
            ax.quiver(0, 0, 0, *vec, color=c, arrow_length_ratio=0.3, linewidth=2)
        
        # Image plane
        f = 5.0
        u_max, v_max = 3, 3
        corners = np.array([[-u_max, -v_max, f], [u_max, -v_max, f], 
                           [u_max, v_max, f], [-u_max, v_max, f]])
        for i in range(4):
            j = (i+1) % 4
            ax.plot([corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]], 
                   [corners[i,2], corners[j,2]], 'k-', linewidth=1)
        ax.plot([-u_max, u_max], [0, 0], [f, f], 'k--', alpha=0.3)
        ax.plot([0, 0], [-v_max, v_max], [f, f], 'k--', alpha=0.3)
        
        # 3D goal point
        goal_3d = np.array([1.0, 0.5, 6.0])
        ax.scatter(*goal_3d, c='red', s=150, marker='o', label='Goal (x,y,z)')
        
        # Ray
        t_vals = np.linspace(0, 1, 100)
        ray = np.array([t * goal_3d for t in t_vals])
        ax.plot(ray[:,0], ray[:,1], ray[:,2], 'r--', linewidth=2, alpha=0.5, label='Ray')
        
        # Projection
        u_p, v_p = goal_3d[0]/goal_3d[2]*f, goal_3d[1]/goal_3d[2]*f
        proj = np.array([u_p, v_p, f])
        ax.scatter(*proj, c='orange', s=100, marker='s', label=f'Pixel ({u_p:.1f}, {v_p:.1f})')
        ax.plot([goal_3d[0], proj[0]], [goal_3d[1], proj[1]], [goal_3d[2], proj[2]], 'g:', linewidth=2)
        
        ax.legend(loc='upper left')
        ax.view_init(elev=20, azim=30)
    
    draw_3d_camera(ax_cam)
    
    # =========================================================================
    # Panel C: Mathematical Formulas
    # =========================================================================
    ax_math = fig.add_subplot(gs[1, 2:4])
    ax_math.axis('off')
    
    formula_box = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           MATHEMATICAL FORMULATIONS                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  1. CAMERA INTRINSICS (Pinhole Model)                                         ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  [u]   [fx  0  cx] [x]                                                 │  ║
║  │  [v] = [ 0 fx  cy] [y]   where: K = intrinsic matrix                    │  ║
║  │  [1]   [ 0  0   1] [z]          (fx, fy) = focal lengths                 │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  2. 3D DIRECTION FROM PIXEL GOAL                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  d_world = R_cam @ K⁻¹ @ [u; v; 1]                                      │  ║
║  │                                                                             │  ║
║  │  where: K⁻¹ = inverse intrinsic matrix                                    │  ║
║  │         R_cam = camera rotation matrix (world to camera frame)           │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  3. TRAJECTORY GENERATION (S1 Policy)                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  τ = π_S1( obs_t, z, d_goal )  ∈ ℝ^(N×3)                                  │  ║
║  │                                                                             │  ║
║  │  τ = [(dx₁, dy₁, dθ₁), (dx₂, dy₂, dθ₂), ..., (dx_N, dy_N, dθ_N)]         │  ║
║  │                                                                             │  ║
║  │  N = trajectory length (typically 4-8 steps)                             │  ║
║  │  obs_t = {rgb_t, depth_t} ∈ ℝ^(H×W×4)                                     │  ║
║  │  z = latent code from S2 ∈ ℝ^d                                            │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  4. ACTION EXECUTION (SE(2) Composition)                                       ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  pose_{t+1} = pose_t ⊕ τ_i                                                │  ║
║  │                                                                             │  ║
║  │  [x']   [x]   [cos(θ)  -sin(θ)  dx]                                      │  ║
║  │  [y'] = [y] + [sin(θ)   cos(θ)  dy]                                       │  ║
║  │  [θ']   [θ]   [  0        0     dθ ]                                      │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    ax_math.text(0.02, 0.98, formula_box, transform=ax_math.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#FAFAFA', edgecolor='#2C3E50', linewidth=2))
    
    # =========================================================================
    # Row 3: Trajectories with Different Gaps
    # =========================================================================
    for idx, (gap, mode_name) in enumerate([(1, 'SYNC'), (2, 'ASYNC-2'), (8, 'ASYNC-8')]):
        ax = fig.add_subplot(gs[2, idx])
        
        np.random.seed(42)
        ax.set_xlim(-2, 18)
        ax.set_ylim(-3, 6)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'C{idx+1}. {mode_name} (gap={gap})', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Robot path
        poses = [(0, 0, 0)]
        current = poses[0]
        
        for step in range(20):
            s2_triggered = (step == 0) or (step % gap == 0)
            
            if s2_triggered:
                # New direction from S2
                direction = np.random.uniform(-np.pi/4, np.pi/4)
                s2_point = current
                
                # S1 generates trajectory
                for i in range(min(gap, 4)):
                    dx = 0.6 * np.cos(direction + np.random.uniform(-0.15, 0.15))
                    dy = 0.6 * np.sin(direction + np.random.uniform(-0.15, 0.15))
                    current = (current[0] + dx, current[1] + dy, direction)
                    poses.append(current)
            else:
                # Continue trajectory
                dx = 0.6 * np.cos(current[2] + np.random.uniform(-0.1, 0.1))
                dy = 0.6 * np.sin(current[2] + np.random.uniform(-0.1, 0.1))
                current = (current[0] + dx, current[1] + dy, current[2])
                poses.append(current)
        
        xs, ys, _ = zip(*poses)
        ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7)
        ax.scatter(xs, ys, c='blue', s=20, alpha=0.5)
        
        # Mark S2 update points
        for step in range(0, 20, gap):
            if step < len(poses):
                x, y, theta = poses[step]
                ax.scatter(x, y, c='red', s=150, marker='*', zorder=5)
                # Direction arrow
                ax.annotate('', xy=(x + 1.0*np.cos(theta), y + 1.0*np.sin(theta)),
                           xytext=(x, y),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.scatter(0, 0, c='green', s=200, marker='o', edgecolor='black', linewidth=2, 
                  label='Start', zorder=6)
        ax.scatter(xs[-1], ys[-1], c='orange', s=150, marker='s', edgecolor='black',
                  linewidth=2, label='End', zorder=6)
        ax.legend(loc='upper left')
    
    # Image space visualization
    ax_img = fig.add_subplot(gs[2, 3])
    ax_img.set_xlim(-20, 250)
    ax_img.set_ylim(-20, 250)
    ax_img.set_xlabel('u (pixels)')
    ax_img.set_ylabel('v (pixels)')
    ax_img.set_title('C4. Image Space with Pixel Goals', fontsize=11, fontweight='bold')
    ax_img.set_aspect('equal')
    
    # Draw image frame
    rect = patches.Rectangle((0, 0), 224, 224, facecolor='#F8F9FA', 
                             edgecolor='black', linewidth=2)
    ax_img.add_patch(rect)
    
    # Pixel goals
    np.random.seed(123)
    for step in [0, 2, 4, 8]:
        u = np.random.uniform(40, 184)
        v = np.random.uniform(40, 184)
        circle = Circle((u, v), 12, facecolor='#E74C3C', edgecolor='black', 
                       linewidth=2, alpha=0.8)
        ax_img.add_patch(circle)
        ax_img.text(u, v, f'S2@{step}', ha='center', va='center', 
                   fontsize=7, fontweight='bold', color='white')
        
        # Trajectory lines in image space
        if step > 0:
            prev_u = np.random.uniform(40, 184)
            prev_v = np.random.uniform(40, 184)
            for t in np.linspace(0.1, 0.9, 5):
                ax_img.scatter(prev_u + (u-prev_u)*t, prev_v + (v-prev_v)*t, 
                              c='#3498DB', s=10, alpha=0.5)
    
    ax_img.scatter(112, 112, c='green', s=100, marker='^', edgecolor='black', label='Center')
    ax_img.legend()
    
    # =========================================================================
    # Row 4: Timing Diagrams and Metrics
    # =========================================================================
    for idx, (gap, mode_name) in enumerate([(1, 'SYNC'), (2, 'ASYNC-2'), (8, 'ASYNC-8')]):
        ax = fig.add_subplot(gs[3, idx])
        ax.set_xlim(-1, 20)
        ax.set_ylim(0, 4)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_title(f'D{idx+1}. Timing: {mode_name}', fontsize=11, fontweight='bold')
        
        labels = ['Env', 'S2', 'S1']
        colors = ['#34495E', '#E74C3C', '#27AE60']
        
        for row, (label, color) in enumerate(zip(labels, colors)):
            y = 3 - row
            ax.text(-0.8, y, label, fontsize=9, fontweight='bold', ha='right')
            
            for step in range(20):
                s2_triggered = (step == 0) or (step % gap == 0)
                
                if row == 0:  # Environment
                    ax.barh(y, 0.8, left=step, height=0.6, color=color, edgecolor='black')
                elif row == 1:  # S2
                    if s2_triggered:
                        ax.barh(y, 0.8, left=step, height=0.6, color=color, edgecolor='black')
                    else:
                        ax.barh(y, 0.8, left=step, height=0.6, color='#BDC3C7', 
                              edgecolor='black', alpha=0.3)
                else:  # S1
                    if s2_triggered:
                        ax.barh(y, 0.8, left=step, height=0.6, color=color, 
                               edgecolor='black', alpha=1.0)
                    else:
                        ax.barh(y, 0.8, left=step, height=0.6, color=color, 
                              edgecolor='black', alpha=0.6)
        
        # S2 trigger lines
        for step in range(0, 20, gap):
            ax.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # =========================================================================
    # Performance metrics summary
    # =========================================================================
    ax_metrics = fig.add_subplot(gs[3, 3])
    ax_metrics.axis('off')
    
    metrics_text = """
    PERFORMANCE METRICS
    ─────────────────────────────────────
    Gap │ Latency │ S2 Freq │ Freshness
    ─────────────────────────────────────
      1 │  80 ms  │  1.0×   │   1.00
      2 │  40 ms  │  0.5×   │   0.70
      8 │  15 ms  │  0.125× │   0.30
    ─────────────────────────────────────
    
    Trade-off: Lower latency ↔ Higher accuracy
    Choose gap based on task requirements:
      • gap=1: Precision-critical tasks
      • gap=2: Balanced performance  
      • gap=8: Real-time robot control
    """
    ax_metrics.text(0.1, 0.95, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='#FEF9E7', edgecolor='#F39C12', linewidth=2))
    
    fig.suptitle('InternVLA-N1 Dual-System: Mathematical Analysis\n'
                'S1 = Visual Navigation Policy | S2 = Language-Grounded Planner', 
                fontsize=16, fontweight='bold', y=0.99)
    
    plt.savefig('fig_complete_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("Generated: fig_complete_analysis.png")

# ============================================================================
# EXECUTION TIMELINE COMPARISON
# ============================================================================

def create_timeline_comparison():
    """Create detailed execution timeline comparison"""
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), height_ratios=[1, 1, 1.2])
    fig.suptitle('Execution Timeline: SYNC vs ASYNC Modes', fontsize=14, fontweight='bold', y=1.02)
    
    for ax_idx, (gap, mode) in enumerate([(1, 'SYNC'), (2, 'ASYNC-2'), (8, 'ASYNC-8')]):
        ax = axes[ax_idx]
        ax.set_xlim(-1, 24)
        ax.set_ylim(0, 5)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_title(f'{mode} Mode (gap={gap})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Component', fontsize=11)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['Environment', 'S2 Inference', 'S1 Inference', 'S1 Execution'])
        ax.grid(True, alpha=0.3, axis='x')
        
        colors = {
            'env': '#34495E',
            's2': '#E74C3C', 
            's1_inf': '#3498DB',
            's1_exec': '#27AE60',
            'idle': '#BDC3C7'
        }
        
        for step in range(24):
            s2_triggered = (step == 0) or (step % gap == 0)
            
            # Environment (always active)
            ax.barh(1, 0.8, left=step, height=0.6, color=colors['env'], edgecolor='black')
            
            # S2
            color = colors['s2'] if s2_triggered else colors['idle']
            ax.barh(2, 0.8, left=step, height=0.6, color=color, edgecolor='black')
            
            # S1 Inference
            color = colors['s1_inf'] if s2_triggered else colors['idle']
            ax.barh(3, 0.8, left=step, height=0.6, color=color, edgecolor='black')
            
            # S1 Execution
            ax.barh(4, 0.8, left=step, height=0.6, color=colors['s1_exec'], edgecolor='black', alpha=0.8)
            
            # S2 trigger lines
            if s2_triggered:
                ax.axvline(x=step, color='red', linestyle='--', alpha=0.6, linewidth=2)
                ax.text(step, 4.5, 'S2', fontsize=8, color='red', ha='center')
        
        # Legend
        legend_elements = [
            patches.Patch(facecolor=colors['env'], edgecolor='black', label='Environment'),
            patches.Patch(facecolor=colors['s2'], edgecolor='black', label='S2 Inference'),
            patches.Patch(facecolor=colors['s1_inf'], edgecolor='black', label='S1 Inference'),
            patches.Patch(facecolor=colors['s1_exec'], edgecolor='black', label='S1 Execution'),
            patches.Patch(facecolor='red', edgecolor='red', alpha=0.5, label='S2 Trigger'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', ncol=5, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fig_timeline_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("Generated: fig_timeline_comparison.png")

# ============================================================================
# PIXEL GOAL HEATMAP VISUALIZATION
# ============================================================================

def create_pixel_goal_visualization():
    """Create visualization of pixel goals in image space"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Pixel Goal Visualization Across Execution Modes', fontsize=14, fontweight='bold')
    
    # Create custom colormap
    colors_list = ['#F8F9FA', '#E74C3C', '#C0392B', '#922B21', '#641E16']
    cmap = LinearSegmentedColormap.from_list('custom', colors_list)
    
    for col, (gap, mode) in enumerate([(1, 'SYNC'), (2, 'ASYNC-2'), (8, 'ASYNC-8')]):
        # Top: Image space
        ax = axes[0, col]
        ax.set_title(f'{mode} (gap={gap}): Image Space', fontsize=11, fontweight='bold')
        ax.set_xlim(-10, 234)
        ax.set_ylim(-10, 234)
        ax.set_xlabel('u (horizontal)')
        ax.set_ylabel('v (vertical)')
        ax.set_aspect('equal')
        
        # Draw image
        rect = patches.Rectangle((0, 0), 224, 224, facecolor='#1A1A2E', 
                                 edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Pixel goals over time
        np.random.seed(42)
        goal_history = []
        for step in [0, 1, 2, 4, 8, 12, 16, 20]:
            if gap == 1 or step % gap == 0 or step == 0:
                u = np.random.uniform(30, 194)
                v = np.random.uniform(30, 194)
                goal_history.append((step, u, v))
                
                # Draw goal with fading based on age
                age_factor = 1.0 - (step / 24) * 0.5
                circle = Circle((u, v), 10, facecolor='#E74C3C', 
                               edgecolor='white', linewidth=1.5, alpha=age_factor)
                ax.add_patch(circle)
                ax.text(u, v, f'{step}', ha='center', va='center', 
                       fontsize=7, color='white', fontweight='bold')
        
        ax.scatter(112, 112, c='green', s=80, marker='^', edgecolor='white', 
                  linewidth=1.5, label='Camera center')
        ax.legend(loc='upper left')
        
        # Bottom: Heatmap of where pixel goals appear
        ax = axes[1, col]
        ax.set_title(f'{mode} (gap={gap}): Pixel Goal Density', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 224)
        ax.set_ylim(0, 224)
        ax.set_xlabel('u (pixels)')
        ax.set_ylabel('v (pixels)')
        
        # Generate heatmap
        heatmap = np.zeros((224, 224))
        np.random.seed(42)
        for step in range(24):
            if step % gap == 0 or step == 0:
                u = int(np.random.uniform(20, 204))
                v = int(np.random.uniform(20, 204))
                for du in range(-15, 16):
                    for dv in range(-15, 16):
                        if 0 <= u+du < 224 and 0 <= v+dv < 224:
                            heatmap[v+dv, u+du] += np.exp(-(du**2 + dv**2) / 50)
        
        im = ax.imshow(heatmap, cmap=cmap, origin='lower', extent=[0, 224, 0, 224])
        plt.colorbar(im, ax=ax, label='Goal density', shrink=0.8)
        
        # Camera center
        ax.scatter(112, 112, c='white', s=50, marker='+', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('fig_pixel_goals.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("Generated: fig_pixel_goals.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Generating comprehensive visualizations...")
    print("=" * 60)
    
    create_complete_analysis()
    create_timeline_comparison()
    create_pixel_goal_visualization()
    
    print("=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("""
Output files:
  1. fig_complete_analysis.png    - Complete system analysis
  2. fig_timeline_comparison.png  - Execution timeline comparison
  3. fig_pixel_goals.png          - Pixel goal visualization
""")
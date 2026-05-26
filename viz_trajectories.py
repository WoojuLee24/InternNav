"""
InternVLA-N1: Trajectory & Action Visualization
================================================
Detailed visualization of trajectories across different modes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow, Wedge, Arc
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FIGURE 5: Trajectory Visualization - Comprehensive
# ============================================================================
def create_trajectory_comprehensive():
    """Comprehensive trajectory visualization"""
    
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25)
    
    fig.suptitle('TRAJECTORY VISUALIZATION: FROM PIXEL GOAL TO ROBOT MOTION', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # =========================================================================
    # Row 1: Pixel Goal to Trajectory Mapping
    # =========================================================================
    
    # Panel A: Image Space - Pixel Goal
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 224)
    ax.set_ylim(0, 224)
    ax.set_title('A. Image Space: Pixel Goal Selection', fontsize=11, fontweight='bold')
    ax.set_xlabel('u (horizontal pixels)')
    ax.set_ylabel('v (vertical pixels)')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Draw grid background
    for i in range(0, 225, 28):
        ax.axhline(i, color='gray', alpha=0.2, linewidth=0.5)
        ax.axvline(i, color='gray', alpha=0.2, linewidth=0.5)
    
    # Current view
    rect = patches.Rectangle((0, 0), 224, 224, facecolor='#1A1A2E', 
                             edgecolor='black', linewidth=3)
    ax.add_patch(rect)
    
    # Simulated scene elements
    elements = [(50, 50, 'kitchen'), (170, 40, 'door'), (180, 180, 'table'), 
               (30, 180, 'chair'), (100, 100, 'center')]
    for x, y, name in elements:
        circle = Circle((x, y), 15, facecolor='#3498DB', edgecolor='white', alpha=0.5)
        ax.add_patch(circle)
        ax.text(x, y, name[:3], ha='center', va='center', fontsize=6, color='white')
    
    # Pixel goal
    goal_u, goal_v = 150, 60
    circle = Circle((goal_u, goal_v), 12, facecolor='#E74C3C', edgecolor='white', 
                    linewidth=3, alpha=0.9)
    ax.add_patch(circle)
    ax.plot(goal_u, goal_v, 'w*', markersize=25, markeredgecolor='black')
    ax.annotate(f'Pixel Goal\n(u={goal_u}, v={goal_v})', (goal_u, goal_v), 
               xytext=(goal_u + 30, goal_v - 20), fontsize=9, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))
    
    # Camera center
    ax.scatter(112, 112, c='green', s=80, marker='^', edgecolor='white', 
               linewidth=2, label='Camera')
    ax.legend(loc='upper left')
    
    # =========================================================================
    # Panel B: Ray Projection in 3D
    # ax = fig.add_subplot(gs[0, 1], projection='3d')
    # =========================================================================
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 15)
    ax.set_title('B. 3D Ray Projection', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_aspect('equal')
    
    # Camera
    ax.scatter(0, 0, c='green', s=100, marker='^', label='Camera', zorder=5)
    
    # Camera FOV
    fov_angles = np.linspace(-np.pi/6, np.pi/6, 20)
    fov_x = np.tan(fov_angles) * 10
    fov_z = np.ones(20) * 10
    ax.fill_between(fov_z, -fov_x, fov_x, alpha=0.2, color='blue', label='FOV')
    
    # Image plane
    ax.plot([-2, 2, 2, -2, -2], [10, 10, 10, 10, 10], [-2, -2, 2, 2, -2], 'b-', linewidth=2)
    
    # 3D goal position
    goal_3d = np.array([1.5, 0, 8])
    ax.scatter(goal_3d[0], goal_3d[2], c='red', s=150, marker='o', zorder=6)
    
    # Ray from camera to goal
    ax.plot([0, goal_3d[0]], [0, goal_3d[2]], 'r--', linewidth=2, label='Ray to goal')
    
    # Projection point
    f = 5.0
    proj_z = 5
    proj_x = goal_3d[0] * proj_z / goal_3d[2]
    ax.scatter(proj_x, proj_z, c='orange', s=80, marker='s', label='Image plane')
    
    # Annotations
    ax.text(goal_3d[0] + 0.3, goal_3d[2], 'Goal\n(x=1.5, z=8)', fontsize=8)
    ax.text(proj_x + 0.3, proj_z, f'Pixel proj\n(u={proj_x:.1f})', fontsize=8)
    
    # =========================================================================
    # Panel C: Trajectory Decomposition
    # =========================================================================
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 5)
    ax.set_title('C. Trajectory Decomposition', fontsize=11, fontweight='bold')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)
    
    # Trajectory points
    traj = [(0, 0, 0), (0.5, 0.2, 0.1), (0.9, 0.4, 0.2), (1.3, 0.6, 0.1), (1.7, 0.7, 0.0)]
    xs, ys, _ = zip(*traj)
    
    ax.plot(xs, ys, 'g-', linewidth=3, marker='o', markersize=10, label='Trajectory')
    
    # Decompose into components
    for i, (x, y, theta) in enumerate(traj):
        if i > 0:
            prev_x, prev_y = traj[i-1][0], traj[i-1][1]
            ax.arrow(prev_x, prev_y, x - prev_x, y - prev_y, 
                    head_width=0.08, head_length=0.05, fc='blue', ec='black')
            ax.text((prev_x + x)/2, (prev_y + y)/2 + 0.15, f'dx={x-prev_x:.2f}', 
                   fontsize=7, ha='center', color='blue')
            ax.text((prev_x + x)/2, (prev_y + y)/2 - 0.2, f'dy={y-prev_y:.2f}', 
                   fontsize=7, ha='center', color='red')
    
    ax.text(5, -1.5, 'τ = [(dx₁,dy₁,dθ₁), (dx₂,dy₂,dθ₂), ...]', fontsize=10,
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='yellow'))
    
    # =========================================================================
    # Panel D: SE(2) Pose Update
    # =========================================================================
    ax = fig.add_subplot(gs[0, 3])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_title('D. SE(2) Pose Update', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Formula box
    formula = """
    SE(2) POSE COMPOSITION
    ───────────────────────────
    
    Current pose:
    ┌     ┐
    │ x   │
    │ y   │ ∈ SE(2)
    ��� θ   │
    └     ┘
    
    Action delta:
    ┌     ┐
    │ dx  │
    │ dy  │ ∈ ℝ³
    │ dθ  │
    └     ┘
    
    Update:
    ┌ x' ┐   ┌ x ┐   ┌ cosθ  -sinθ  dx ┐
    │ y' │ = │ y │ + │ sinθ   cosθ  dy │
    │ θ' │   │ θ │   │   0       0  dθ │
    └    ┘   └ ┘     └               ┘
    """
    ax.text(0.1, 0.95, formula, transform=ax.transAxes, fontsize=9,
           fontfamily='monospace', verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))
    
    # =========================================================================
    # Row 2: Trajectories with Different Gaps
    # =========================================================================
    
    for idx, (gap, mode) in enumerate([(1, 'SYNC'), (2, 'ASYNC-2'), (8, 'ASYNC-8')]):
        ax = fig.add_subplot(gs[1, idx])
        ax.set_xlim(-2, 20)
        ax.set_ylim(-3, 6)
        ax.set_title(f'{mode} (gap={gap}) - Workspace Trajectory', fontsize=11, fontweight='bold')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        np.random.seed(42)
        poses = [(0, 0, 0)]
        current = poses[0]
        
        for step in range(20):
            s2_triggered = (step == 0) or (step % gap == 0)
            
            if s2_triggered:
                direction = np.random.uniform(-np.pi/4, np.pi/4)
                s2_point = current
                
                for i in range(min(gap, 4)):
                    dx = 0.6 * np.cos(direction + np.random.uniform(-0.15, 0.15))
                    dy = 0.6 * np.sin(direction + np.random.uniform(-0.15, 0.15))
                    current = (current[0] + dx, current[1] + dy, direction)
                    poses.append(current)
            else:
                dx = 0.6 * np.cos(current[2] + np.random.uniform(-0.1, 0.1))
                dy = 0.6 * np.sin(current[2] + np.random.uniform(-0.1, 0.1))
                current = (current[0] + dx, current[1] + dy, current[2])
                poses.append(current)
        
        xs, ys, thetas = zip(*poses)
        
        # Draw trajectory
        ax.plot(xs, ys, 'b-', linewidth=2.5, alpha=0.7, label='Path')
        ax.scatter(xs, ys, c='blue', s=30, alpha=0.5)
        
        # S2 update points (star markers)
        for step in range(0, 20, gap):
            if step < len(poses):
                x, y, theta = poses[step]
                ax.scatter(x, y, c='red', s=200, marker='*', edgecolor='black', 
                           linewidth=1.5, zorder=5)
                # Direction arrow
                ax.annotate('', xy=(x + 1.2*np.cos(theta), y + 1.2*np.sin(theta)),
                           xytext=(x, y),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
                ax.text(x, y - 0.4, f'{step}', fontsize=7, ha='center', color='red')
        
        ax.scatter(0, 0, c='green', s=200, marker='o', edgecolor='black', 
                  linewidth=2, label='Start', zorder=6)
        ax.scatter(xs[-1], ys[-1], c='orange', s=150, marker='s', edgecolor='black',
                  linewidth=2, label='End', zorder=6)
        ax.legend(loc='upper left', fontsize=8)
    
    # =========================================================================
    # Row 3: Action Execution Timeline
    # =========================================================================
    ax = fig.add_subplot(gs[2, :])
    ax.set_xlim(-1, 25)
    ax.set_ylim(0, 5)
    ax.set_title('Action Execution Timeline: Comparing Gap Modes', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['SYNC\n(gap=1)', 'ASYNC-2\n(gap=2)', 'ASYNC-8\n(gap=8)', 'Legend'])
    ax.grid(True, alpha=0.3, axis='x')
    
    colors = {
        's2': '#E74C3C',
        's1': '#27AE60',
        'exec': '#1ABC9C',
        'idle': '#BDC3C7'
    }
    
    for row, (gap, label) in enumerate([(1, 'SYNC'), (2, 'ASYNC-2'), (8, 'ASYNC-8')]):
        y = 4 - row
        ax.text(-0.8, y, '', fontsize=9, fontweight='bold')
        
        for step in range(24):
            s2_triggered = (step == 0) or (step % gap == 0)
            
            if s2_triggered:
                # S2 inference
                rect = patches.Rectangle((step, y + 0.3), 0.9, 0.4, 
                                        facecolor=colors['s2'], edgecolor='black')
                ax.add_patch(rect)
                ax.text(step + 0.45, y + 0.5, 'S2', fontsize=6, ha='center', 
                       va='center', fontweight='bold', color='white')
                
                # S1 inference
                rect = patches.Rectangle((step, y + 0.7), 0.9, 0.25, 
                                        facecolor=colors['s1'], edgecolor='black')
                ax.add_patch(rect)
                
                # Trajectory steps
                traj_len = min(gap, 4)
                for i in range(traj_len):
                    if step + i < 24:
                        rect = patches.Rectangle((step + i, y - 0.1), 0.9, 0.3, 
                                                facecolor=colors['exec'], edgecolor='black', 
                                                alpha=1.0 - i*0.15)
                        ax.add_patch(rect)
                        if i == 0:
                            ax.text(step + i + 0.45, y + 0.05, f'T{i}', fontsize=5, 
                                   ha='center', va='center', color='white')
            else:
                # Idle step
                rect = patches.Rectangle((step, y + 0.3), 0.9, 0.4, 
                                        facecolor=colors['idle'], edgecolor='black', alpha=0.3)
                ax.add_patch(rect)
    
    # Legend
    legend_items = [
        ('S2', colors['s2']),
        ('S1 Infer', colors['s1']),
        ('Traj Exec', colors['exec']),
        ('Idle', colors['idle']),
    ]
    for i, (name, color) in enumerate(legend_items):
        rect = patches.Rectangle((20 + i*1.2, 3.5), 0.8, 0.4, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(20 + i*1.2 + 0.4, 3.2, name, fontsize=6, ha='center')
    
    plt.tight_layout()
    plt.savefig('fig5_trajectory_comprehensive.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Generated: fig5_trajectory_comprehensive.png")

# ============================================================================
# FIGURE 6: Pixel Goal Effects on Trajectories
# ============================================================================
def create_pixel_goal_effects():
    """Visualize how different pixel goals affect trajectories"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EFFECT OF PIXEL GOAL ON TRAJECTORY GENERATION', 
                fontsize=16, fontweight='bold')
    
    # Different pixel goal scenarios
    scenarios = [
        ('Goal: Left', 50, 112, 'left'),
        ('Goal: Center', 112, 112, 'center'),
        ('Goal: Right', 174, 112, 'right'),
        ('Goal: Near', 112, 150, 'near'),
        ('Goal: Far', 112, 30, 'far'),
        ('Goal: Corner', 170, 170, 'corner'),
    ]
    
    for idx, (scenario, u, v, key) in enumerate(scenarios):
        ax = axes[idx // 3, idx % 3]
        
        if idx < 3:
            # Top row: Image space
            ax.set_title(f'{scenario}', fontsize=11, fontweight='bold')
            ax.set_xlim(0, 224)
            ax.set_ylim(0, 224)
            ax.set_xlabel('u (pixels)')
            ax.set_ylabel('v (pixels)')
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # Background
            rect = patches.Rectangle((0, 0), 224, 224, facecolor='#1A1A2E', 
                                     edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Goal point
            circle = Circle((u, v), 15, facecolor='#E74C3C', edgecolor='white', 
                           linewidth=2, alpha=0.9)
            ax.add_patch(circle)
            ax.plot(u, v, 'w*', markersize=20, markeredgecolor='black')
            
            # Camera center
            ax.scatter(112, 112, c='green', s=60, marker='^', edgecolor='white')
            
            # Direction indicator
            dx = (u - 112) / 50
            dy = (v - 112) / 50
            ax.annotate('', xy=(u, v), xytext=(112, 112),
                       arrowprops=dict(arrowstyle='->', color='yellow', lw=2))
            
        else:
            # Bottom row: Workspace trajectory
            ax.set_title(f'{scenario}', fontsize=11, fontweight='bold')
            ax.set_xlim(-2, 8)
            ax.set_ylim(-2, 5)
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Generate trajectory based on goal direction
            if key == 'left':
                direction = np.pi / 4
            elif key == 'right':
                direction = -np.pi / 4
            elif key == 'center':
                direction = 0
            elif key == 'near':
                direction = np.pi / 6
            elif key == 'far':
                direction = -np.pi / 6
            else:  # corner
                direction = np.pi / 3
            
            np.random.seed(42)
            poses = [(0, 0, 0)]
            current = poses[0]
            
            for step in range(12):
                dx = 0.5 * np.cos(direction + np.random.uniform(-0.1, 0.1))
                dy = 0.5 * np.sin(direction + np.random.uniform(-0.1, 0.1))
                current = (current[0] + dx, current[1] + dy, direction)
                poses.append(current)
            
            xs, ys, _ = zip(*poses)
            ax.plot(xs, ys, 'b-', linewidth=2.5, alpha=0.7)
            ax.scatter(xs, ys, c='blue', s=30, alpha=0.5)
            ax.scatter(0, 0, c='green', s=100, marker='o', edgecolor='black', label='Start')
            ax.scatter(xs[-1], ys[-1], c='red', s=100, marker='*', edgecolor='black', label='Goal')
            
            # Direction arrow
            ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            
            ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fig6_pixel_goal_effects.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Generated: fig6_pixel_goal_effects.png")

# ============================================================================
# FIGURE 7: State Evolution & Mode Selection
# ============================================================================
def create_state_evolution():
    """Visualize state evolution and mode selection"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    fig.suptitle('STATE EVOLUTION & MODE SELECTION', fontsize=16, fontweight='bold')
    
    # =========================================================================
    # Panel 1-3: State Variables Over Time
    # =========================================================================
    for idx, (gap, mode) in enumerate([(1, 'SYNC'), (2, 'ASYNC-2'), (8, 'ASYNC-8')]):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_title(f'{mode} (gap={gap})', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('State Value')
        
        steps = np.arange(20)
        
        # dual_forward_step
        dual_step = np.zeros(20)
        current = 0
        for i, step in enumerate(steps):
            s2_triggered = (step == 0) or (step % gap == 0)
            if s2_triggered:
                current = 0
            else:
                current += 1
            dual_step[i] = current
        
        ax.plot(steps, dual_step, 'b-', linewidth=2, marker='o', label='dual_forward_step')
        ax.axhline(y=gap, color='red', linestyle='--', label=f'threshold (={gap})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Mark S2 triggers
        for step in range(20):
            if step % gap == 0 or step == 0:
                ax.axvline(x=step, color='green', linestyle=':', alpha=0.5)
        
        ax.text(0.02, 0.95, 'dual_forward_step resets on S2', transform=ax.transAxes,
               fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # =========================================================================
    # Panel 4: Mode Decision Logic
    # =========================================================================
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.set_title('Mode Selection Decision Logic', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Decision boxes
    decisions = [
        (2, 6, '#3498DB', 'episode_step == 0?', 'S2 triggered'),
        (6, 6, '#E74C3C', 's2_output.is_infering?', 'Wait (sleep)'),
        (10, 6, '#27AE60', 'dual_forward_step >= gap?', 'S2 triggered'),
        (14, 6, '#9B59B6', 'output_action != None?', 'Return discrete'),
        (2, 3, '#F39C12', 'output_latent != None?', 'S1 trajectory'),
        (6, 3, '#16A085', 'mode == "sync"?', 'Direct S1'),
        (10, 3, '#1ABC9C', 'mode == "partial_async"?', 'Async S1'),
        (14, 3, '#E74C3C', 'Execute action', 'Step env'),
    ]
    
    for x, y, color, text, next_text in decisions:
        rect = FancyBboxPatch((x, y), 3, 1.5, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.5, y + 1, text, ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
        ax.text(x + 1.5, y + 0.3, f'→ {next_text}', ha='center', va='center', 
               fontsize=6, color='white')
    
    # =========================================================================
    # Panel 5: Performance Comparison
    # =========================================================================
    ax = fig.add_subplot(gs[2, :])
    ax.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    
    metrics = ['Latency\n(ms/step)', 'S2 Update\nFrequency', 'Trajectory\nFreshness', 
               'Path\nDeviation', 'Computation\nCost']
    
    gap1_vals = [80, 100, 100, 0, 100]
    gap2_vals = [40, 50, 70, 5, 50]
    gap8_vals = [15, 12.5, 30, 15, 12.5]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = ax.bar(x - width, gap1_vals, width, label='SYNC (gap=1)', color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x, gap2_vals, width, label='ASYNC (gap=2)', color='#F39C12', alpha=0.8)
    bars3 = ax.bar(x + width, gap8_vals, width, label='ASYNC (gap=8)', color='#27AE60', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('fig7_state_evolution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Generated: fig7_state_evolution.png")

if __name__ == '__main__':
    create_trajectory_comprehensive()
    create_pixel_goal_effects()
    create_state_evolution()
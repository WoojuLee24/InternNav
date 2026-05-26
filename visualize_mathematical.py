"""
InternVLA-N1 Dual-System: Mathematical Visualization
===================================================
S1: Visual Navigation Policy (generates trajectories from pixel goals)
S2: Language-Grounded Planner (outputs pixel goals in image coordinates)

Mathematical Model:
- Pixel Goal: (u, v) in image coordinates → converted to 3D goal via camera intrinsics
- Trajectory: sequence of action deltas [dx, dy, dθ]
- Async gap: number of S1 steps per S2 update
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge, Arc
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Affine2D
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def draw_coordinate_frame(ax, origin, angle=0, scale=1.0, label=''):
    """Draw coordinate frame with x (red), y (green) axes"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ax.arrow(origin[0], origin[1], scale*cos_a*0.8, scale*sin_a*0.8,
            head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.arrow(origin[0], origin[1], scale*-sin_a*0.8, scale*cos_a*0.8,
            head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)
    ax.text(origin[0] + scale*cos_a*1.0, origin[1] + scale*sin_a*1.0, 'X', 
            fontsize=12, fontweight='bold', color='red')
    ax.text(origin[0] - scale*sin_a*1.0, origin[1] + scale*cos_a*1.0, 'Y', 
            fontsize=12, fontweight='bold', color='green')

# ============================================================================
# FIGURE 1: Complete System Architecture with Data Flow
# ============================================================================
fig1 = plt.figure(figsize=(20, 14))
gs1 = gridspec.GridSpec(3, 4, figure=fig1, hspace=0.4, wspace=0.3,
                        height_ratios=[1, 1.2, 1])

# Row 1: System Block Diagram
ax1 = fig1.add_subplot(gs1[0, :])

# Define blocks
blocks = {
    'obs': {'pos': (0.5, 0.5), 'size': (1.2, 0.8), 'color': '#3498DB', 
            'text': 'OBSERVATION\n(rgb, depth)', 'y_text': 0.65},
    's2': {'pos': (2.2, 0.5), 'size': (1.5, 0.8), 'color': '#E74C3C', 
           'text': 'S2: LANGUAGE\nPLANNER', 'y_text': 0.65},
    'pixel_goal': {'pos': (4.2, 0.5), 'size': (1.2, 0.8), 'color': '#F39C12', 
                   'text': 'PIXEL GOAL\n(u, v)', 'y_text': 0.65},
    'latent': {'pos': (4.2, 1.8), 'size': (1.2, 0.6), 'color': '#9B59B6', 
               'text': 'LATENT\nz', 'y_text': 1.95},
    's1': {'pos': (6.0, 0.5), 'size': (1.5, 0.8), 'color': '#27AE60', 
           'text': 'S1: VISUAL\nNAV POLICY', 'y_text': 0.65},
    'trajectory': {'pos': (8.0, 0.5), 'size': (1.4, 0.8), 'color': '#1ABC9C', 
                  'text': 'TRAJECTORY\n[dx,dy,dθ]×N', 'y_text': 0.65},
    'action': {'pos': (9.8, 0.5), 'size': (1.0, 0.8), 'color': '#16A085', 
               'text': 'ACTION', 'y_text': 0.65},
    'env': {'pos': (11.3, 0.5), 'size': (1.4, 0.8), 'color': '#34495E', 
            'text': 'ENVIRONMENT\n(Step)', 'y_text': 0.65},
}

for name, props in blocks.items():
    x, y = props['pos']
    w, h = props['size']
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                          facecolor=props['color'], edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x + w/2, y + props['y_text'], props['text'], 
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrows
arrows = [
    ('obs', 's2', 'rgb, depth'),
    ('s2', 'pixel_goal', 'pixel (u,v)'),
    ('s2', 'latent', 'latent z'),
    ('pixel_goal', 's1', 'goal'),
    ('latent', 's1', 'z'),
    ('s1', 'trajectory', 'traj'),
    ('trajectory', 'action', 'exec'),
    ('action', 'env', ''),
    ('env', 'obs', 'new obs', 'loop'),
]

for arrow in arrows:
    if len(arrow) == 4 and arrow[3] == 'loop':
        # Feedback loop
        ax1.annotate('', xy=(0.5, 1.8), xytext=(0.5, 0.5),
                    arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2, 
                                  connectionstyle='arc3,rad=0.3'))
        ax1.text(0.0, 1.15, 'feedback', fontsize=8, color='#2C3E50', rotation=90)
    else:
        x1 = arrow[0]
        x2 = arrow[1]
        ax1.annotate('', xy=(blocks[x2]['pos'][0], blocks[x2]['pos'][1] + 0.4),
                    xytext=(blocks[x1]['pos'][0] + blocks[x1]['size'][0], 
                           blocks[x1]['pos'][1] + 0.4),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax1.set_xlim(-0.5, 13)
ax1.set_ylim(-0.5, 3)
ax1.axis('off')
ax1.set_title('Fig 1: InternVLA-N1 Dual-System Architecture\nS1 = Visual Navigation Policy | S2 = Language-Grounded Planner', 
             fontsize=14, fontweight='bold', pad=20)

# Row 2: Camera Geometry and Pixel Goal Projection
ax2 = fig1.add_subplot(gs1[1, 0:2], projection='3d')
ax3 = fig1.add_subplot(gs1[1, 2:4])

# 3D camera model
def draw_camera_model(ax):
    ax.set_xlabel('X (world)')
    ax.set_ylabel('Y (world)')
    ax.set_zlabel('Z (depth)')
    
    # Camera position
    cam_pos = np.array([0, 0, 0])
    ax.scatter(*cam_pos, c='blue', s=100, marker='^')
    ax.text(0, 0, 0.3, 'Camera', fontsize=10)
    
    # Image plane (at z = focal_length)
    f = 5.0
    plane_size = 2
    corners = np.array([
        [-plane_size/2, -plane_size/2, f],
        [plane_size/2, -plane_size/2, f],
        [plane_size/2, plane_size/2, f],
        [-plane_size/2, plane_size/2, f],
        [-plane_size/2, -plane_size/2, f],
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'b-', linewidth=2)
    
    # Pixel goal point in 3D
    goal_3d = np.array([0.5, 0.3, 8.0])  # Point at depth 8m
    ax.scatter(*goal_3d, c='red', s=150, marker='o')
    ax.text(goal_3d[0]+0.2, goal_3d[1], goal_3d[2], 'Goal (x,y,z)', fontsize=10)
    
    # Ray from camera through pixel
    ray_points = np.array([cam_pos, goal_3d])
    ax.plot(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], 'r--', linewidth=2)
    
    # Projection to image plane
    u_proj, v_proj = goal_3d[0] * f / goal_3d[2], goal_3d[1] * f / goal_3d[2]
    proj_point = np.array([u_proj, v_proj, f])
    ax.scatter(*proj_point, c='orange', s=100, marker='s')
    ax.text(u_proj, v_proj, f+0.3, f'Pixel ({u_proj:.1f}, {v_proj:.1f})', fontsize=9)
    
    # Draw projection lines
    proj_line = np.array([goal_3d, proj_point])
    ax.plot(proj_line[:, 0], proj_line[:, 1], proj_line[:, 2], 'g:', linewidth=2)
    
    ax.set_title('3D Camera Model\nPerspective Projection', fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)

draw_camera_model(ax2)

# Mathematical formulas box
ax3.axis('off')
formula_text = """
MATHEMATICAL MODEL
═══════════════════════════════════════════════════════════

1. CAMERA PROJECTION (Pinhole Model)
────────────────────────────────────────────
  u = fx × (x / z) + cx
  v = fy × (y / z) + cy
  
  where:  fx, fy = focal lengths (pixels)
          cx, cy = principal point
          (x, y, z) = 3D point in camera frame

2. PIXEL GOAL TO 3D DIRECTION
────────────────────────────────────────────
  d_world = Rotation_matrix @ inv(K) @ [u; v; 1]
  
  where:  K = camera intrinsic matrix
          Rotation_matrix = camera orientation

3. TRAJECTORY GENERATION (S1 Policy)
────────────────────────────────────────────
  τ = f_S1(obs_t, pixel_goal, latent_z)
  
  where:  τ = [Δx₁, Δy₁, Δθ₁, Δx₂, Δy₂, Δθ₂, ...]
          obs_t = current RGB-D observation
          pixel_goal = S2 output (u, v)
          latent_z = learned latent representation

4. ACTION EXECUTION
────────────────────────────────────────────
  pose_{t+1} = pose_t ⊕ τ_i
  
  where:  ⊕ = SE(2) composition operator
          τ_i = i-th step of trajectory
"""
ax3.text(0.05, 0.95, formula_text, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#ECF0F1', edgecolor='black'))

def draw_timing_diagram(ax, gap, mode):
    """Draw timing diagram showing S1/S2 execution over time"""
    ax.set_title(f'{mode}', fontsize=11, fontweight='bold')
    ax.set_xlim(-1, 25)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['S2', 'S1\nInference', 'S1\nExecute'])
    ax.grid(True, alpha=0.3, axis='x')
    
    colors = {
        's2': '#E74C3C',
        's1_inf': '#3498DB',
        's1_exec': '#27AE60',
        'idle': '#BDC3C7'
    }
    
    for step in range(25):
        s2_triggered = (step == 0) or (step % gap == 0)
        
        # S2 bar
        if s2_triggered:
            ax.barh(3, 0.8, left=step, height=0.5, color=colors['s2'], edgecolor='black')
        else:
            ax.barh(3, 0.8, left=step, height=0.5, color=colors['idle'], edgecolor='black', alpha=0.3)
        
        # S1 Inference bar
        if s2_triggered:
            ax.barh(2, 0.8, left=step, height=0.5, color=colors['s1_inf'], edgecolor='black')
        else:
            ax.barh(2, 0.8, left=step, height=0.5, color=colors['idle'], edgecolor='black', alpha=0.3)
        
        # S1 Execution bar
        traj_len = min(gap, 4)
        if step % gap == 0 or step == 0:
            ax.barh(1, 0.8, left=step, height=0.5, color=colors['s1_exec'], edgecolor='black')
        else:
            ax.barh(1, 0.8, left=step, height=0.5, color=colors['s1_exec'], edgecolor='black', alpha=0.6)


# Row 3: Timing diagrams
for idx, (gap, mode) in enumerate([(1, 'SYNC'), (2, 'ASYNC gap=2'), (8, 'ASYNC gap=8')]):
    ax = fig1.add_subplot(gs1[2, idx])
    draw_timing_diagram(ax, gap, mode)

plt.savefig('fig1_system_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================================
# FIGURE 2: Trajectory Visualization with Different Gaps
# ============================================================================
fig2 = plt.figure(figsize=(18, 12))
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.25)

# Top row: Robot trajectories in workspace
ax1 = fig2.add_subplot(gs2[0, 0])
ax2 = fig2.add_subplot(gs2[0, 1])
ax3 = fig2.add_subplot(gs2[0, 2])

# Bottom row: Image space with pixel goals
ax4 = fig2.add_subplot(gs2[1, 0])
ax5 = fig2.add_subplot(gs2[1, 1])
ax6 = fig2.add_subplot(gs2[1, 2])

def draw_workspace_trajectory(ax, gap, title):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-2, 20)
    ax.set_ylim(-5, 8)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    np.random.seed(42)
    positions = [(0, 0, 0)]  # (x, y, theta)
    s2_update_positions = [(0, 0, 0)]
    
    current = positions[0]
    num_steps = 24
    
    for step in range(num_steps):
        s2_triggered = (step == 0) or (step % gap == 0)
        
        if s2_triggered:
            # S2 provides new goal direction
            goal_angle = np.random.uniform(-np.pi/3, np.pi/3)
            s2_update_positions.append(current)
            
            # S1 generates trajectory toward goal
            traj_len = min(gap, 4)
            for i in range(traj_len):
                dx = 0.5 * np.cos(goal_angle + np.random.uniform(-0.2, 0.2))
                dy = 0.5 * np.sin(goal_angle + np.random.uniform(-0.2, 0.2))
                dtheta = np.random.uniform(-0.1, 0.1)
                new_pos = (current[0] + dx, current[1] + dy, current[2] + dtheta)
                current = new_pos
                positions.append(current)
        else:
            # Continue along current direction
            dx = 0.5 * np.cos(current[2] + np.random.uniform(-0.1, 0.1))
            dy = 0.5 * np.sin(current[2] + np.random.uniform(-0.1, 0.1))
            new_pos = (current[0] + dx, current[1] + dy, current[2])
            current = new_pos
            positions.append(current)
    
    # Draw trajectory
    xs, ys, _ = zip(*positions)
    ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7, label='Robot path')
    
    # Draw direction vectors at S2 updates
    for i, (x, y, theta) in enumerate(s2_update_positions):
        if i > 0:
            ax.scatter(x, y, c='red', s=100, marker='*', zorder=5)
            # Draw direction arrow
            dx_arrow = 1.0 * np.cos(theta)
            dy_arrow = 1.0 * np.sin(theta)
            ax.arrow(x, y, dx_arrow, dy_arrow, head_width=0.2, head_length=0.1,
                    fc='red', ec='red', zorder=6)
    
    ax.scatter(0, 0, c='green', s=150, marker='o', edgecolor='black', 
              linewidth=2, label='Start', zorder=7)
    ax.scatter(xs[-1], ys[-1], c='orange', s=150, marker='s', edgecolor='black',
              linewidth=2, label='End', zorder=7)
    ax.legend(loc='upper left')
    
    return positions

pos1 = draw_workspace_trajectory(ax1, 1, 'SYNC (gap=1)\nS2 updates every step')
draw_workspace_trajectory(ax2, 2, 'ASYNC (gap=2)\nS2 updates every 2 steps')
draw_workspace_trajectory(ax3, 8, 'ASYNC (gap=8)\nS2 updates every 8 steps')

def draw_image_space(ax, gap, title, img_size=224):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-10, img_size + 10)
    ax.set_ylim(-10, img_size + 10)
    ax.set_xlabel('u (horizontal pixel)')
    ax.set_ylabel('v (vertical pixel)')
    ax.set_aspect('equal')
    
    # Draw image border
    rect = patches.Rectangle((0, 0), img_size, img_size, 
                             facecolor='#F8F9FA', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    np.random.seed(42)
    pixel_goals = []
    
    for step in range(0, 24, gap):
        u = np.random.uniform(50, img_size - 50)
        v = np.random.uniform(50, img_size - 50)
        pixel_goals.append((step, u, v))
        
        # Draw pixel goal
        circle = Circle((u, v), 15, facecolor='#E74C3C', edgecolor='black', 
                       linewidth=2, alpha=0.8)
        ax.add_patch(circle)
        ax.text(u, v, f'S2@{step}', ha='center', va='center', 
               fontsize=7, fontweight='bold', color='white')
        
        # Draw trajectory from previous goal
        if len(pixel_goals) > 1:
            prev_step, prev_u, prev_v = pixel_goals[-2]
            # Simulate trajectory points
            for i in range(1, min(gap, 4) + 1):
                t = i / (min(gap, 4) + 1)
                traj_u = prev_u + (u - prev_u) * t
                traj_v = prev_v + (v - prev_v) * t
                ax.scatter(traj_u, traj_v, c='#3498DB', s=20, alpha=0.5)
    
    # Draw current camera position marker
    ax.scatter(img_size/2, img_size/2, c='green', s=100, marker='^', 
               edgecolor='black', linewidth=2, label='Camera center')
    ax.legend(loc='upper right')

draw_image_space(ax4, 1, 'SYNC: Image Space (gap=1)')
draw_image_space(ax5, 2, 'ASYNC: Image Space (gap=2)')
draw_image_space(ax6, 8, 'ASYNC: Image Space (gap=8)')

fig2.suptitle('Fig 2: Trajectory Comparison Across Execution Modes\n'
             'Top: Workspace (X-Y) trajectories | Bottom: Image Space (u-v) pixel goals', 
             fontsize=14, fontweight='bold', y=1.02)

plt.savefig('fig2_trajectory_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================================
# FIGURE 3: Timing and State Evolution
# ============================================================================
fig3 = plt.figure(figsize=(18, 10))
gs3 = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.35, wspace=0.2)

# State machine visualization
ax1 = fig3.add_subplot(gs3[0, :])

def draw_state_machine(ax):
    ax.set_xlim(0, 24)
    ax.set_ylim(-1, 5)
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Component Activity', fontsize=11)
    
    np.random.seed(42)
    gap = 2  # Example gap
    
    components = [
        ('S2 Inference', '#E74C3C', [1]),
        ('S1 Inference', '#3498DB', [0.8]),
        ('S1 Execution', '#27AE60', [0.6]),
        ('Env Step', '#9B59B6', [0.4]),
    ]
    
    y_positions = [3.5, 2.5, 1.5, 0.5]
    
    for comp_idx, (name, color, alpha) in enumerate(components):
        y = y_positions[comp_idx]
        ax.text(-0.5, y, name, fontsize=10, fontweight='bold', ha='right')
        
        for step in range(24):
            s2_triggered = (step == 0) or (step % gap == 0)
            
            if comp_idx == 0:  # S2
                if s2_triggered:
                    ax.barh(y, 0.8, left=step, height=0.6, color=color, 
                           edgecolor='black', alpha=1.0)
                else:
                    ax.barh(y, 0.8, left=step, height=0.6, color='#BDC3C7', 
                           edgecolor='black', alpha=0.3)
                    
            elif comp_idx == 1:  # S1 Inference
                if s2_triggered:
                    ax.barh(y, 0.8, left=step, height=0.6, color=color, 
                           edgecolor='black', alpha=0.8)
                else:
                    ax.barh(y, 0.8, left=step, height=0.6, color='#BDC3C7', 
                           edgecolor='black', alpha=0.3)
                            
            elif comp_idx == 2:  # S1 Execution
                ax.barh(y, 0.8, left=step, height=0.6, color=color, 
                       edgecolor='black', alpha=0.9)
                if not s2_triggered:
                    ax.barh(y, 0.8, left=step, height=0.6, color=color, 
                           edgecolor='black', alpha=0.5)
                            
            elif comp_idx == 3:  # Env
                ax.barh(y, 0.8, left=step, height=0.6, color=color, 
                       edgecolor='black', alpha=0.9)
    
    # Add annotations for S2 triggers
    for step in range(0, 24, gap):
        ax.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.set_title('State Evolution Over Time (gap=2 example)\nRed dashed = S2 inference triggers', 
                fontsize=12, fontweight='bold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in components])

draw_state_machine(ax1)

# Gap comparison bars
ax2 = fig3.add_subplot(gs3[1, 0])
ax3 = fig3.add_subplot(gs3[1, 1])
ax4 = fig3.add_subplot(gs3[1, 2])

def draw_performance_comparison(ax, title):
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    metrics = ['Latency\n(ms/step)', 'S2 Update\nFrequency', 'Trajectory\nFreshness']
    gap1_vals = [80, 1.0, 1.0]  # Normalized
    gap2_vals = [40, 0.5, 0.7]
    gap8_vals = [15, 0.125, 0.3]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = ax.bar(x - width, gap1_vals, width, label='gap=1 (SYNC)', color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x, gap2_vals, width, label='gap=2 (ASYNC)', color='#F39C12', alpha=0.8)
    bars3 = ax.bar(x + width, gap8_vals, width, label='gap=8 (ASYNC)', color='#27AE60', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')

draw_performance_comparison(ax2, 'Performance Trade-offs')
draw_performance_comparison(ax3, 'Freshness Score')
draw_performance_comparison(ax4, 'Speed Score')

fig3.suptitle('Fig 3: Timing Analysis and Performance Comparison', 
             fontsize=14, fontweight='bold', y=1.02)

plt.savefig('fig3_timing_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*80)
print("GENERATED FIGURES:")
print("="*80)
print("1. fig1_system_architecture.png - Complete system architecture with data flow")
print("2. fig2_trajectory_comparison.png - Workspace and image space trajectories")
print("3. fig3_timing_analysis.png - State evolution and performance metrics")
print("="*80)
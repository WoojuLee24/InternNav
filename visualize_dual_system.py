import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('InternVLA-N1 Dual-System: Synchronous vs Asynchronous Execution', fontsize=16, fontweight='bold')

def draw_step_box(ax, x, y, color, label, alpha=1.0):
    r = FancyBboxPatch((x-0.3, y-0.3), 0.6, 0.6, boxstyle="round,pad=0.05", 
                       facecolor=color, edgecolor='black', linewidth=2, alpha=alpha)
    ax.add_patch(r)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

def draw_trajectory(ax, points, color, label, alpha=0.8):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=3, alpha=alpha, marker='o', markersize=8, label=label)
    ax.scatter(xs[-1], ys[-1], color=color, s=200, marker='*', edgecolor='black', linewidth=2, zorder=5)

def draw_system_timeline(ax, step_gap, title, sys2_color='#E74C3C', sys1_color='#3498DB'):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 32)
    ax.set_ylim(-1, 5)
    ax.set_xlabel('Step', fontsize=10)
    ax.set_yticks([])
    
    max_steps = 24
    
    sys2_points = []
    sys1_traj_points = []
    
    for step in range(max_steps):
        s2_triggered = (step == 0) or (step % step_gap == 0)
        
        if s2_triggered:
            sys2_points.append((step, 3.5))
            draw_step_box(ax, step, 3.5, sys2_color, 'S2', 0.9)
            
            traj_len = 4 if step_gap > 4 else step_gap
            base_x, base_y = step, 1
            
            if step_gap == 1:
                traj_points = [(base_x + i*0.3, base_y + np.sin(i*0.5)*0.3) for i in range(min(traj_len, 3))]
            elif step_gap == 2:
                traj_points = [(base_x + i*0.5, base_y + np.sin(i*0.7)*0.5) for i in range(min(traj_len, 4))]
            elif step_gap == 8:
                traj_points = [(base_x + i*0.7, base_y + np.sin(i*0.4)*0.8) for i in range(traj_len)]
            else:
                traj_points = [(base_x + i*0.6, base_y + np.sin(i*0.5)*0.6) for i in range(traj_len)]
            
            sys1_traj_points.append(traj_points)
            draw_trajectory(ax, traj_points, sys1_color, f'S1 traj (len={len(traj_points)})')
        else:
            draw_step_box(ax, step, 3.5, '#BDC3C7', f'{step}', 0.3)
    
    return sys2_points, sys1_traj_points

ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[0, 2]
ax4 = axes[1, 0]
ax5 = axes[1, 1]
ax6 = axes[1, 2]

sys2_pts_s, sys1_trajs_s = draw_system_timeline(ax1, 1, 'SYNC Mode (gap=1)\nS2 runs every step')
sys2_pts_a2, sys1_trajs_a2 = draw_system_timeline(ax2, 2, 'ASYNC Mode (gap=2)\nS2 runs every 2 steps')
sys2_pts_a8, sys1_trajs_a8 = draw_system_timeline(ax3, 8, 'ASYNC Mode (gap=8)\nS2 runs every 8 steps')

def draw_workspace(ax, title, step_gap):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 25)
    ax.set_ylim(-1, 10)
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('Position', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    current_pos = np.array([2.0, 2.0])
    positions = [current_pos.copy()]
    traj_start_steps = []
    
    max_steps = 20
    np.random.seed(42)
    
    for step in range(max_steps):
        s2_triggered = (step == 0) or (step % step_gap == 0)
        
        if s2_triggered:
            traj_start_steps.append(step)
            direction = np.random.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            traj_len = min(step_gap, 4) if step_gap < 8 else 4
            
            for i in range(traj_len):
                step_noise = np.random.randn(2) * 0.15
                delta = direction * 0.8 + step_noise
                current_pos = current_pos + delta
                positions.append(current_pos.copy())
        else:
            positions.append(current_pos.copy())
    
    xs, ys = zip(*positions)
    ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.5, label='robot path')
    ax.scatter(xs[0], ys[0], c='green', s=150, marker='o', edgecolor='black', linewidth=2, label='start', zorder=5)
    ax.scatter(xs[-1], ys[-1], c='red', s=150, marker='*', edgecolor='black', linewidth=2, label='current', zorder=5)
    
    for i, tsp in enumerate(traj_start_steps):
        ax.axvline(x=tsp, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(tsp, 9.5, f'S2 @t={tsp}', rotation=90, fontsize=8, color='red', va='top')
    
    ax.legend(loc='upper right')
    
    return positions

pos_sync = draw_workspace(ax4, 'Robot Trajectory - SYNC (gap=1)', 1)
pos_async2 = draw_workspace(ax5, 'Robot Trajectory - ASYNC gap=2', 2)
pos_async8 = draw_workspace(ax6, 'Robot Trajectory - ASYNC gap=8', 8)

legend_elements = [
    patches.Patch(facecolor='#E74C3C', edgecolor='black', label='S2 (System 2 - Planner)'),
    patches.Patch(facecolor='#3498DB', edgecolor='black', label='S1 (System 1 - Trajectory)'),
    patches.Patch(facecolor='#BDC3C7', edgecolor='black', alpha=0.3, label='No S2 inference'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.subplots_adjust(bottom=0.1, hspace=0.35, wspace=0.25)
plt.savefig('dual_system_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("="*60)
print("DUAL-SYSTEM EXECUTION MODES COMPARISON")
print("="*60)
print(f"""
SYNC Mode (gap=1):
  - S2 (Planner) runs EVERY step (1 step gap)
  - S1 (Trajectory) executes 1-3 steps per S2 output
  - Use when: precision critical, slow-moving environments
  - Latency: ~50-100ms per step

ASYNC Mode (gap=2):
  - S2 runs every 2 steps
  - S1 executes ~2-4 trajectory steps per S2 output
  - Use when: moderate speed requirements
  - Latency: ~25-50ms per step (improved)

ASYNC Mode (gap=8):
  - S2 runs every 8 steps
  - S1 executes 4 trajectory steps per S2 output
  - Use when: real-time requirements, fast response
  - Latency: ~10-20ms per step (fastest)
""")
print("="*60)
print("Generated: dual_system_timeline.png")
print("="*60)
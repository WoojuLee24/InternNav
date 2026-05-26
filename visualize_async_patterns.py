import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import warnings
warnings.filterwarnings('ignore')

fig = plt.figure(figsize=(18, 16))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

def draw_async_timeline(ax, title, gap_value, max_step=20):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1, max_step+1)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['Environment\nStep', 'S2 (Planner)\nInference', 'S1 (Trajectory)\nExecution'])
    ax.grid(True, alpha=0.3, axis='x')
    
    colors_s2 = '#E74C3C'
    colors_s1 = '#3498DB'
    colors_exec = '#27AE60'
    colors_wait = '#BDC3C7'
    
    for step in range(max_step):
        s2_triggered = (step == 0) or (step % gap_value == 0)
        
        ax.barh(0, 0.8, left=step, height=0.7, color=colors_exec, edgecolor='black', alpha=0.9)
        
        if s2_triggered:
            ax.barh(2, 0.8, left=step, height=0.7, color=colors_s2, edgecolor='black')
            ax.barh(1, 0.8, left=step, height=0.7, color=colors_s1, edgecolor='black', alpha=0.7)
            
            traj_len = min(gap_value, 4)
            for i in range(1, traj_len):
                if step + i < max_step:
                    pass
            
            for i in range(1, traj_len):
                if step + i < max_step:
                    ax.barh(0, 0.8, left=step+i, height=0.7, color=colors_exec, edgecolor='black', alpha=0.5)
        else:
            ax.barh(0, 0.8, left=step, height=0.7, color=colors_wait, edgecolor='black', alpha=0.3)

draw_async_timeline(ax1, f'SYNC Mode: gap=1 (S2 every step)', 1)
draw_async_timeline(ax2, f'ASYNC Mode: gap=2 (S2 every 2 steps)', 2)

def draw_trajectory_2d(ax, title, gap_value, num_trajectories=3):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 10)
    ax.set_xlabel('X Position', fontsize=10)
    ax.set_ylabel('Y Position', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    start_pos = (1, 1)
    ax.plot(*start_pos, 'go', markersize=15, markeredgecolor='black', label='Start')
    
    np.random.seed(42)
    for traj_idx in range(num_trajectories):
        current_pos = list(start_pos)
        positions = [current_pos.copy()]
        
        num_traj_steps = min(gap_value * 2, 8)
        
        for step in range(num_traj_steps):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(0.3, 0.8)
            dx = np.cos(angle) * dist
            dy = np.sin(angle) * dist
            
            new_pos = [current_pos[0] + dx, current_pos[1] + dy]
            current_pos = new_pos
            positions.append(current_pos.copy())
        
        xs, ys = zip(*positions)
        colors = ['#E74C3C', '#3498DB', '#27AE60']
        ax.plot(xs, ys, '-', linewidth=2, color=colors[traj_idx % len(colors)], alpha=0.7)
        ax.scatter(xs[-1], ys[-1], c=colors[traj_idx % len(colors)], s=100, marker='o', edgecolor='black')
        ax.annotate(f'Traj {traj_idx+1}', (xs[-1], ys[-1]), fontsize=9, 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.plot(10, 8, 'r*', markersize=20, markeredgecolor='black', label='Goal')
    ax.legend(loc='upper left')

draw_trajectory_2d(ax3, 'Trajectory Paths (gapdependent)', 1)

def create_system_diagram(ax, title):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    obs = FancyBboxPatch((0.5, 8), 2.5, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='#3498DB', edgecolor='black', linewidth=2)
    ax.add_patch(obs)
    ax.text(1.75, 8.75, 'OBSERVATION\n(rgb, depth)', ha='center', va='center', 
           fontsize=9, fontweight='bold', color='white')
    
    s2 = FancyBboxPatch((3.5, 8), 2.5, 1.5, boxstyle="round,pad=0.1", 
                       facecolor='#E74C3C', edgecolor='black', linewidth=2)
    ax.add_patch(s2)
    ax.text(4.75, 8.75, 'SYSTEM 2\n(Language Planner)', ha='center', va='center', 
           fontsize=9, fontweight='bold', color='white')
    
    pixel = FancyBboxPatch((6.5, 8), 2.5, 1.5, boxstyle="round,pad=0.1", 
                         facecolor='#F39C12', edgecolor='black', linewidth=2)
    ax.add_patch(pixel)
    ax.text(7.75, 8.75, 'PIXEL GOAL\n(u, v)', ha='center', va='center', 
           fontsize=9, fontweight='bold', color='white')
    
    s1 = FancyBboxPatch((0.5, 5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='#27AE60', edgecolor='black', linewidth=2)
    ax.add_patch(s1)
    ax.text(1.75, 5.75, 'SYSTEM 1\n(Trajectory Gen)', ha='center', va='center', 
           fontsize=9, fontweight='bold', color='white')
    
    traj = FancyBboxPatch((3.5, 5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='#9B59B6', edgecolor='black', linewidth=2)
    ax.add_patch(traj)
    ax.text(4.75, 5.75, 'TRAJECTORY\n[d x,dy,dθ]xN', ha='center', va='center', 
           fontsize=9, fontweight='bold', color='white')
    
    action = FancyBboxPatch((6.5, 5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                          facecolor='#1ABC9C', edgecolor='black', linewidth=2)
    ax.add_patch(action)
    ax.text(7.75, 5.75, 'ACTION\n(Execute)', ha='center', va='center', 
           fontsize=9, fontweight='bold', color='white')
    
    env = FancyBboxPatch((3.5, 1.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='#34495E', edgecolor='black', linewidth=2)
    ax.add_patch(env)
    ax.text(4.75, 2.25, 'ENVIRONMENT\n(Step + Reset)', ha='center', va='center', 
           fontsize=9, fontweight='bold', color='white')
    
    ax.annotate('', xy=(2.5, 8.75), xytext=(1.75, 8.75),
              arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(6, 8.75), xytext=(5.5, 8.75),
              arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(6.5, 6.75), xytext=(6.5, 7.5),
              arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(5.25, 6.75), xytext=(5.25, 7.5),
              arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(3.5, 5.75), xytext=(3, 5.75),
              arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(4.75, 3), xytext=(4.75, 1.5),
              arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(4.75, 7.5), xytext=(4.75, 8),
              arrowprops=dict(arrowstyle='->', color='black', lw=2))

create_system_diagram(ax4, 'Dual-System Architecture')

legend_elements = [
    patches.Patch(facecolor='#3498DB', edgecolor='black', label='Observation'),
    patches.Patch(facecolor='#E74C3C', edgecolor='black', label='S2 (Language Planner)'),
    patches.Patch(facecolor='#F39C12', edgecolor='black', label='Pixel Goal'),
    patches.Patch(facecolor='#27AE60', edgecolor='black', label='S1 (Trajectory Generator)'),
    patches.Patch(facecolor='#9B59B6', edgecolor='black', label='Trajectory'),
    patches.Patch(facecolor='#1ABC9C', edgecolor='black', label='Action'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10, 
          bbox_to_anchor=(0.5, 0.01))

fig.suptitle('InternVLA-N1 Dual-System: Async Execution Patterns\n(S1 = Visual Navigation Policy, S2 = Language-Grounded Planner)', 
            fontsize=14, fontweight='bold', y=0.98)

plt.savefig('dual_system_async_patterns.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*70)
print("ASYNC EXECUTION PATTERNS SUMMARY")
print("="*70)
print("""
┌────────────────────────────────────────────────────────────────────┐
│                   EXECUTION PATTERNS                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  SYNC Mode (gap=1)                                      │    │
│  │  ────────────────────────────────                         │    │
│  │  Step 0:        [S2]──>pixel──>[S1]──>traj──>exec     │    │
│  │  Step 1:        [S2]──>pixel──>[S1]──>traj──>exec     │    │
│  │  Step 2:        [S2]──>pixel──>[S1]──>traj──>exec     │    │
│  │                                                              │    │
│  │  - S2 runs EVERY frame (1 step gap)                        │    │
│  │  - Pixel goal always fresh                               │    │
│  │  - Max trajectory: 1-3 steps per S2                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ASYNC Mode (gap=2)                                      │    │
│  │  ────────────────────────────────────                     │    │
│  │  Step 0:        [S2]──>pixel──>[S1]──>traj──>exec     │    │
│  │  Step 1:                              exec              │    │
│  │  Step 2:        [S2]──>pixel──>[S1]──>traj──>exec     │    │
│  │  Step 3:                              exec              │    │
│  │                                                              │    │
│  │  - S2 runs every 2nd frame                              │    │
│  │  - Trajectory spans 2 steps                             │    │
│  │  - Max trajectory: 2-4 steps per S2                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ASYNC Mode (gap=8)                                      │    │
│  │  ────────────────────────────────────                     │    │
│  │  Step 0:        [S2]──>pixel──>[S1]──>traj──>exec     │    │
│  │  Step 1:                              exec              │    │
│  │  Step 2:                              exec              │    │
│  │  Step 3:                              exec              │    │
│  │  Step 4:                              exec              │    │
│  │  Step 5:                              exec              │    │
│  │  Step 6:                              exec              │    │
│  │  Step 7:                              exec              │    │
│  │  Step 8:        [S2]──>pixel──>[S1]──>traj──>exec     │    │
│  │                                                              │    │
│  │  - S2 runs every 8th frame (fastest response)              │    │
│  │  - Trajectory cached for 8 steps                        │    │
│  │  - Max trajectory: 4 steps per S2                    │    │
│  │  - Best for real-time robot control                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────────┘

Key Parameters:
  - gap: Number of environment steps between S2 inferences
  - sys2_max_forward_step: Maximum gap in async mode (default: 8)
  - Trajectory execution: Uses cached trajectory until next S2 update
""")
print("="*70)
print("Generated: dual_system_async_patterns.png")
print("="*70)
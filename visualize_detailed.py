import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow, FancyArrow
import warnings
warnings.filterwarnings('ignore')

fig = plt.figure(figsize=(20, 14))

gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[1, :2])
ax6 = fig.add_subplot(gs[1, 2:])

def create_pixel_goal_visualization(ax, title, gap_value):
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, 10))
    for i in range(10):
        for j in range(10):
            intensity = np.random.uniform(0.3, 1.0) if (i+j) % gap_value == 0 else np.random.uniform(0.1, 0.3)
            rect = patches.Rectangle((j, 9-i), 1, 1, facecolor=plt.cm.YlOrRd(intensity)[0:3], 
                                     edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
    
    pixel_goal = (3 + gap_value, 6)
    goal_circle = Circle(pixel_goal, 0.6, facecolor='#2ECC71', edgecolor='black', linewidth=3)
    ax.add_patch(goal_circle)
    ax.plot(*pixel_goal, 'w*', markersize=20, markeredgecolor='black')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f'gap={gap_value}', fontsize=10)

create_pixel_goal_visualization(ax1, 'Pixel Goal Heatmap', 1)
create_pixel_goal_visualization(ax2, 'Pixel Goal (gap=2)', 2)
create_pixel_goal_visualization(ax3, 'Pixel Goal (gap=8)', 8)

def draw_trajectory_comparison(ax, title, gap_value, num_steps=16):
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-1, num_steps+1)
    ax.set_ylim(-1, 8)
    ax.set_xlabel('Step', fontsize=10)
    ax.set_ylabel('Y Position', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    np.random.seed(42)
    positions_x = list(range(num_steps))
    positions_y = [2.0]
    
    for step in range(1, num_steps):
        s2_triggered = (step == 0) or (step % gap_value == 0)
        
        if s2_triggered:
            direction = np.random.choice([-1, 1]) * np.random.uniform(0.3, 0.8)
            new_y = positions_y[-1] + direction
            positions_y.append(new_y)
            
            start_y = new_y
            direction = np.random.choice([-1, 1]) * np.random.uniform(0.3, 0.6)
            for _ in range(min(gap_value, 4)):
                new_y = new_y + direction
                positions_y.append(new_y)
        else:
            direction = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3)
            new_y = positions_y[-1] + direction
            positions_y.append(new_y)
    
    positions_y = positions_y[:num_steps]
    
    ax.plot(positions_x, positions_y, 'b-', linewidth=2, alpha=0.7)
    ax.scatter(positions_x, positions_y, c='blue', s=50, zorder=5)
    
    for step in range(0, num_steps, gap_value):
        ax.scatter(step, positions_y[step], c='red', s=150, marker='*', edgecolor='black', linewidth=1.5, zorder=6)
        ax.axvline(x=step, color='red', linestyle='--', alpha=0.4)
    
    ax.scatter(0, positions_y[0], c='green', s=100, marker='o', edgecolor='black', label='Start')
    ax.scatter(num_steps-1, positions_y[-1], c='orange', s=100, marker='s', edgecolor='black', label='End')

draw_trajectory_comparison(ax5, 'Trajectory Comparison - Different Gaps', 1)

def draw_timeline(ax, title, gap_value):
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-1, 25)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Step', fontsize=10)
    ax.set_ylabel('Component', fontsize=10)
    ax.set_yticks([2, 5, 8])
    ax.set_yticklabels(['S1 Execute', 'S1 Infer', 'S2 Infer'])
    ax.grid(True, alpha=0.3, axis='x')
    
    for step in range(25):
        s2_triggered = (step == 0) or (step % gap_value == 0)
        
        if s2_triggered:
            ax.barh(8, 0.8, left=step, height=0.6, color='#E74C3C', edgecolor='black')
            ax.barh(5, 0.8, left=step, height=0.6, color='#3498DB', edgecolor='black', alpha=0.8)
            
            traj_len = min(gap_value, 4)
            for i in range(traj_len):
                if step + i < 25:
                    ax.barh(2, 0.8, left=step + i, height=0.6, color='#27AE60', edgecolor='black', alpha=0.7)
        else:
            ax.barh(2, 0.8, left=step, height=0.6, color='#27AE60', edgecolor='black', alpha=0.4)

draw_timeline(ax6, 'Execution Timeline - Multiple Step Gaps', 1)

legend_elements = [
    patches.Patch(facecolor='#E74C3C', edgecolor='black', label='S2 Inference'),
    patches.Patch(facecolor='#3498DB', edgecolor='black', label='S1 Inference'),
    patches.Patch(facecolor='#27AE60', edgecolor='black', alpha=0.7, label='S1 Execution'),
    patches.Patch(facecolor='#2ECC71', edgecolor='black', label='Pixel Goal'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, 
          bbox_to_anchor=(0.5, 0.01))

fig.suptitle('InternVLA-N1 DUAL-SYSTEM: Pixel Goals & Trajectories\n(S1 = Visual Navigation Policy, S2 = Language-Grounded Planner)', 
            fontsize=14, fontweight='bold', y=0.98)

plt.savefig('dual_system_detailed.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*70)
print("DETAILED ANALYSIS: PIXEL GOALS AND TRAJECTORIES")
print("="*70)
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    PIXEL GOAL OUTPUTS                                │
├─────────────────────────────────────────────────────────────────────┤
│ Pixel Goal = Target location in image coordinates (u, v)              │
│ - Used by S1 to generate trajectory toward the selected goal pixel      │
│ - Updated by S2 (System 2 - Language Planner) every 'gap' steps       │
│                                                                     │
│ gap=1 (Sync):   Pixel goal updated every step                      │
│ gap=2 (Async): Pixel goal updated every 2 steps                   │
│ gap=8 (Async): Pixel goal updated every 8 steps                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    TRAJECTORY OUTPUTS                                │
├─────────────────────────────────────────────────────────────────────┤
│ Trajectory = Sequence of (dx, dy, dtheta) action deltas             │
│ - Generated by S1 from pixel goal and current observation         │
│ - Each step in trajectory = movement toward goal                     │
│ - If gap=1:  S1 executes 1-2 steps per S2 update                  │
│ If gap=2:  S1 executes 2-4 steps per S2 update                  │
│ If gap=8:  S1 executes 4+ steps per S2 update                   │
└──────────────────���─��────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    EXECUTION MODES                                 │
├─────────────────────────────────────────────────────────────────────┤
│ SYNC (gap=1):                                                      │
│   - S2 infers EVERY step, S1 runs ~1-3 steps                    │
│   - Pros: Most accurate, always fresh pixel goal                 │
│   - Cons: Higher latency (~50-100ms per step)                  │
│                                                                     │
│ ASYNC (gap=2):                                                    │
│   - S2 infers every 2nd step, S1 runs ~2-4 steps                 │
│   - Pros: Balanced accuracy/speed                                │
│   - Cons: Slightly stale pixel goal (up to 2 steps)               │
│                                                                     │
│ ASYNC (gap=8):                                                    │
│   - S2 infers every 8th step, S1 runs max 4-step trajectory    │
│   - Pros: Fastest response (~10-20ms per step)                    │
│   - Cons: Uses older pixel goal for longer                         │
└─────────────────────────────────────────────────────────────────────┘
""")
print("="*70)
print("Generated: dual_system_detailed.png")
print("="*70)
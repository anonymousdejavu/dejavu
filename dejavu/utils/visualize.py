import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

def visualize_reuse_maps_from_inference(
        original_pixel_values,
        frame_idxs,
        output,
        show=False,
        save_path=None
    ):
    NUM_ROWS = 10

    num_maps = np.sum(m is not None for m in output.maps)
    target_indices = np.linspace(0, num_maps - 1, num=NUM_ROWS - 1, dtype=int)
    print(f"Out of {num_maps} reuse maps, we will visualize maps from layer {target_indices}")

    fig, axs = plt.subplots(NUM_ROWS, 4, figsize=(15, 30))

    # First row: original frames
    for i, frame in enumerate(original_pixel_values):
        axs[0, i].imshow(frame.cpu().numpy())
        axs[0, i].set_title(f'Frame {frame_idxs[i]}')
        axs[0, i].axis('off')

    # Rest: reuse maps
    for i, target_idx in enumerate(target_indices):
        ax = axs[i + 1]
        for j in range(4):
            reuse_map = output.maps[target_idx][j, 0][1:]
            patch_per_axis = int(math.sqrt(reuse_map.numel()))
            ax[j].imshow(
                reuse_map.reshape(patch_per_axis, patch_per_axis).cpu().numpy(),
                vmin=0,
                vmax=1
            )
            ax[j].set_title(f'Layer {target_idx}')
            ax[j].axis('off')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='jpeg', bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    
    plt.close(fig)
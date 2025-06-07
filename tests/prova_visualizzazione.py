import matplotlib.pyplot as plt
from truckscenes import TruckScenes


def render_fused_lidar(trucksc, sample_token: str, nsweeps: int = 5):
    """
    Render all LIDAR sensors for a sample into a single fused top-down plot.
    """
    sample = trucksc.get('sample', sample_token)

    lidar_data = {
        channel: token
        for channel, token in sample['data'].items()
        if trucksc.get('sample_data', token)['sensor_modality'] == 'lidar'
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for i, (_, sd_token) in enumerate(lidar_data.items()):
        trucksc.render_sample_data(
            sd_token,
            with_anns=(i == 0),   # Only show annotations once
            axes_limit=(84, 40),
            ax=ax,
            nsweeps=nsweeps,
            use_flat_vehicle_coordinates=True
        )

    ax.set_title('Fused LIDARs')
    plt.tight_layout()
    plt.show()



trucksc = TruckScenes('v1.0-mini', 'data\mini_dataset\man-truckscenes', True)
render_fused_lidar(trucksc, sample_token=trucksc.sample[10]['token'], nsweeps=5)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pyrow.spatial_operations as spatial_operations


def plot_cartopy_globe(
    show_image: bool = True,
    focus_coords: tuple[float] = (0, 0),
    extent: list[float] | None = None,
    figsize: tuple[float] = (10, 6),
    background_color: str | None = None,
    grid_color: str | None = None,
    land_color: str | None = None,
    ocean_color: str | None = None,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(
        1, 1, 1, projection=ccrs.Orthographic(focus_coords[0], focus_coords[1])
    )

    if show_image:
        img = mpimg.imread("images/world.topo.bathy.200412.3x5400x2700.jpg")
        img_extent = [-180, 180, -90, 90]
        ax.imshow(img, origin="upper", extent=img_extent, transform=ccrs.PlateCarree())

    if ocean_color is not None:
        ax.add_feature(cfeature.OCEAN, zorder=0, fc="grey")
    if land_color is not None:
        ax.add_feature(cfeature.LAND, zorder=0, fc="black")
    if grid_color is not None:
        ax.gridlines(color="k")
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    if background_color is not None:
        fig.patch.set_facecolor(background_color)

    return fig, ax


if __name__ == "__main__":
    import pyrow.config as config

    fig, ax = plot_cartopy_globe(
        show_image=True,
        focus_coords=(75, -10),
        extent=[20, 130, -40, +0],
        grid_color="r",
        background_color="k",
    )

    start_coords = config.START_COORDS
    target_coords = config.TARGET_COORDS["Mombassa"]

    for start in start_coords.values():
        lon, lat = spatial_operations.calculate_geodesic_line(start, target_coords)
        ax.plot(lon, lat, "r-", transform=ccrs.PlateCarree())
        ax.plot(start[0], start[1], "ro", transform=ccrs.PlateCarree())

    ax.plot(target_coords[0], target_coords[1], "ro", transform=ccrs.PlateCarree())

    plt.show()

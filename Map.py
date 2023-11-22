import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Create a Basemap instance for South America
m = Basemap(
    projection='merc',
    llcrnrlat=-56,
    urcrnrlat=15,
    llcrnrlon=-85,
    urcrnrlon=-35,
    resolution='i'
)

# Create the map
m.drawcountries()
m.drawcoastlines()
m.drawmapboundary(fill_color='aqua')
m.drawcountries(linewidth=0.5, linestyle='solid', color='gray')
m.drawcoastlines(linewidth=0.5, linestyle='solid', color='green')
m.drawparallels(range(-60, 21, 10), labels=[1, 0, 0, 0])
m.drawmeridians(range(-90, -34, 10), labels=[0, 0, 0, 1])

# Coordinates of Salta, Argentina
salta_lat, salta_lon = -24.7859, -65.4117

# Convert coordinates to map projection
x, y = m(salta_lon, salta_lat)
xlabel, ylabel = m(salta_lon, -29.0)

# Plot the point and label the city of Salta
m.plot(x, y, 'ro', markersize=2)
plt.text(xlabel, ylabel, ' Salta', fontsize=12, ha='center')

# Show the map
plt.title('South America Map with Salta')
plt.show()






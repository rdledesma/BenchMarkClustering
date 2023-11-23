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

# # Plot the point and label the city of Salta
# m.plot(x, y, 'ro', markersize=2)
# plt.text(xlabel, ylabel, ' Salta', fontsize=12, ha='center')

# Show the map
plt.title('South America Map with Salta')
plt.show()




import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Crear una instancia de Basemap para Sudamérica
mapa = Basemap(llcrnrlon=-90, llcrnrlat=-60, urcrnrlon=-30, urcrnrlat=20, resolution='l')

# Dibujar la costa de Sudamérica
mapa.drawcoastlines()

# Dibujar las fronteras de los países de Sudamérica
mapa.drawcountries()

# Obtener las coordenadas de la ciudad de Salta
latitud = -24.7859
longitud = -65.4117

# Convertir las coordenadas a las coordenadas del mapa
x, y = mapa(longitud, latitud)

# Dibujar un recuadro alrededor de la provincia de Salta
mapa.plot([x-1, x+1, x+1, x-1, x-1], [y-1, y-1, y+1, y+1, y-1], color='blue', linewidth=2)

# Agregar una etiqueta en la ciudad de Salta
#plt.text(x, y, 'Salta', fontsize=12, ha='center', va='center', color='red')

# Dibujar un cuadro de diálogo a la derecha del mapa
cuadro_dialogo_x = x + 20  # Ajustar la posición en el eje x
cuadro_dialogo_y = y - 20 # Mantener la posición en el eje y
cuadro_dialogo_width = 20
cuadro_dialogo_height = 20
mapa.plot([cuadro_dialogo_x - cuadro_dialogo_width/2, cuadro_dialogo_x + cuadro_dialogo_width/2, cuadro_dialogo_x + cuadro_dialogo_width/2, cuadro_dialogo_x - cuadro_dialogo_width/2, cuadro_dialogo_x - cuadro_dialogo_width/2], [cuadro_dialogo_y - cuadro_dialogo_height/2, cuadro_dialogo_y - cuadro_dialogo_height/2, cuadro_dialogo_y + cuadro_dialogo_height/2, cuadro_dialogo_y + cuadro_dialogo_height/2, cuadro_dialogo_y - cuadro_dialogo_height/2], color='black', linewidth=2)

# Mostrar el mapa
plt.show()






import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Crear el mapa base de Sudamérica
mapa_sudamerica = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=20, llcrnrlon=-100, urcrnrlon=-30, resolution='l')

# Obtener las coordenadas de Salta
latitud_salta = -24.7829
longitud_salta = -65.4232

# Dibujar el mapa de Sudamérica
mapa_sudamerica.drawcoastlines()
mapa_sudamerica.drawcountries()
mapa_sudamerica.fillcontinents(color='lightgray')

# Dibujar un rectángulo alrededor de Salta
x, y = mapa_sudamerica(longitud_salta, latitud_salta)
rect = plt.Rectangle((x - 1000000, y - 500000), 2000000, 1000000, edgecolor='red', facecolor='none')
plt.gca().add_patch(rect)

# Agregar etiqueta "Salta" en la ciudad
plt.text(x, y, 'Salta', ha='center', va='center', color='red')

# Crear el cuadro de diálogo a la derecha
cuadro_dialogo = plt.axes([0.7, 0.2, 0.2, 0.2], facecolor='white')

# Crear el mapa de la provincia de Salta
mapa_salta = Basemap(ax=cuadro_dialogo, projection='merc', llcrnrlat=-27, urcrnrlat=-20, llcrnrlon=-67, urcrnrlon=-62, resolution='l')

# Dibujar el mapa de Salta
mapa_salta.drawcoastlines()
mapa_salta.drawcountries()
mapa_salta.fillcontinents(color='lightgray')
mapa_salta.readshapefile('provincias/provincias', 'provincias', encoding='UTF-8') 

# Establecer propiedades
mapa_salta.provincias.set_color('gray')
mapa_salta.provincias.set_linewidth(1)

# Mostrar el gráfico
plt.show()
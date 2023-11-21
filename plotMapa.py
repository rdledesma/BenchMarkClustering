import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Coordenadas aproximadas de los límites de la provincia de Salta
salta_limits = {
    'min_lon': -70.9351,
    'max_lon': -60.4840,
    'min_lat': -26.6882,
    'max_lat': -22.9489
}

# Coordenadas de la ciudad de Salta
salta_city = {'lon': -65.4167, 'lat': -24.7833}

# Crear un gráfico
fig, ax = plt.subplots(figsize=(10, 10))

# Dibujar la provincia de Salta
salta_polygon = mpatches.Rectangle((salta_limits['min_lon'], salta_limits['min_lat']),
                                   salta_limits['max_lon'] - salta_limits['min_lon'],
                                   salta_limits['max_lat'] - salta_limits['min_lat'],
                                   edgecolor='black', facecolor='lightblue')
ax.add_patch(salta_polygon)

# Agregar un marcador para la ciudad de Salta
ax.plot(salta_city['lon'], salta_city['lat'], marker='o', color='red', markersize=8)
ax.text(salta_city['lon'] + 0.02, salta_city['lat'], 'Salta', fontsize=8, ha='left')

# Añadir leyenda
ax.legend(['Salta'], loc='upper right')

# Añadir etiquetas y título
plt.title("Mapa Político de la Provincia de Salta")
plt.xlabel("Longitud")
plt.ylabel("Latitud")

# Mostrar el mapa
plt.show()

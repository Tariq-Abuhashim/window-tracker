import pyproj

import pyproj

def gps_to_utm(lat, lon, alt):
    """Convert GPS coordinates to UTM."""
    utm_zone = int((lon + 180) / 6) + 1
    # Specify the correct hemisphere with "+south" if needed
    utm_projection = pyproj.Proj(f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +south")
    x, y = utm_projection(lon, lat)
    return x, y, alt

# Example for Sydney (Approx. -33.8688 latitude, 151.2093 longitude)
test_x, test_y, test_z = gps_to_utm(-33.8688, 151.2093, 0)  # Sea level for altitude
print(f"Converted UTM Coordinates: x={test_x}, y={test_y}, z={test_z}")

# Example for Perth (Approx. -31.9505 latitude, 115.8605 longitude)
test_x, test_y, test_z = gps_to_utm(-31.9505, 115.8605, 0)
print(f"Converted UTM Coordinates for Perth: x={test_x}, y={test_y}, z={test_z}")

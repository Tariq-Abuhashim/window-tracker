import numpy as np
import argparse
import pyproj
import scipy.spatial.transform     
import json

def geodetic2enu(lat, lon, alt, lat_org, lon_org, alt_org):
    transformer = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
    x, y, z = transformer.transform( lon,lat,  alt,radians=False)
    x_org, y_org, z_org = transformer.transform( lon_org,lat_org,  alt_org,radians=False)
    vec=np.array([[ x-x_org, y-y_org, z-z_org]]).T

    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()#angle*-1 : left handed *-1
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()#angle*-1 : left handed *-1

    rotMatrix = rot1.dot(rot3)    
   
    enu = rotMatrix.dot(vec).T.ravel()
    return enu.T

def enu2geodetic(x,y,z, lat_org, lon_org, alt_org):
    transformer1 = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
    transformer2 = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )
    
    x_org, y_org, z_org = transformer1.transform( lon_org,lat_org,  alt_org,radians=False)
    ecef_org=np.array([[x_org,y_org,z_org]]).T
    
    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()#angle*-1 : left handed *-1
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()#angle*-1 : left handed *-1

    rotMatrix = rot1.dot(rot3)

    ecefDelta = rotMatrix.T.dot( np.array([[x,y,z]]).T )
    ecef = ecefDelta+ecef_org
    lon, lat, alt = transformer2.transform( ecef[0,0],ecef[1,0],ecef[2,0],radians=False)

    return [lat,lon,alt]

def read_reference_coordinates(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        parts = first_line.split()
        lat_ref, lon_ref, alt_ref = float(parts[1]), float(parts[2]), float(parts[3])
    return lat_ref, lon_ref, alt_ref

def convert_normal_locations(json_file, reference_file):
    lat_ref, lon_ref, alt_ref = read_reference_coordinates(reference_file)
    with open(json_file, 'r') as file:
        data = json.load(file)

    for item in data:
        x, y, z = item['normal_location']
        lat, lon, alt = enu2geodetic(x, y, z, lat_ref, lon_ref, alt_ref)
        item['geo_coordinates'] = [lat, lon, alt]

    # Print or save the updated data
    #print(json.dumps(data, indent=4))

    # Write the updated data back to the same JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated data has been written back to {json_file}")

def main():
    parser = argparse.ArgumentParser(description="Converts windows location from UTM meters to GPS lat, lon, alt.")
    parser.add_argument("reference_file", help="Path to gps.txt file")
    parser.add_argument("windows_file", help="Path to the normals results file")

    args = parser.parse_args()

    convert_normal_locations(args.windows_file, args.reference_file)

if __name__ == "__main__":
    main()

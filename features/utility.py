import folium
import matplotlib.pyplot as plt
import numpy as np
import math

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180
Zoom = 25

def plot_location(city_location): #[[lat1, lng1]，[lat2, lng2],...,]
    m = folium.Map(location=city_location[0],
                   zoom_start=15,
                   tiles='openstreetmap')
    for cl in city_location:
        folium.Marker(cl,
                      popup='<i>Teste</i>', 
                      tooltip='Click me!').add_to(m)
    m.save('index.html')

def plot_trajs(tdf): #dataframe
    m = tdf.plot_trajectory()
    m.save('index.html')

def plot_offline_trajs(tdf, uid, condition): #uid = [uid1,uid2] condition=['2020-06','2020-07']
    color = {'06': "blue", '07':"red"}
    label = {'06': "June", '07':"July"}
    tmp = tdf[(tdf['uid']==uid[0])|(tdf['uid']==uid[1])]
    for c in condition:
        lat, lng = [], []
        for i in range(len(tmp)):
            if c in str(tmp.iloc[i,2]):
                print(tmp.iloc[i,0],tmp.iloc[i,1],tmp.iloc[i,2])
                lat.append(tmp.iloc[i,0])
                lng.append(tmp.iloc[i,1])
        #print(lat, lng)
        plt.plot(np.array(lat), np.array(lng), linewidth=0.5)

def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)

def map_size(levelOfDetail):
    return 256 << levelOfDetail

def latlon2pxy(latitude, longitude, levelOfDetail = Zoom):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY

def txy2quadkey(tileX, tileY, levelOfDetail = Zoom):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return ''.join(quadKey)

def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY

def latlon2quadkey(lat,lon, level = Zoom):
    pixelX, pixelY = latlon2pxy(lat, lon, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY, level)

#plot_location([[30.668535,104.135169], [30.694533, 104.108396]])    
plot_location([[30.731035,104.555504]])
print(latlon2pxy(30.731035,104.555504))
pixelX, pixelY = latlon2pxy(30.731035,104.555504)
print(pxy2txy(pixelX, pixelY))
print(type(latlon2quadkey(30.731035,104.555504)))       
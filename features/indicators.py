import skmob
from skmob.preprocessing import filtering
from skmob.preprocessing import compression
from skmob.preprocessing import detection
from skmob.measures.individual import jump_lengths, radius_of_gyration, k_radius_of_gyration
from skmob.measures.collective import origin_destination_matrix
from scipy.stats import entropy
from datetime import timedelta
from datetime import datetime
import pandas as pd
import math
import numpy as np
import pickle
import constants
import utility as F
from skmob.utils.gislib import getDistanceByHaversine
from math import sin, cos, sqrt, atan2, radians
from sklearn.neighbors import DistanceMetric
from pandas import Series, DataFrame
from rtree import index
import pandas as pd
import os
from tool import *

mins = 60.0
cellsize = 100

def usersum():
    filenames = os.listdir('Data')
    for f in filenames:
        traj_dict = []
        # traj_dict[f] = []
        pltnames = os.listdir('../Data/'+f+'/Trajectory')
        for p in pltnames:
            data = np.genfromtxt('../Data/'+f+'/Trajectory/'+p, delimiter=',',
                                 skip_header=6, dtype=[float,float,float,float,float,'S10','S8'])
            for item in data:
                traj_dict.append([item[0], item[1],
                                     datetime.strptime(str(item[5], encoding='utf8')+'-'
                                                       + str(item[6], encoding='utf8'),
                                                       '%Y-%m-%d-%H:%M:%S')])

        pickle.dump(traj_dict, open('../user/'+f, 'wb'), protocol=2)

class RtreeCase():
    def __init__(self):
        self.idx = index.Index()
    
    def insert(self, id, data, obj): #id = str, data = (lat, lon)
        self.idx.insert(id, data, obj=obj)

    def handle(self, width, num=1, objects=True):
        res=list(self.idx.nearest(width, num, objects=objects))
        return res

def haversine_distance(lat1, lon1, lat2, lon2):
    dist = DistanceMetric.get_metric('haversine')
    X = [[radians(lat1), radians(lon1)], [radians(lat2), radians(lon2)]]
    distance_sklearn = 6373.0 * dist.pairwise(X)
    return np.array(distance_sklearn).item(1) #km


def _home_location_individual(traj, start_night='22:00', end_night='07:00'):
    night_visits = traj.set_index(pd.DatetimeIndex(traj.datetime)).between_time(start_night, end_night)
    if len(night_visits) != 0:
        lat, lng = night_visits.groupby([constants.LATITUDE, constants.LONGITUDE]).count().sort_values(by=constants.DATETIME, ascending=False).iloc[0].name
    else:
        lat, lng = traj.groupby([constants.LATITUDE, constants.LONGITUDE]).count().sort_values(by=constants.DATETIME, ascending=False).iloc[0].name
    home_coords = (lat, lng)
    return home_coords


def home_location(traj, start_night='22:00', end_night='07:00', show_progress=True, col_name='home_loc'):
    # if 'uid' column in not present in the TrajDataFrame
    if constants.UID not in traj.columns:
        return pd.DataFrame([_home_location_individual(traj, start_night=start_night, end_night=end_night)], columns=[constants.LATITUDE, constants.LONGITUDE])
    
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _home_location_individual(x, start_night=start_night, end_night=end_night))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _home_location_individual(x, start_night=start_night, end_night=end_night))
    tmp = pd.DataFrame({'uid':df.index.to_list(), col_name:df.to_list()})
    tmp[[col_name+'_lat', col_name+'_lng']] = tmp[col_name].apply(pd.Series)
    return tmp.drop([col_name], axis=1)

def home_address(hl_df, home_add, col_name='home_add'):
    temp = []
    for i in range(len(hl_df)):
        if str(hl_df.iloc[i,0]).split('-')[0] in home_add.keys():
            temp.append(home_add[str(hl_df.iloc[i,0]).split('-')[0]])
        else:
            print('missing uid', str(hl_df.iloc[i,0]))
    hl_df.insert(hl_df.shape[1], col_name, temp)
    return hl_df

def To_traj(data_path):
    print('------TrajDataFrame-------')
    data_list = pickle.load(open('../user/'+data_path, 'rb'), encoding='bytes')
    tdf = skmob.TrajDataFrame(data_list, latitude=0, longitude=1, datetime=2)
    print(tdf.head())
    return tdf

def Noise_filtering(tdf):
    print('------noise_filtering-------')
    #filter out all points with a speed (in km/h) from the previous point higher than 500 km/h
    ftdf = filtering.filter(tdf, max_speed_kmh=500.)
    #print(ftdf.parameters)
    #print(ftdf.head())
    print('n_deleted_points', len(tdf) - len(ftdf)) # number of deleted points
    return ftdf

def Traj_compression(tdf):
    print('------traj_compression-------')
    #compress the trajectory using a spatial radius of 0.2 km
    ctdf = compression.compress(tdf, spatial_radius_km=0.1)
    #print the difference in points between original and filtered TrajDataFrame
    print('Points of the original trajectory:\t%s'%len(tdf))
    print('Points of the compressed trajectory:\t%s'%len(ctdf))
    return ctdf

def Stop_detection(tdf, minutes_for_a_stop=60.0):
    print('------stop_detection-------')
    #compute the stops for each individual in the TrajDataFrame
    stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=minutes_for_a_stop, spatial_radius_km=0.3, leaving_time=True)
    #print a portion of the detected stops
    print(stdf.shape)
    return stdf

def Radius_of_gyration(tdf):
    return radius_of_gyration(tdf)

def K_radius_of_gyration(tdf):
    return k_radius_of_gyration(tdf, k=2)

def Activity_locations(tdf):
    return tdf.groupby(['uid'],as_index=False)['uid'].agg({'cnt':'count'})

def Activity_entropy(tdf):
    tdf['leaving_datetime'] = pd.to_datetime(tdf['leaving_datetime'],format = '%Y-%m-%d %H:%M:%S')
    tdf['datetime'] = pd.to_datetime(tdf['datetime'],format = '%Y-%m-%d %H:%M:%S') 
    tdf['duration'] = tdf['leaving_datetime'].subtract(tdf['datetime'])
    print(tdf.head())
    prob = {}
    for i in range(len(tdf)): #'3' is uid and '-1' is 'duration'
        if tdf.iloc[i,3] not in prob:
            prob[tdf.iloc[i,3]] = []
            prob[tdf.iloc[i,3]].append(tdf.iloc[i,-1])
        else:
            prob[tdf.iloc[i,3]].append(tdf.iloc[i,-1])
    temp = []
    for key in prob:
        _sum= sum(prob[key], timedelta())
        probalities_list = [val/_sum for val in prob[key]]
        #print(sum(probalities_list))
        temp.append([key, entropy(probalities_list, base=2)]) #log2
    #print(temp)
    my_df = pd.DataFrame(temp)
    my_df.columns = ['uid','activity_entropy']
    return my_df

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def Travel_diversity(tdf):
    latlim, lonlim = [min(list(tdf['lat'])), max(list(tdf['lat']))], [min(list(tdf['lng'])), max(list(tdf['lng']))]
    delta = 1000
    prob = {}
    for i in range(len(tdf)-1): #'3' is uid
        lon1, lat1 = lonlat2meters(tdf.iloc[i,1], tdf.iloc[i,0])
        lon2, lat2 = lonlat2meters(tdf.iloc[i+1,1], tdf.iloc[i+1,0])
        lat1_grid = int((lat1-latlim[0])/delta)
        lon1_grid = int((lon1-lonlim[0])/delta)
        lat2_grid = int((lat2-latlim[0])/delta)
        lon2_grid = int((lon2-lonlim[0])/delta)
        if tdf.iloc[i,3] not in prob:
            prob[tdf.iloc[i,3]] = {}
            prob[tdf.iloc[i,3]][frozenset([str(lat1_grid)+'$'+str(lon1_grid), str(lat2_grid)+'$'+str(lon2_grid)])] = 1
        else:
            if tdf.iloc[i,3] != tdf.iloc[i+1,3]:
                continue
            if frozenset([str(lat1_grid)+'$'+str(lon1_grid), str(lat2_grid)+'$'+str(lon2_grid)]) in prob[tdf.iloc[i,3]]:
                prob[tdf.iloc[i,3]][frozenset([str(lat1_grid)+'$'+str(lon1_grid), str(lat2_grid)+'$'+str(lon2_grid)])] += 1
            else:
                prob[tdf.iloc[i,3]][frozenset([str(lat1_grid)+'$'+str(lon1_grid), str(lat2_grid)+'$'+str(lon2_grid)])] = 1
    temp = []
    for key in prob:
        _sum= sum(list(prob[key].values()))
        probalities_list = [val/_sum for val in prob[key].values()]
        #print(sum(probalities_list))
        temp.append([key, entropy(probalities_list, base=2)]) #log2
    #print(temp)
    my_df = pd.DataFrame(temp)
    my_df.columns = ['uid','travel_diversity']
    return my_df
            
def Unicity(tdf):
    topL = 2
    total = len(set(tdf['uid']))
    latlim, lonlim = [min(list(tdf['lat'])), max(list(tdf['lat']))], [min(list(tdf['lng'])), max(list(tdf['lng']))]
    delta = 1000
    prob = {}
    for i in range(len(tdf)): #'3' is uid
        lon1, lat1 = lonlat2meters(tdf.iloc[i,1], tdf.iloc[i,0])
        lat1_grid = int((lat1-latlim[0])/delta)
        lon1_grid = int((lon1-lonlim[0])/delta)
        if tdf.iloc[i,3] not in prob:
            prob[tdf.iloc[i,3]] = {}
            prob[tdf.iloc[i,3]][str(lat1_grid)+'$'+str(lon1_grid)] = tdf['duration'][i]
        else:
            if str(lat1_grid)+'$'+str(lon1_grid) in prob[tdf.iloc[i,3]]:
                prob[tdf.iloc[i,3]][str(lat1_grid)+'$'+str(lon1_grid)] += tdf['duration'][i]
            else:
                prob[tdf.iloc[i,3]][str(lat1_grid)+'$'+str(lon1_grid)] = tdf['duration'][i]
    temp = {}
    for key in prob:
        d = prob[key]
        _sort = sorted(d.items(), key = lambda d:d[1], reverse=True)
        #print('_sort', _sort)
        temp[key] = set(np.array(_sort)[0:topL,0]) 
    #print(temp)
    Temp = []
    for key1 in temp:
        cnt = 0
        for key2 in temp:
            if temp[key1] != temp[key2]:
                cnt += 1
        Temp.append([key1, cnt/total])
    #print(Temp)
    my_df = pd.DataFrame(Temp)
    my_df.columns = ['uid','unicity']
    return my_df

def output(rg_df, krg_df, act_loc, act_ent, tra_div, uni):
    tmp = pd.merge(rg_df, krg_df, on='uid')
    tmp = pd.merge(tmp, act_loc, on='uid')
    tmp = pd.merge(tmp, act_ent, on='uid')
    tmp = pd.merge(tmp, tra_div, on='uid')
    tmp = pd.merge(tmp, uni, on='uid')
    return tmp

def Filter_1(tdf):
    user_list = set(tdf['uid'])
    tmp = []
    c = 0
    for u in user_list:
        if c % 100 ==0:
            print('process ', c, len(user_list))
        c = c + 1
        if len(tdf[tdf['uid']==u]) < 100:
            tmp.append(u)
    tdf = tdf[-tdf['uid'].isin(tmp)]
    print('after ', len(set(tdf['uid'])))
    return tdf

def Filter_2(tdf):
    print('before ', len(tdf['uid'].unique()))
    L = len(tdf)
    tmp = []
    uid_c = 1
    c = 0    
    for i in range(1, L):
        if c % 100000 ==0:
            print('process ', c, L)
        c =  c + 1
        if tdf.loc[i, 'uid'] == tdf.loc[i-1, 'uid']:
            uid_c += 1
        else:
            if uid_c < 100:
                tmp.append(tdf.loc[i-1, 'uid'])
            uid_c = 1
    tdf = tdf[-tdf['uid'].isin(tmp)]
    print('after ', len(tdf['uid'].unique()))
    return tdf

def _radius_of_ctr(traj):
    lats_lngs = traj[['lat', 'lng']].values
    center_of_mass = np.mean(lats_lngs, axis=0)
    return center_of_mass

def get_dist(data, tdf, col_name ='home_loc'):
    df = tdf.groupby('uid').progress_apply(lambda x: _radius_of_ctr(x))
    df = pd.DataFrame(df).reset_index().rename(columns={0: 'ctr'}) 
    temp = []
    for i in range(len(data)):
        lat, lng = data.loc[i, col_name+'_lat'], data.loc[i, col_name+'_lng']
        if i > 0 and data.loc[i, 'uid'].split('-')[0] == data.loc[i-1, 'uid'].split('-')[0]:
            lat, lng = data.loc[i-1, col_name+'_lat'], data.loc[i-1, col_name+'_lng']
        temp.append(getDistanceByHaversine((lat, lng), df.loc[i,'ctr']))
    return temp
    
def get_general_feature():
    print('mobility indicators')
    # data_path = '../'+city+'_data'
    home_lat = []
    home_lng = []
    company_lat = []
    company_lng = []
    features = pd.DataFrame()
    for i in range(182):
        if i<10:
            data_path = '00'+str(i)
        elif i<100:
            data_path = '0'+str(i)
        else:
            data_path = str(i)
        tdf = To_traj(data_path) #read the trajectory data (GeoLife, Beijing, China)
        uids = [i for j in range(len(tdf))]
        tdf['uid'] = uids
        #tdf = Filter_2(tdf)
        #F.plot_offline_trajs(tdf, uid=[1,5], condition=['2008-10-24', '2008-10-25','2008-10-26','2008-10-27','2008-10-28'])
        ftdf = Noise_filtering(tdf)
        ctdf = Traj_compression(ftdf)
        stdf = Stop_detection(ctdf)
        if len(stdf) <= 1:
            continue

        hl_df = home_location(tdf)
        home_lat.append(hl_df.iloc[0][1])
        home_lng.append(hl_df.iloc[0][2])

        cl_df = home_location(tdf, start_night='09:00', end_night='17:00', col_name='company_loc')
        company_lat.append(cl_df.iloc[0][1])
        company_lng.append(cl_df.iloc[0][2])

        #get mobility indicators -> users*(uid, indicator_value)
        rg_df = Radius_of_gyration(tdf) #LMI
        # rg_df['uid'] = [i]

        pickle.dump(stdf, open('../user/stdf/stdf'+data_path, 'wb'), protocol=2)
        # stdf = pd.concat([stdf,Stop_detection(ctdf)])
        # pickle.dump(stdf, open(path+'_stdf', 'wb'), protocol=2)

        krg_df = K_radius_of_gyration(stdf) #HMI-K_radius_of_gyration
        # krg_df['uid'] = [i]
        act_loc = Activity_locations(stdf) #HMI-Activity_locations
        # act_loc['uid'] = [i]
        act_ent = Activity_entropy(stdf) #HMI-Activity_entropy
        # act_ent['uid'] = [i]
        tra_div = Travel_diversity(stdf) #HMI-Travel_diversity
        # tra_div['uid'] = [i]
        uni = Unicity(stdf) #HMI-Unicity
        # uni['uid'] = [i]
        features_t = output(rg_df, krg_df, act_loc, act_ent, tra_div, uni)
        print(features_t)
        features = pd.concat([features, features_t])
    #
    # print('features', features.shape)
    pickle.dump(features, open('../user/features', 'wb'), protocol=2)
    features['home_lat'] = home_lat
    features['home_lng'] = home_lng
    features['company_lat'] = company_lat
    features['company_lng'] = company_lng
    pickle.dump(features, open('../user/features', 'wb'), protocol=2)
    # pickle.dump(hl_df, open(path+'_home_feature', 'wb'), protocol=2)
    # pickle.dump(cl_df, open(path+'_company_feature', 'wb'), protocol=2)
    
def get_week_feature():
    print('mobility indicators for week')
    features = pd.DataFrame()
    for i in range(182):
        # features = pd.DataFrame()
        if i < 10:
            data_path = '00' + str(i)
        elif i < 100:
            data_path = '0' + str(i)
        else:
            data_path = str(i)
        tdf = To_traj(data_path)  # read the trajectory data (GeoLife, Beijing, China)
        uids = [data_path for j in range(len(tdf))]
        tdf['uid'] = uids
        week = []
        for index, row in tdf.iterrows():
            w = row['datetime'].isocalendar()
            week.append(str(w[0]) + str(w[1]))
        weekset = list(set(week))
        
        os.makedirs('../user/weeks'+str(mins)[:2]+'/' + data_path)
        tdf['week'] = week
        for j in weekset:
            home_lat = []
            home_lng = []
            company_lat = []
            company_lng = []
            t = tdf[tdf['week']==j]
            # tdf = Filter_2(tdf)
            # F.plot_offline_trajs(tdf, uid=[1,5], condition=['2008-10-24', '2008-10-25','2008-10-26','2008-10-27','2008-10-28'])
            ftdf = Noise_filtering(t)
            ctdf = Traj_compression(ftdf)
            if len(ctdf)<=0:
                continue
            stdf = Stop_detection(ctdf, minutes_for_a_stop=mins)
            if len(stdf) <= 1:
                continue

            hl_df = home_location(t)
            home_lat.append(hl_df.iloc[0][1])
            home_lng.append(hl_df.iloc[0][2])

            cl_df = home_location(t, start_night='09:00', end_night='17:00', col_name='company_loc')
            company_lat.append(cl_df.iloc[0][1])
            company_lng.append(cl_df.iloc[0][2])

            # get mobility indicators -> users*(uid, indicator_value)
            rg_df = Radius_of_gyration(t)  # LMI
            # rg_df['uid'] = [i]
            path_t = '../user/weeks'+str(mins)[:2]+ '/' + data_path+'/'+str(weekset.index(j)+1)+'/'
            os.makedirs(path_t)
            pickle.dump(stdf, open(path_t+'stdf', 'wb'), protocol=2)
            # stdf = pd.concat([stdf,Stop_detection(ctdf)])
            # pickle.dump(stdf, open(path+'_stdf', 'wb'), protocol=2)

            krg_df = K_radius_of_gyration(stdf)  # HMI-K_radius_of_gyration
            # krg_df['uid'] = [i]
            act_loc = Activity_locations(stdf)  # HMI-Activity_locations
            # act_loc['uid'] = [i]
            act_ent = Activity_entropy(stdf)  # HMI-Activity_entropy
            # act_ent['uid'] = [i]
            tra_div = Travel_diversity(stdf)  # HMI-Travel_diversity
            # tra_div['uid'] = [i]
            uni = Unicity(stdf)  # HMI-Unicity
            # uni['uid'] = [i]
            features_t = output(rg_df, krg_df, act_loc, act_ent, tra_div, uni)
            features_t.iloc[0,0] = features_t.iloc[0,0]+'-'+str(weekset.index(j)+1)
            print(features_t)
            # print('features', features.shape)
            features_t['home_lat'] = home_lat
            features_t['home_lng'] = home_lng
            features_t['company_lat'] = company_lat
            features_t['company_lng'] = company_lng
            pickle.dump(features_t, open(path_t + 'features', 'wb'), protocol=2)
            features = pd.concat([features, features_t])
    pickle.dump(features, open('../user/weeks'+str(mins)[:2]+'/'+'features', 'wb'), protocol=2)


def sum_feature():
    feature = pd.DataFrame()
    userdirs = os.listdir('../user/weeks'+str(mins)[:2])
    for u in userdirs:
        if os.path.isdir('../user/weeks'+str(mins)[:2]+'/'+u):
            weekdirs = os.listdir('../user/weeks'+str(mins)[:2]+'/'+u)
            for w in weekdirs:
                f = pickle.load(open('../user/weeks'+str(mins)[:2]+'/'+u+'/'+w+'/features', 'rb'), encoding='bytes')
                feature = pd.concat([feature, f])
    pickle.dump(feature, open('../user/features2', 'wb'), protocol=2)

def sum_stdf():
    userdirs = os.listdir('../user/weeks'+str(mins)[:2]+'/')
    stdf_all = pd.DataFrame()
    for u in userdirs:
        if os.path.isdir('../user/weeks'+str(mins)[:2]+'/' + u):
            weekdirs = os.listdir('../user/weeks'+str(mins)[:2]+'/' + u)
            for w in weekdirs:
                stdf = pickle.load(open('../user/weeks'+str(mins)[:2]+'/' + u + '/' + w + '/stdf', 'rb'), encoding='bytes')
                t = u+'-'+w
                uids = [t for i in range(len(stdf))]
                stdf['uid'] = uids
                pickle.dump(stdf, open('../user/weeks'+str(mins)[:2]+'/' + u + '/' + w + '/stdf', 'wb'), protocol=2)
                stdf_all = pd.concat([stdf_all, stdf])
    pickle.dump(stdf_all, open('../user/weeks'+str(mins)[:2]+'/'+'stdf_all', 'wb'), protocol=2)

def filter_by_home():
    features = pickle.load(open('../user/weeks'+str(mins)[:2]+'/'+'features', 'rb'), encoding='bytes')
    features_in_bj = features[(features['home_lat']>=39.4)&(features['home_lat']<=41.1)|
                        (features['home_lng']>=115.4)&(features['home_lng']<=117.5)]
    # uids = features_in_bj['uid']
    print(len(features_in_bj))
    print(len(features))
    pickle.dump(features_in_bj, open('../user/weeks'+str(mins)[:2]+'/'+'features_in_bj', 'wb'), protocol=2)
    # print(uids)

def get_loc_dict(cellsize=100):
    loc_set = set()
    stdf_all = pickle.load(open('../user/weeks'+str(mins)[:2]+'/'+'stdf_all', 'rb'), encoding='bytes')
    features_in_bj = pickle.load(open('../user/weeks'+str(mins)[:2]+'/'+'features_in_bj', 'rb'), encoding='bytes')
    uidlist = list(features_in_bj['uid'])
    stdf_all = stdf_all[stdf_all['uid'].isin(uidlist)]
    pickle.dump(stdf_all, open('../user/weeks'+str(mins)[:2]+'/'+'stdf_all', 'wb'), protocol=2)
    for index, row in stdf_all.iterrows():
        location = get_loc(row['lng'], row['lat'], cellsize)
        loc_set.add(location)
    index = [i for i in range(1, len(loc_set)+1)]
    loc_dict = dict(zip(loc_set, index))
    print(len(loc_dict))
    #pickle.dump(loc_dict, open('../user/weeks'+str(mins)[:2]+'/'+'loc_dict'+str(cellsize)+'m', 'wb'), protocol=2)



# get_general_feature()
usersum()
get_week_feature()
# sum_feature()
sum_stdf()
filter_by_home()
get_loc_dict(100)

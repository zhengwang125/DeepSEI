import numpy as np
from datetime import datetime
import os
import pickle
from tool import *
import requests, json
from multiprocessing.dummy import Pool as ThreadPool
from rtree.index import Index

def usersum():
    filenames = os.listdir('Data')
    for f in filenames:
        traj_dict = []
        # traj_dict[f] = []
        pltnames = os.listdir('Data/'+f+'/Trajectory')
        for p in pltnames:
            data = np.genfromtxt('Data/'+f+'/Trajectory/'+p, delimiter=',',
                                 skip_header=6, dtype=[float,float,float,float,float,'S10','S8'])
            for item in data:
                traj_dict.append([item[0], item[1],
                                     datetime.strptime(str(item[5], encoding='utf8')+'-'
                                                       + str(item[6], encoding='utf8'),
                                                       '%Y-%m-%d-%H:%M:%S')])

        pickle.dump(traj_dict, open('user/'+f, 'wb'), protocol=2)


def get_time_bin(time):
    if time.weekday()<5:
        return time.hour
    return time.hour+24

# poi_categories = ['work','home','美食','酒店','购物','生活服务','丽人','旅游景点','休闲娱乐','运动健身','教育培训',
#                   '公司企业','房地产','文化传媒','医疗','汽车服务']
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'}
def get_Activity(lng, lat, home_lng, home_lat, company_lng, company_lat):
    if is_company(lng, lat, company_lng, company_lat):
        return 0
    if is_home(lng, lat, home_lng, home_lat):
        return 1
    poi_list = poi_categories[2:]
    count = [0 for i in range(len(poi_list))]
    i = 0
    lng, lat = wgs84_to_bd09(lng, lat)
    for p in poi_list:
        response = requests.get(
            'https://api.map.baidu.com/place/v2/search?query='+
            p+
            '&radius=100&location='+str(lat)+','+str(lng)+'&output=json&ak=bQzehITh1lNGwQWGyynbxn6gV3mz2wdB',
            headers=headers)
        state = json.loads(response.text)
        count[i] = len(state['results'])
        i += 1
    if max(count)==0:
        return len(poi_categories)
    return np.argmax(count)+2

global home_lng, home_lat, company_lng, company_lat
#

maps = []
for i in range(9):
    maps.append(pickle.load(open('./Maps/map_'+str(i), 'rb'), encoding='bytes'))
def get_map_Activity(lng, lat):
    if is_company(lng, lat, company_lng, company_lat):
        return 1
    if is_home(lng, lat, home_lng, home_lat):
        return 2
    cellsize = 100
    loc = get_loc(lng, lat, cellsize)
    neighbours = get_neighbour(loc, cellsize)
    act_list = np.zeros(9)
    for n in neighbours:
        for m_l in range(len(maps)):
            act_list[m_l] += len(maps[m_l][maps[m_l]['loc_'+str(cellsize)]==n])
    return act_list.argmax()+3

def get_map_Activity_wrap(args):
    return get_map_Activity(*args)

def generate_sequence(cellsize=100):
    features_in_bj = pickle.load(open('./user/weeks/features_in_bj', 'rb'), encoding='bytes')
    loc_dict = pickle.load(open('./user/loc_dict'+str(cellsize)+'m', 'rb'), encoding='bytes')
    uidlist = list(features_in_bj['uid'])
    sequencelist = []
    for uid in uidlist:
        u = uid[:3]
        w = uid[4:]
        stdf = pickle.load(open('./user/weeks/' + u + '/' + w + '/stdf', 'rb'), encoding='bytes')
        wday = stdf.iloc[0]['datetime'].weekday()
        d = 0
        week = [[] for i in range(7)]
        tem_f = features_in_bj[features_in_bj.uid == uid]
        home_lng = tem_f['home_lng'][0]
        home_lat = tem_f['home_lat'][0]
        company_lng = tem_f['company_lng'][0]
        company_lat = tem_f['company_lat'][0]
        for index, row in stdf.iterrows():
            if wday != row['datetime'].weekday():
                d += 1
                wday = row['datetime'].weekday()
            location = get_loc(row['lng'], row['lat'], cellsize)
            location = loc_dict[location]

            # activity = get_Activity(row['lng'], row['lat'], home_lng, home_lat, company_lng, company_lat)
            t = [uid, location, get_time_bin(row['datetime']), row['lng'], row['lat'], row['activity_'+str(cellsize)]]
            print(t)
            week[d].append(t)
        sequencelist.append(week)
    pickle.dump(sequencelist, open('./user/sequencelist', 'wb'), protocol=2)

def append_activity(cellsize=100):
    features_in_bj = pickle.load(open('./user/weeks/features_in_bj', 'rb'), encoding='bytes')
    # loc_dict = pickle.load(open('./user/loc_dict'+str(cellsize)+'m', 'rb'), encoding='bytes')
    uidlist = list(features_in_bj['uid'])
    sequencelist = []
    for uid in uidlist:
        u = uid[:3]
        w = uid[4:]
        stdf = pickle.load(open('./user/weeks/' + u + '/' + w + '/stdf', 'rb'), encoding='bytes')
        wday = stdf.iloc[0]['datetime'].weekday()
        d = 0
        week = [[] for i in range(7)]
        tem_f = features_in_bj[features_in_bj.uid == uid]
        global home_lng, home_lat, company_lng, company_lat
        home_lng = tem_f['home_lng'][0]
        home_lat = tem_f['home_lat'][0]
        company_lng = tem_f['company_lng'][0]
        company_lat = tem_f['company_lat'][0]
        lngs = list(stdf['lng'])
        lats = list(stdf['lat'])
        z = [(i,j) for (i,j) in zip(lngs, lats)]
        pool = ThreadPool(20)
        activities = list(pool.map(get_map_Activity_wrap, z))
        stdf['activity_'+str(cellsize)] = activities
        print(uid)
        print(activities)
        pool.close()
        pool.join()
        pickle.dump(stdf, open('./user/weeks/' + u + '/' + w + '/stdf', 'wb'), protocol=2)

def get_lstm_input(cellsize=100):
    loc_input = []
    time_input = []
    activity_input = []
    sequencelist = pickle.load(open('./user/sequencelist', 'rb'), encoding='bytes')
    for week in sequencelist:
        loc_week = []
        time_week = []
        activity_week = []
        for day in week:
            loc_day = []
            time_day = []
            activity_day = []
            for day_t in  day:
                loc_day.append(day_t[1])
                time_day.append(day_t[2]+1)
                activity_day.append(day_t[5])
            loc_week.append(loc_day)
            time_week.append(time_day)
            activity_week.append(activity_day)
        loc_input.append(loc_week)
        time_input.append(time_week)
        activity_input.append(activity_week)
    pickle.dump(loc_input, open('./user/inputs/loc_input_'+str(cellsize), 'wb'), protocol=2)
    pickle.dump(time_input, open('./user/inputs/time_input', 'wb'), protocol=2)
    pickle.dump(activity_input, open('./user/inputs/activity_input', 'wb'), protocol=2)
    print(loc_input)
    print(time_input)
    print(activity_input)

def get_bp_input(cellsize = 100):
    features_in_bj = pickle.load(open('./user/weeks/features_in_bj', 'rb'), encoding='bytes')

    # com = []
    # home = []
    # for index, row in features_in_bj.iterrows():
    #     com.append(get_loc(row['company_lng'], row['company_lat'], cellsize))
    #     home.append(get_loc(row['home_lng'], row['home_lat'], cellsize))
    activity_entropy = list(features_in_bj['activity_entropy'])
    travel_diversity = list(features_in_bj['travel_diversity'])
    radius_of_gyration = list(features_in_bj['radius_of_gyration'])
    pickle.dump(activity_entropy, open('./user/inputs/activity_entropy', 'wb'), protocol=2)
    pickle.dump(travel_diversity, open('./user/inputs/travel_diversity', 'wb'), protocol=2)
    pickle.dump(radius_of_gyration, open('./user/inputs/radius_of_gyration', 'wb'), protocol=2)
    # features_in_bj['com_'+str(cellsize)] = com
    # features_in_bj['home_' + str(cellsize)] = home
    # com_set = set(com)
    # home_set = set(home)
    # index_com = [i for i in range(1, len(com_set) + 1)]
    # index_home = [i for i in range(1, len(home_set) + 1)]
    # com_dict = dict(zip(com_set, index_com))
    # home_dict = dict(zip(home_set, index_home))
    # com2 = []
    # home2 = []
    # for c in com:
    #     com2.append(com_dict[c])
    # for h in home:
    #     home2.append(home_dict[h])
    # print(com2)
    # print(home2)
    # pickle.dump(features_in_bj, open('./user/weeks/features_in_bj', 'wb'), protocol=2)
    # pickle.dump(com2, open('./user/inputs/com_'+str(cellsize), 'wb'), protocol=2)
    # pickle.dump(home2, open('./user/inputs/home_' + str(cellsize), 'wb'), protocol=2)

def get_label():
    features_in_bj = pickle.load(open('./user/weeks/features_in_bj', 'rb'), encoding='bytes')
    voronois_bj = pickle.load(open('voronois_bj', 'rb'), encoding='bytes')
    landuse_idx = Index()
    for i in range(len(voronois_bj['geometry'])):
        (xmin, ymin, xmax, ymax) = voronois_bj.loc[i, 'geometry'].bounds
        landuse_idx.insert(i, (xmin, ymin, xmax, ymax))
    label = []
    for index, row in features_in_bj.iterrows():
        i = 0
        l = list(landuse_idx.nearest((row['home_lng'], row['home_lat']), 50))
        while np.isnan(voronois_bj['price'][l[i]]):
            i += 1
        label.append(voronois_bj['price'][l[i]])

    features_in_bj['label'] = label
    pickle.dump(features_in_bj, open('./user/weeks/features_in_bj', 'wb'), protocol=2)
    pickle.dump(label, open('./user/inputs/labels', 'wb'), protocol=2)

def get_label_general():
    features_in_bj = pickle.load(open('./user/weeks/features_in_bj', 'rb'), encoding='bytes')
    features_general = pickle.load(open('./user/features', 'rb'), encoding='bytes')
    voronois_bj = pickle.load(open('voronois_bj', 'rb'), encoding='bytes')
    landuse_idx = Index()
    for i in range(len(voronois_bj['geometry'])):
        (xmin, ymin, xmax, ymax) = voronois_bj.loc[i, 'geometry'].bounds
        landuse_idx.insert(i, (xmin, ymin, xmax, ymax))
    label = []
    for index, row in features_in_bj.iterrows():
        i = 0
        home_lng_t = features_general[features_general['uid']==int(row['uid'][:3])]['home_lng'].values[0]
        home_lat_t = features_general[features_general['uid'] == int(row['uid'][:3])]['home_lat'].values[0]
        print(home_lng_t, home_lat_t)
        l = list(landuse_idx.nearest((home_lng_t, home_lat_t), 5))
        while np.isnan(voronois_bj['price'][l[i]]):
            i += 1
        label.append(voronois_bj['price'][l[i]])

    features_in_bj['label'] = label
    pickle.dump(features_in_bj, open('./user/weeks/features_in_bj', 'wb'), protocol=2)
    pickle.dump(label, open('./user/inputs/labels', 'wb'), protocol=2)


append_activity()
generate_sequence()
# usersum()
get_lstm_input()
get_bp_input()
# get_label_general()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.neighbors import LocalOutlierFactor


def LOF(data, predict, k):
    
    clf = LocalOutlierFactor(n_neighbors=k+1, algorithm='auto', contamination=0.1,n_jobs=-1)
    clf.fit(data)
    predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
    predict['local outlier factor'] = -clf._decision_function(predict.iloc[:,:-1])
    
    return predict


def plot_lof(result, method):
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 4)).add_subplot(111)
    plt.scatter(result[result['local outlier factor'] > method].index,
                result[result['local outlier factor'] > method]['local outlier factor'], c='red', s=20,
                marker='.', alpha=None,
                label='Outliers')
    plt.scatter(result[result['local outlier factor'] <= method].index,
                result[result['local outlier factor'] <= method]['local outlier factor'], c='black', s=20,
                marker='.', alpha=None, label='Inliers')
    plt.hlines(method, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF detection', fontsize=13)
    plt.ylabel('Factor', fontsize=15)
    plt.legend()
    plt.show()


def lof(data, predict=None, k=5, method=1, plot=False):
    if predict == None:
        predict = data.copy()

    predict = pd.DataFrame(predict)
    predict = LOF(data, predict, k)
    if plot == True:
        plot_lof(predict,method)

    outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <= method].sort_values(by='local outlier factor')
    return outliers, inliers


#tool functions
def cal_duration(st, ed):
    s = st[11:]
    e = ed[11:]
    s_t = int(s[6:]) + int(s[3:5]) * 60 + int(s[0:2]) * 3600
    e_t = int(e[6:]) + int(e[3:5]) * 60 + int(e[0:2]) * 3600
    return e_t - s_t

#tool functions
def timestamp_to_cnt(ts):
    ts = ts[11:]
    return int(ts[6:]) + int(ts[3:5]) * 60 + int(ts[0:2]) * 3600


#--------------------------------  preprocess  -----------------------------#
f = open('data//sensor_info.csv')
f2 = open('data//event.csv')
f3 = open('data//traffic_data_01_29_2020.csv')
f4 = open('data//location_match.csv')
r_lines = csv.reader(f, delimiter=',')
r_lines2 = csv.reader(f2, delimiter=',')
r_lines3 = csv.reader(f3, delimiter=',')
r_lines4 = csv.reader(f4, delimiter=',')

#get intersection of sensor_id in "sensor_info.csv" and "traffic_data_01_29_2020.csv"
list_sensorinfo = []
i = 0
for row in r_lines:
    if i == 0:
        i += 1
        continue
    list_sensorinfo.append(row[0].rstrip())
list_traffic = []
i = 0
for row in r_lines3:
    if i == 0:
        i += 1
        continue
    list_traffic.append(row[0])
list_intersection=[new for new in list_sensorinfo if new in list_traffic]
f3.close()

# build a dict for location_match (sensor ——> event)
Sen_Eve_Dict = dict()
sensor_list = []
i = 0
for row in r_lines4:
    if i == 0:
        i += 1
        continue
    Sen_Eve_Dict[row[0].rstrip()] = int(row[1].lstrip())

#convert data in event.csv to list
event_id = []
event_st = [] #start_time of each event
event_et = [] #end_time of each_event
for row in r_lines2:
    if row[0]=='event_id':
        continue
    event_id.append(int(row[0]))
    event_st.append(timestamp_to_cnt(row[1]))
    event_et.append(timestamp_to_cnt(row[2]))

#--------------------------------  preprocess ends -----------------------------#


def core_func(sensor_id):
    Timestamp = []
    Speed = []
    f3 = open('data//traffic_data_01_29_2020.csv')
    r_lines3 = csv.reader(f3, delimiter=',')
    for row in r_lines3:
        if row[0] == sensor_id:
            Timestamp.append(row[3])
            Speed.append(row[6])
    if len(Timestamp) == 0:
        return []

    Speed = list(map(eval, Speed))

    # transform timepoint to duration(compared to start_time)
    st = Timestamp[len(Timestamp) - 1]  # start_time in all timestamps
    Duration = []
    for i in range(len(Timestamp)):
        Duration.append(cal_duration(st, Timestamp[i]))

    # drop the isolated time data (like '02:15:01','01:30:01')
    filter_end = 0
    filter_dura = []
    for i in range(len(Duration)):
        filter_dura.append(Duration[i])
        if i + 1 < len(Duration) and Duration[i] - Duration[i + 1] > 10000:
            filter_end = i
            break
    filter_speed = Speed[0:filter_end + 1]

    dat = list(zip(filter_dura, filter_speed))
    dat = np.array(dat)
    if len(dat) > 5:
        outliers, inliers = lof(dat, k=5, method=2, plot=False)

        # pick out outliers according to k distances
        time_list = inliers[0].values.tolist()
        k_dis_list = inliers['k distances'].values.tolist()

        out_st = 0
        for i in range(len(k_dis_list)):
            if i + 1 < len(k_dis_list) and k_dis_list[i + 1] - k_dis_list[i] > 500:
                out_st = i + 1
                break

        #out_k_dis = k_dis_list[out_st:]
        out_time = time_list[out_st:] # outliers for sensor_id
        for item in out_time:
            item += timestamp_to_cnt(st)
            item = int(item)

        print(out_time)

        ans = []
        for ot in out_time:
            for i in range(25):  #25 events
                if ot>=event_st[i] and ot<=event_et[i]:  #time match
                    if sensor_id in Sen_Eve_Dict and Sen_Eve_Dict[sensor_id] == event_id[i]: #location match
                        ans.append(event_id[i])

        return ans

        #print(inliers)
        #plt.scatter(np.array(dat)[:, 0], np.array(dat)[:, 1], s=10, c='b', alpha=0.5)
        #plt.scatter(outliers[0], outliers[1], s=10 + outliers['local outlier factor'] * 100, c='r', alpha=0.2)

for s in list_intersection:
    print(core_func(s))
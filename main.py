from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.mllib.tree import DecisionTree


def create_spark_context():
    global sc, path
    sc = SparkContext(conf=SparkConf().setAppName('RunDecisionTreeBinary'))
    path = "hdfs://node1:8020/input/"


def read_data():
    global lines
    raw_data_with_header = sc.textFile(path + "hour.csv")
    header = raw_data_with_header.first()
    raw_data = raw_data_with_header.filter(lambda x: x != header)
    lines = raw_data.map(lambda x: x.split(','))
    print(f"The lines first row is: {lines.first()}")


def convert_float(x):
    result = 0 if x == "?" else float(x)
    return result


def extract_features(record, feature_end):
    feature_season = [convert_float(i) for i in record[2]]
    features = [convert_float(i) for i in record[4: feature_end-2]]
    result = np.concatenate((feature_season, features))
    return result


def extract_label(field):
    label = field[-1]
    result = float(label)-1
    return result


def prepare_data():
    global label_point_RDD
    print("Before standard:")
    label_RDD = lines.map(lambda x: extract_label(x))
    feature_RDD = lines.map(lambda r: extract_features(r, len(r)-1))
    print("After standard:")
    label_point = label_RDD.zip(feature_RDD)
    label_point_RDD = label_point.map(lambda x: LabeledPoint(x[0], x[1]))
    result = label_point_RDD.randomSplit([8, 1, 1])
    return result


def evaluate_model(model, validation_data):
    score = model.predict(validation_data.map(lambda x: x.features))
    score_and_labels = score.zip(validation_data.map(lambda x: x.label)).map(lambda x: (float(x[0]), float(x[1])))
    metrics = RegressionMetrics(score_and_labels)
    accuracy = metrics.rootMeanSquaredError
    return accuracy


def train_evaluation_model(train_data,
                           validation_data,
                           impurity,
                           max_depth,
                           max_bins):
    start_time = time()
    model = DecisionTree.trainRegressor(train_data,
                                         categoricalFeaturesInfo={},
                                         impurity=impurity,
                                         maxDepth=max_depth,
                                         maxBins=max_bins
                                         )
    accuracy = evaluate_model(model, validation_data)
    duration = time() - start_time
    return accuracy, impurity, max_depth, max_bins, duration, model


def eval_parameter(train_data, validation_data):
    impurity_list = ["variance"]
    max_depth_list = [3, 5, 10, 15, 20, 25]
    max_bins_list = [3, 5, 10, 50, 100, 200]
    my_metrics = [
        train_evaluation_model(train_data,
                               validation_data,
                               impurity,
                               max_depth,
                               max_bins)
        for impurity in impurity_list
        for max_depth in max_depth_list
        for max_bins in max_bins_list
    ]
    s_metrics = sorted(my_metrics, key=lambda x: x[0], reverse=True)
    best_parameter = s_metrics[0]
    print(best_parameter)
    print(f"the best max_depth is:{best_parameter[2]}\n"
          f"the best max_bins is:{best_parameter[3]}\n"
          f"the best RMSE is:{best_parameter[0]}\n")
    best_RMSE = best_parameter[0]
    best_model = best_parameter[5]
    return best_RMSE, best_model


def predict_data(best_model):
    SeasonDict = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
    HoildayDict = {0: "NoHoliday", 1: "Holiday"}
    WeekDict = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    WorkDayDict = {1: "Workday", 0: "NoWorkday"}
    WeatherDict = {1: "Sunny", 2: "Cloudy", 3: "Rain", 4: "Downpour"}
    for lp in label_point_RDD.take(20):
        predict = int(best_model.predict(lp.features))
        label = lp.label
        features = lp.features
        error = abs(label - predict)
        data_desc = "  Factors: " + SeasonDict[features[0]] + ',' + \
                   "Month:" + str(features[1]) + ',' + \
                   str(features[2]) + "Hour," + \
                   HoildayDict[features[3]] + "," + \
                   "WeekDay:" + WeekDict[features[4]] + "," + \
                   WorkDayDict[features[5]] + "," + \
                   WeatherDict[features[6]] + "," + \
                   str(features[7] * 41) + " Celsius," + \
                   "FeelsLike:" + format(features[8] * 50, '.2f') + " Celsius," + \
                   "Humidity:" + format(features[9] * 100, '.1f') + "," + \
                   "Wind:" + format(features[10] * 67, '.2f') + \
                   " ==> Prediction:" + str(predict) + \
                   "  , Real:" + str(label) + ",  Gap:" + str(error)
        print(data_desc)


def show_chart(df, eval_parm, bar_parm, line_parm, y_min=0.5, y_max=1.0):
    ax = df[bar_parm].plot(kind='bar', title=eval_parm, figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(eval_parm, fontsize=12)
    ax.set_ylim([y_min, y_max])
    ax.set_ylabel(bar_parm, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[line_parm].values, linestyle='-', marker='o', linewidth=2, color='r')
    plt.show()


def draw_graph(train_data, validation_data, draw_type):
    impurity_list = ["variance"]
    max_depth_list = [3, 5, 10, 15, 20, 25]
    max_bins_list = [3, 5, 10, 50, 100, 200]
    if draw_type == "maxDepth":
        my_index = max_depth_list
        impurity_list = impurity_list
        max_bins_list = [5]
    elif draw_type == "maxBins":
        my_index = max_bins_list
        impurity_list = impurity_list
        max_depth_list = [3]
    my_metrics = [
        train_evaluation_model(train_data,
                               validation_data,
                               impurity,
                               max_depth,
                               max_bins)
        for impurity in impurity_list
        for max_depth in max_depth_list
        for max_bins in max_bins_list
    ]
    df = pd.DataFrame(my_metrics,
                      index=my_index,
                      columns=['RMSE', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    show_chart(df, draw_type, 'RMSE', 'duration', 0.5, 200)


if __name__ == "__main__":
    s_time = time()
    create_spark_context()
    print("Reading data stage".center(60, "="))
    read_data()
    train_d, validation_d, test_d = prepare_data()
    print(train_d.first())
    train_d.persist()
    validation_d.persist()
    test_d.persist()
    # print("Draw".center(60, "="))
    # draw_graph(train_d, validation_d, "maxDepth")
    # draw_graph(train_d, validation_d, "maxBins")
    print("Evaluate parameter".center(60, "="))
    b_RMSE, b_model = eval_parameter(train_d, validation_d)
    print(f"The best RMSE is: {b_RMSE}")
    print("Test".center(60, "="))
    test_data_auc = evaluate_model(b_model, test_d)
    print(f"best auc is:{format(b_RMSE, '.4f')}, test_data_auc is: {format(test_data_auc, '.4f')}, "
          f"they are only slightly different:{format(abs(float(b_RMSE) - float(test_data_auc)), '.4f')}")
    print("Predict".center(60, "="))
    predict_data(b_model)

    train_d.unpersist()
    validation_d.unpersist()
    test_d.unpersist()










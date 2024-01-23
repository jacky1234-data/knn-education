import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import os
import base64
import io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#Initialize the flask App
app = Flask(__name__)

#Plot are always running in the backend
plt.switch_backend('Agg')

#Some global data and the dataset
k_values = [i for i in range(3, 13)]

data = pd.read_csv(datapoints.csv") #Use your own path

#Calculate distance
def cal_distance(pointA, pointB):
    return ((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2) ** 0.5

#Find k neighbours
def find_closest_points(point, df, k):
    nrow = df.shape[0]
    closest_points = []
    for i in range(nrow):
        data_point = [df.loc[i, "x"], df.loc[i, "y"]]
        group = df.loc[i, "group"]
        dist = cal_distance(point, data_point)
        if len(closest_points) < k:
            closest_points.append([data_point[0], data_point[1], dist, group])
        else:
            max_dist = closest_points[0][2]
            max_index = 0
            for j in range(len(closest_points)):
                if j == 0:
                    continue  
                new_dist = closest_points[j][2]
                if new_dist > max_dist:
                    max_dist = new_dist
                    max_index = j
            if dist < max_dist:
                closest_points.pop(max_index)
                closest_points.append([data_point[0], data_point[1], dist, group])
    return closest_points

#Find the group
def identify_group(closest_points):
    groups = [i[3] for i in closest_points]
    group_count = {}
    for each in groups:
        if each not in group_count:
            group_count[each] = 1
        else:
            group_count[each] = group_count[each] + 1
    group_count = [[y, x] for x, y in group_count.items()]
    group_count.sort(key = lambda x:x[0], reverse = True)
    return group_count[0][1]

def plot_original(df):
    colors = ["red","purple","blue"]
    groups = ["A", "B", "C"]
    sns.scatterplot(data = df , x = "x", y = "y", hue = "group", 
                palette = {g:c for g, c in zip(groups, colors)}).set(title='Datapoints')
    plt.legend(bbox_to_anchor=(1.12, 1),loc='upper right',fontsize="8")

    # Convert the plot to SVG format
    svg_buffer = io.BytesIO()
    plt.savefig(svg_buffer, format='svg')
    svg_buffer.seek(0)
    svg_code = base64.b64encode(svg_buffer.read()).decode('utf-8')

    # Clear the Matplotlib figure to avoid overlapping plots
    plt.clf()
    return svg_code

def plot_neighbour(point, df, k):
    colors = ["red","purple","blue"]
    groups = ["A", "B", "C"]
    neighbours = find_closest_points(point, df, k)
    sns.scatterplot(data = df , x = "x", y = "y", hue = "group", 
                palette = {g:c for g, c in zip(groups, colors)}).set(title='Datapoints')
    for neigh in neighbours:
        sns.lineplot(x=[point[0], neigh[0]], y=[point[1], neigh[1]], 
                     color = colors[groups.index(neigh[3])], marker=False, linestyle='--')   
    plt.legend(bbox_to_anchor=(1.12, 1),loc='upper right',fontsize="8")

    # Convert the plot to SVG format
    svg_buffer = io.BytesIO()
    plt.savefig(svg_buffer, format='svg')
    svg_buffer.seek(0)
    svg_code = base64.b64encode(svg_buffer.read()).decode('utf-8')

    # Clear the Matplotlib figure to avoid overlapping plots
    plt.clf()
    return svg_code

#Split training and testing set
def find_train_data(df, percent_num):
    percentage = percent_num / 100
    nrow = df.shape[0]
    position = int(percentage * nrow)
    train_data = df.iloc[:position,:]
    train_data = train_data.reset_index(drop = True)
    return train_data

def find_test_data(df, percent_num):
    percentage = percent_num / 100
    nrow = df.shape[0]
    position = int(percentage * nrow)
    test_data = df.iloc[position:,:]
    test_data = test_data.reset_index(drop = True)
    return test_data

#Predict training and testing set
def predict_training_set(train_df, k):
    train_df["predicted group"] = "n/a"
    for row in range(train_df.shape[0]):
        point = [train_df.loc[row, "x"], train_df.loc[row, "y"]]
        close_points = find_closest_points(point, train_df, k)
        train_df.loc[row, "predicted group"] = identify_group(close_points)
    return train_df

def predict_testing_set(test_df, train_df, k):
    test_df["predicted group"] = "n/a"
    for row in range(test_df.shape[0]):
        point = [test_df.loc[row, "x"], test_df.loc[row, "y"]]
        close_points = find_closest_points(point, train_df, k)
        test_df.loc[row, "predicted group"] = identify_group(close_points)
    return test_df

#Plot confusion matrix
def plot_confusion(df, train_test):
    actual = df["group"]
    predict = df["predicted group"]
    cm = confusion_matrix(actual, predict, labels=["A","B","C"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=["A","B","C"])
    title = "for Training Set" if train_test == "train" else "for Testing Set" if train_test=="test" else ""
    disp.plot()
    plt.title('Confusion Matrix ' + title)

    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image as base64
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Clear the Matplotlib figure to avoid overlapping plots
    plt.clf()
    return plot_base64

#Plot lassification report
def make_classification_report(df):
    report = classification_report(df["group"], df["predicted group"], target_names=["A","B","C"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df["precision"] = round(report_df["precision"], 3)
    report_df["recall"] = round(report_df["recall"], 3)
    report_df["f1-score"] = round(report_df["f1-score"], 3)
    report_df["support"]['accuracy'] = report_df["support"]['macro avg']
    report_df["support"] = report_df["support"].astype("int64")
    report_html = report_df.to_html(classes='table table-striped', justify='center')
    return report_html


@app.route('/')
def home():
    graph = plot_original(data)
    return render_template('index.html', k_val = 3, k_opt = k_values, plot_graph = graph, hide_image = True)


#Running models
@app.route('/predict', methods=['POST'])
def predict():
    kselected = int(request.form.get("kvalue"))
    x_cord = int(request.form.get("x_coord_val"))
    y_cord = int(request.form.get("y_coord_val"))
    train_percent = int(request.form.get("percentage_val"))
    point = [x_cord, y_cord]
    point_group = identify_group(find_closest_points(point, data, kselected))
    pred_text = "The predicted group of location ({}, {}) is {}.".format(x_cord, y_cord, point_group)
    graph = plot_neighbour(point, data, kselected)

    train_data = find_train_data(data, train_percent)
    test_data = find_test_data(data, train_percent)
    train_data = predict_training_set(train_data, kselected)
    test_data = predict_testing_set(test_data, train_data, kselected)
    cm_train = plot_confusion(train_data, "train")
    cm_test = plot_confusion(test_data, "test")
    train_report = make_classification_report(train_data)
    test_report = make_classification_report(test_data)

    return render_template('index.html', k_val = kselected, k_opt = k_values, x_coord_val = x_cord, y_coord_val = y_cord,
                           percentage_val = train_percent, pred = pred_text, plot_graph = graph, confuse_matrix_train = cm_train,
                           confuse_matrix_test = cm_test, train_report_html = train_report, test_report_html = test_report)

#Clear every output
@app.route('/', methods=['POST'])
def clear():
    graph = plot_original(data)
    return render_template('index.html', k_val = 3, k_opt = k_values, plot_graph = graph, hide_image = True)

if __name__ == "__main__":
    app.run(debug=True)

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class read_dataset():
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

    def load_dataset(self):
        data = pd.read_csv(self.path + self.filename)
        return data

class config_attributes():
    def list_attributes(self):
        atts = ["G","MP","FG","FGA","FG%","3P","3PA","3P%","2P","2PA","2P%","FT","FTA","FT%","ORB","DRB","AST","STL","BLK","TOV","PTS"]
        return atts
    
    def aggregation_map(self):
        agg = {"G":"sum","MP":"sum","FG":"mean","FGA":"mean","FG%":"mean","3P":"mean","3PA":"mean","3P%":"mean",
                "2P":"mean","2PA":"mean","2P%":"mean","FT":"mean","FTA":"mean","FT%":"mean",
                "ORB":"mean","DRB":"mean","AST":"mean","STL":"mean","BLK":"mean","TOV":"mean","PTS":"mean"}
        return agg

class plot():
    def __init__(self, path, list_att, df):
        self.path = path
        self.atts = list_att
        self.df = df[list_att].fillna(0)

    def wcss_plot(self):
        wcss = []

        max_n = len(self.atts) - 1
        for i in range(1, max_n):
            kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
            kmeans.fit(self.df.to_numpy().tolist())
            wcss.append(kmeans.inertia_)
        
        plt.plot(range(1, max_n), wcss)
        plt.title("The elbow method")
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        plt.savefig(self.path + "wcss.png")

class clustering():
    def __init__(self, list_att, df, n_clusters):
        self.atts = list_att
        self.df = df[atts].fillna(0)
        self.n = n_clusters

    def cluster_kmeans(self):
        kmeans = KMeans(n_clusters = self.n, random_state = 0)
        model = kmeans.fit(self.df.to_numpy().tolist())
        pred = model.labels_
        df["Cluster"] = pred
        return df

if __name__ == "__main__":
    source_path = "sources/"
    output_path = "output/"
    filename = "datasets_78041_177907_nba201718.csv"

    data = read_dataset(source_path, filename).load_dataset()

    # aggregate the raw data by Player
    agg_data = data.groupby("Player", as_index=False).agg(config_attributes().aggregation_map())

    # remove any players with total minutes played less than 1000 minutes
    df = agg_data[agg_data["MP"] > 1000].copy()

    atts = [k for k,v in config_attributes().aggregation_map().items()]

    # Within Cluster Sum Square
    plot(output_path, atts, df).wcss_plot()

    n_clusters = 3

    df_cluster = clustering(atts, df, n_clusters)
    df_cluster = df_cluster.cluster_kmeans()

    df_cluster.to_csv(output_path + "nba_cluster.csv", index=False)
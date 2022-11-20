# %%
import timm
import torch
import torch.nn.functional as F

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedGroupKFold
import networkx as nx

# %%
FOLDER = 'data/'
TEST1_FOLDER = 'data/public/videos/'
TEST2_FOLDER = 'data/public_test_2/videos/'

# %%
train_embeddings = np.load('data/train_embeddings_frame0.npy')
public_embeddings = np.load('data/public_embeddings_frame0.npy')
public_test_2_embeddings = np.load('data/public_test_2_embeddings_frame0.npy')

# %%
embeddings = np.concatenate([train_embeddings, public_embeddings, public_test_2_embeddings])

# %%
df_train = pd.read_csv('data/train/label.csv')
df_train['set'] = 'train'

public_df = pd.DataFrame(os.listdir(TEST1_FOLDER))
public_df.columns = ['fname']
public_df['set'] = 'public'

public_test_2_df = pd.DataFrame(os.listdir(TEST2_FOLDER))
public_test_2_df.columns = ['fname']
public_test_2_df['set'] = 'public_test_2'

# %%
df = pd.concat([df_train, public_df, public_test_2_df]).reset_index(drop=True)

# %%
df

# %%
len(df), embeddings.shape

# %%
# get classes of neighrest neighbor in index set
knn = NearestNeighbors(n_neighbors=100, metric='cosine')
knn.fit(embeddings)
all_distances, all_indices = knn.kneighbors(embeddings)
all_distances = all_distances[:,1:]
all_indices = all_indices[:,1:]

# %%
index_name_map = df['fname'].to_dict()
index_label_map = df['liveness_score'].to_dict()
index_set_map = df['set'].to_dict()
name_label_map = df.set_index('fname')['liveness_score'].to_dict()

# %%
predictions_by_knn = []
min_distances = []
for dist, inds in tqdm(zip(all_distances, all_indices), total=len(df)):
    pred = [(i, d, index_label_map[i]) for d, i in zip(dist, inds)]
    min_distances.append(np.min(dist))
    predictions_by_knn.append(pred)
    # break

# %%
df['knn_queries'] = predictions_by_knn
df['min_dist'] = min_distances

# %%
df = df.reset_index()

# %%
# 1. Remove duplicates (videos almost exactly the same)

# %%
THRESH = 0.05

# %%
def cluster_similar(data_fr, threshold):
    G = nx.Graph()
    existing_nodes = data_fr['index'].tolist()
    for i, row in tqdm(data_fr.iterrows()):
        query = row['index']
        keys = row['knn_queries']
        G.add_nodes_from([query])
        for (key, dist, _) in keys:
            if key not in existing_nodes:
                continue
            if dist <= threshold:
                G.add_edges_from([(query, key)])

    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    return S
        

# %%
duplicated_clusters = cluster_similar(df, THRESH)

# %%
len(duplicated_clusters)

# %%
for i, s in enumerate(duplicated_clusters):
    nodes = s.nodes
    for node in nodes:
        df.loc[node, 'dup_id'] = i
df['dup_id'] = df['dup_id'].astype(int)

# %%
df

# %%
df_dedup = pd.concat([
    df[df.set=='train'].drop_duplicates(subset=['dup_id', 'liveness_score']),
    df[df.set!='train']
])
# df_dedup = df.drop_duplicates(subset=['dup_id', 'liveness_score'])

# %%
len(duplicated_clusters)

# %%
some_cases = []

for cluster in duplicated_clusters:
    nodes = cluster.nodes
    if len(nodes) > 2:
        some_cases.append(nodes)


# %%
idx = 0

# %%
case = some_cases[idx]
for vid_index in case:
    vid_name = index_name_map[vid_index]
    _set = index_set_map[vid_index]

    vid_path = os.path.join(FOLDER, _set, 'videos', vid_name)
    cap = cv2.VideoCapture(vid_path)
    cap.set(1, 0)  # Where frame_no is the frame you want
    ret, im = cap.read()
    plt.figure()
    plt.imshow(im[:,:,::-1])
    plt.title(vid_name + '. set:' + _set + ' .Liveness score:' + str(index_label_map[vid_index]))
    plt.show()

idx += 1

# %%
# 2. Identify same individuals

# %%
THRESH = 0.11

# %%
individual_clusters = cluster_similar(df_dedup, THRESH)

# %%
some_cases = []

for cluster in individual_clusters:
    nodes = cluster.nodes
    if len(nodes) > 2:
        some_cases.append(nodes)


# %%
idx = 40

# %%
case = some_cases[idx]
for vid_index in case:
    vid_name = index_name_map[vid_index]
    _set = index_set_map[vid_index]

    vid_path = os.path.join(FOLDER, _set, 'videos', vid_name)
    cap = cv2.VideoCapture(vid_path)
    cap.set(1, 0)  # Where frame_no is the frame you want
    ret, im = cap.read()
    plt.figure()
    plt.imshow(im[:,:,::-1])
    plt.title(vid_name + '. set:' + _set + ' .Liveness score:' + str(index_label_map[vid_index]))
    plt.show()
 
idx += 1

# %%
for i, s in enumerate(individual_clusters):
    nodes = s.nodes
    for node in nodes:
        df_dedup.loc[node, 'individual_id'] = i
df_dedup['individual_id'] = df_dedup['individual_id'].astype(int)

# %%
df_dedup

# %%
df_dedup[df_dedup.set=='train'].individual_id.nunique()

# %%
name_df = pd.concat([
    pd.read_csv('data/vn_boy_name.txt', names=['name']),
    pd.read_csv('data/vn_girl_name.txt', names=['name']),
])

name_df = name_df.sample(frac=1.0, replace=False, random_state=67).reset_index(drop=False)
index_indv_name_map = name_df['name'].to_dict()

# %%
df_dedup['individual_name'] = df_dedup['individual_id'].map(index_indv_name_map)

# %%
df_dedup.individual_name.value_counts()

# %%
df_dedup = df_dedup.reset_index(drop=True)

# %%
kfold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=67)

# %%
fold = 0
df_train_set = df_dedup[df_dedup.set=='train']
for train_indices, val_indices in kfold.split(df_train_set, df_train_set['liveness_score'], groups=df_train_set['individual_id']):
    df_dedup.loc[val_indices, 'fold'] = fold
    fold += 1

# %%
df_train_set = df_dedup[df_dedup.set=='train']

# %%
set(df_train_set[df_train_set.individual_id == 0]).intersection(set(df_train_set[df_train_set.fold!=0].individual_id))

# %%
df_train_set.groupby('fold').liveness_score.mean()

# %%
df_train_set[df_train_set.fold==0].liveness_score.value_counts()

# %%
df_train_set[df_train_set.fold==0].individual_id.nunique()

# %%
cnt = df_train_set.individual_id.value_counts()

# %%
df_dedup.drop(['index', 'min_dist', 'dup_id'], axis=1).to_csv('data/identified_metadata.csv', index=False)

# %%


# %%




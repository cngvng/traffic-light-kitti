import pandas as pd
import os
import sys

sys.path.append("/workspaces/PheNet-Traffic_light/")

data_path = 'kitti/3d_ground_truth'
stage = 'raw'

txt_files = []
for folder in range(11):
    folder_str = str(folder).zfill(2)
    txt_file = os.path.join(data_path, folder_str, f"kitti{folder_str}_sign_position_coordinates_{stage}.txt")
    if os.path.exists(txt_file):
        txt_files.append(txt_file)

dataframes = []
for txt_file in txt_files:
    df = pd.read_csv(txt_file, sep=';')
    dataframes.append(df)

df_merged = pd.concat(dataframes)

# df_merged['imageidx'] = df_merged['imageidx'].apply(lambda x: str(x).zfill(6) + '.png')

df_merged.to_csv('kitti/3d_ground_truth_traffic_sign_positions.csv', index=False)
for imageidx, group in df_merged.groupby("imageidx"):
    imageidx_stt = str(imageidx).zfill(6)
    with open(f"/workspaces/PheNet-Traffic_light/kitti/training/label_traffic_sign_{stage}/{imageidx_stt}.txt", "w") as f:
        for i, row in group.iterrows():
            gt_id = row['gt_id']
            x = row["x"]
            y = row["y"]
            z = row["z"]

            f.write(f"{gt_id} {x} {y} {z}\n")

# for imageidx, group in df_merged.groupby("imageidx"):
#     imageidx_stt = str(imageidx).zfill(6)
#     with open(f"/workspaces/PheNet-Traffic_light/kitti/ImageSets/train.txt", "w") as f:
#         f.write(f"{imageidx_stt}\n")
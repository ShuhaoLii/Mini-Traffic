import pandas as pd

raw_data = pd.read_csv('/home/lsh_23110240117/Mini_Pretrain/Mini_Traffic/dataset/PeMS_road_speed_40.csv',index_col=0)
raw_data = raw_data.iloc[:,1:]
raw_data = raw_data.set_index(raw_data['Sample Time'])
raw_data = raw_data.iloc[:,1:]
raw_data.to_csv('/home/lsh_23110240117/Mini_Pretrain/Mini_Traffic/dataset/PeMS_road_speed_40.csv')
print(raw_data)
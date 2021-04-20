import pandas as pd
import numpy as np


df = pd.read_csv("datasets/CoST.csv")
pd.set_option("display.max_rows", None, "display.max_columns", None)

df.columns = df. columns. str. replace(' ','')
df.insert(loc=2, column='repetition', value="")
df['repetition'] = (df['frame'] != (df['frame'].shift(1) + 1)).astype(int).cumsum()
print(df.columns)
grouped = df.groupby(['gesture', 'repetition', 'subject', 'variant'])

labels = []
training_frames = []

for name, touch in grouped:
    # print('\nCREATE TABLE {}'.format(name))
    # saving label gesture only
    labels.append(name[0])
    frames = []
    for row_index, row in touch.iterrows():
        first_frame = row[5:].to_numpy().reshape(8, 8)
        # sort by opposite idx in opposite way
        idx = [7, 6, 5, 4, 3, 2, 1, 0]
        first_frame_sorted = first_frame[idx]
        frames.append(first_frame_sorted)
    training_frames.append(frames)

print("length training frames", len(training_frames))
print("length labels", len(labels))
np.save("datasets/frames.npy", np.asarray(training_frames))
np.save("datasets/labels.npy", np.asarray(labels))

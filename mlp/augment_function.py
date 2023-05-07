import numpy as np   
import random as rd
import pandas as pd

def augmented(X_train, y_train):
        X_train_augmented = X_train.copy()
        y_train_augmented = y_train.copy()

        for _, row in X_train_augmented.iterrows(): 
            augmentation = rd.choice(['mirror', 'shift'])
            
            if augmentation == 'mirror':
                horizzontal_mirror = [k for k in range(len(X_train_augmented.columns)) if k % 3 == 0 and row[k] != 0]
                row[horizzontal_mirror] = 1 - row[horizzontal_mirror]

            else:
                shift_factor = rd.uniform(-0.10, 0.10)
                shift = [k for k in range(len(X_train_augmented.columns)) if row[k] != 0 and k % 3 != 2]
                row[shift] += shift_factor
                row[row < 0] = 0
                row[row > 1] = 0
 
        X_train_augmented = X_train_augmented.append(X_train)
        y_train_augmented = np.append(y_train_augmented, y_train)
        return X_train_augmented, y_train_augmented


'''
X_train = pd.read_csv('mini_dataset.csv')
#drop the columns video_name, video_frame and skill_id
X_train = X_train.drop(['video_name', 'video_frame', 'skill_id'], axis=1)

y_train = pd.read_csv('mini_y.csv')

X_train, y_train = augment(X_train, y_train)
X_train.to_csv('Augmented_provaaaaa.csv', index=False)
pd.DataFrame(y_train, columns=['skill_id']).to_csv('y_train_provaaaa.csv', index=False)

'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('./class_percentages_cameras.csv')
    label_to_class = {
        1: 'PCB',
        2: 'Adapter_Box',
        3: 'Tablar',
        4: 'Table'
        5: 'Floor'
        
    }

    print(df)
    cameras = list(df['camera'])
    percentages = dict()
    for i in range(0,4):
        percentages[i] = df[str(i)]

    x_axis = np.arange(len(cameras))
    width = 0.1
    plt.bar(x_axis - 3 * 0.5*width, percentages[0], width=width, label = label_to_class[0])
    plt.bar(x_axis - 1 * 0.5*width, percentages[1], width=width, label = label_to_class[1])
    plt.bar(x_axis + 1 * 0.5*width, percentages[2], width=width, label = label_to_class[2])
    plt.bar(x_axis + 3 * 0.5*width, percentages[3], width=width, label = label_to_class[3])

    plt.xticks(x_axis, cameras)

    plt.xlabel('Used camera')
    plt.ylabel('Percentage for each class')

    plt.ylim((0, 50))

    plt.legend(loc='upper left')
    plt.savefig('./low_budget_dataset.pdf')

if __name__ == '__main__':
    main()

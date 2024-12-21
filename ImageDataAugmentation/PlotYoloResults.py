import pandas as pd
import matplotlib.pyplot as plt
import csv

plt.rcParams['font.sans-serif']=['SimHei']

def drawLineChart(title, epochText, lossText, legendText, epochData, lossData, resultsFileName):
    plt.figure(figsize=(8, 4))      # 设置画布的尺寸
    plt.title(title, fontsize=10)      # 标题，并设定字号大小
    plt.xlabel(epochText, fontsize=12)     # 设置x轴，并设定字号大小
    plt.ylabel(lossText, fontsize=12)      # 设置y轴，并设定字号大小

    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    #, marker = 'o' , marker='+'  , marker='*'
    plt.plot(epochData, lossData, color="blue", linewidth=1, linestyle='solid', label=legendText)
    #plt.plot(data['Epoch'], data['trainObjLoss'], color="darkblue", linewidth=1, linestyle=':', label='目标损失')
    #plt.plot(data['Epoch'], data['trainClsLoss'], color="goldenrod", linewidth=1, linestyle='-.', label='类别损失')

    plt.legend(loc=7)  #
    plt.savefig("./" + resultsFileName, dpi=200)


if __name__ == "__main__":
    print("This is the start of main program ")
    datafile = 'E:/newpaper2024/Paper2/data/GuizhouResults.csv'
    data = pd.read_csv(datafile)
    print(data.columns)
    #['               epoch', '      train/box_loss', '      train/obj_loss',
    #'      train/cls_loss', '   metrics/precision', '      metrics/recall',
    #'     metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', '        val/box_loss',
    #'        val/obj_loss', '        val/cls_loss', '               x/lr0',
    #'               x/lr1', '               x/lr2']
    #print(data['               epoch'])
    #print(data['      train/box_loss'])
    drawLineChart('Training/Box Loss', 'Epoch', 'Box Loss', 'Box Loss',
                  data['               epoch'], data['      train/box_loss'], 'TrainingBoxLoss.png')
    drawLineChart('Training/Object Loss', 'Epoch', 'Object Loss', 'Object Loss',
                  data['               epoch'], data['      train/obj_loss'], 'TrainingObjectLoss.png')
    drawLineChart('Training/Class Loss', 'Epoch', 'Class Loss', 'Class Loss',
                  data['               epoch'], data['      train/cls_loss'], 'TrainingClsLoss.png')

    drawLineChart('Metrics/Precision', 'Epoch', 'Precision', 'Precision',
                  data['               epoch'], data['   metrics/precision_0.88'], 'metricsPrecision.png')
    drawLineChart('Metrics/Recall', 'Epoch', 'Recall', 'Recall',
                  data['               epoch'], data['      metrics/recall_0.77'], 'metricsRecall.png')

    drawLineChart('Metrics/mAP@0.5', 'Epoch', 'mAP@0.5', 'mAP@0.5',
                  data['               epoch'], data['     metrics/mAP_0.5_0.85'], 'metricsMAP0.5.png')

    drawLineChart('Metrics/mAP@0.5:0.95', 'Epoch', 'mAP@0.5:0.95', 'mAP@0.5:0.95',
                  data['               epoch'], data['metrics/mAP_0.5:0.95_0.49'], 'metricsMAP0.5_0.95.png')
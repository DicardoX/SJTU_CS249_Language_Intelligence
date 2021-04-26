# Test_AUC_EER_Score.py

from vad_utils import read_label_from_file
from evaluate import get_metrics

# Label path
label_path = "../vad/data/train_label.txt"
labels_list = read_label_from_file(label_path, frame_size=0.03, frame_shift=0.015)
labels_list = sorted(labels_list.items(), key=lambda d: d[0])
for i in range(len(labels_list)):
    labels_list[i] = labels_list[i][1]

# Prediction path
pred_path = "./output/train_prediction.txt"
pred_list = read_label_from_file(pred_path, frame_size=0.03, frame_shift=0.015)
pred_list = sorted(pred_list.items(), key=lambda d: d[0])
for i in range(len(pred_list)):
    pred_list[i] = pred_list[i][1]

print("Amount of prediction lists:", len(pred_list))

total_auc = 0
total_eer = 0
for i in range(len(pred_list)):
    # 预测结果分帧后的帧数可能和标签分帧后的帧数不同，直接补0
    if len(pred_list[i]) < len(labels_list[i]):
        count = 0
        delta_size = (len(labels_list[i]) - len(pred_list[i]))
        while count < delta_size:
            pred_list[i].append(0)
            count += 1
    elif len(pred_list[i]) > len(labels_list[i]):
        count = 0
        delta_size = (len(pred_list[i]) - len(labels_list[i]))
        while count < delta_size:
            labels_list[i].append(0)
            count += 1
    # print(len(pred_list[i]), len(labels_list[i]))

    if i % 100 == 0 and i != 0:
        print("Iteration", i, "| Current AUC:", float(total_auc / i), "| Current EER:", float(total_eer / i))
    cur_auc, cur_eer = get_metrics(pred_list[i], labels_list[i])
    total_auc += cur_auc
    total_eer += cur_eer

print("AUC:", float(total_auc / len(pred_list)), "| EER:", float(total_eer / len(pred_list)))

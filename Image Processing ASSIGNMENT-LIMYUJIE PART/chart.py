import matplotlib.pyplot as plt

# 你最新的 Random Forest 实验数据
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
scores = [84.00, 84.00, 84.00, 84.00]

# 对应你图片里的四种颜色 (蓝, 绿, 红, 紫)
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3']

# 设置图表大小
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, scores, color=colors)

# 在每个柱子上方添加百分比数字
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.2f}%', ha='center', va='bottom')

# 设置Y轴范围和标签
plt.ylim(0, 100)
plt.ylabel('Percentage (%)')
plt.title('Performance of Surface Defect Handling Pipeline')

# 自动保存为图片文件
plt.savefig('figure_4_5_bar_chart.png', dpi=300)
plt.show()
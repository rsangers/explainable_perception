def mass_center(segmentation):
    total_sum = segmentation.sum()
    x_sum = 0
    y_sum = 0
    for i in range(segmentation.size(0)):
        for j in range(segmentation.size(1)):
            x_sum += j*segmentation[i][j]
            y_sum += i*segmentation[i][j]
    return float(x_sum/total_sum), float(y_sum/total_sum)

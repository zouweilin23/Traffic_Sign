import utils.autoanchor as autoAC

# 对数据集进行重新计算anchors    第一次k-means聚类下参数为9,640,5.0,1000
new_anchors = autoAC.kmean_anchors('./data/alldata.yaml',9,640,5.0,1000,True) # 
print(new_anchors)

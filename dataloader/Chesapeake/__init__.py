'''
    Chesapeake数据集需求
    输入影像有2种：
        - NIAP HR 影像 1m
        - LandSat LR 影像 30m
    预测标签有3种：
        - NLCD LR 土地覆盖制图 30m 16类
        - LC HR 土地覆盖制图 1m 8类
        - Building 建筑物掩膜
    Note:
        - LR 和 HR 影像的 size 不一致，因此同时使用 LR and HR 需要裁剪
    根据任务需求有三种组合
        - 只要 LR Label，因为 HR size 不一样
        - 只要 HR Label
        - Low-to-high task 两个都需要，为啥？ 因为算 metric 需要 HR Label
        - 测试反金字塔分类器的任务，也需要 both of label，因为需要同时获取 LR and HR Label，算 loss

'''

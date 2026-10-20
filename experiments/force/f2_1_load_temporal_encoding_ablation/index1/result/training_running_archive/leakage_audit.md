# r1 泄漏边界

原始parquet只通过LeRobotDataset明确episodes=Train36或Dev12读取；本地准备也按episode_index过滤。远程整文件SHA256核验是字节校验，不读取Final样本语义。Final列表仅元数据，不进训练、归一化或评估。F1RM Formal5不混入。

state/model/VAE真实6D，current/master不进入batch；A0 batch无observation.load。A1只有load_t，A2按episode内t-14:t首帧左填充。标准化仅Train36，共用均值/标准差，无接触标签。未来action仅作监督，不作为推理latent，评估policy.eval()+zero latent；padding按episode边界排除。

历史8D P_hist只为确认已有主指标流水线，不用于新训练初始化/统计/替代A0。新实验同seed/common hash，固定budget/保存/selection，无后验threshold/窗口/额外variant搜索，无contact/current/residual branch。完整科学与机制结论仍待100k结果。

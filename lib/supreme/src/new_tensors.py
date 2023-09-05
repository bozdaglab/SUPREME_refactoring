def create_new_x(NODE_NETWORKS):
    for netw in NODE_NETWORKS:
        file = PATH / dataset_name / f"{netw}.pkl"
        with open(file, "rb") as f:
            feat = pickle.load(f)
            if not any(FEATURE_SELECTION_PER_NETWORK):
                values = feat.values
            else:
                if (
                    TOP_FEATURES_PER_NETWORK[NODE_NETWORKS.index(netw)]
                    < feat.values.shape[1]
                ):
                    feat_flat = [
                        item for sublist in feat.values.tolist() for item in sublist
                    ]
                    feat_temp = robjects.FloatVector(feat_flat)
                    robjects.globalenv["feat_matrix"] = robjects.r("matrix")(feat_temp)
                    robjects.globalenv["feat_x"] = robjects.IntVector(feat.shape)
                    robjects.globalenv["labels_vector"] = robjects.IntVector(
                        labels.tolist()
                    )
                    robjects.globalenv["top"] = TOP_FEATURES_PER_NETWORK[
                        NODE_NETWORKS.index(netw)
                    ]
                    robjects.globalenv["maxBorutaRuns"] = BORUTA_RUNS
                    robjects.r(
                        """
                        require(rFerns)
                        require(Boruta)
                        labels_vector = as.factor(labels_vector)
                        feat_matrix <- Reshape(feat_matrix, feat_x[1])
                        feat_data = data.frame(feat_matrix)
                        colnames(feat_data) <- 1:feat_x[2]
                        feat_data <- feat_data %>%
                            mutate('Labels' = labels_vector)
                        boruta.train <- Boruta(feat_data$Labels ~ ., data= feat_data, doTrace = 0, getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
                        thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
                        boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
                            """
                    )
                    boruta_signif = robjects.globalenv["boruta_signif"]
                    robjects.r.rm("feat_matrix")
                    robjects.r.rm("labels_vector")
                    robjects.r.rm("feat_data")
                    robjects.r.rm("boruta_signif")
                    robjects.r.rm("thr")
                    topx = []
                    for index in boruta_signif:
                        t_index = re.sub("`", "", index)
                        topx.append((np.array(feat.values).T)[int(t_index) - 1])
                    topx = np.array(topx)
                    values = torch.tensor(topx.T, device=device)
                elif (
                    TOP_FEATURES_PER_NETWORK[NODE_NETWORKS.index(netw)]
                    >= feat.values.shape[1]
                ):
                    values = feat.values

        if is_first == 0:
            new_x = torch.tensor(values, device=device).float()
            is_first = 1
        else:
            new_x = torch.cat(
                (new_x, torch.tensor(values, device=device).float()), dim=1
            )

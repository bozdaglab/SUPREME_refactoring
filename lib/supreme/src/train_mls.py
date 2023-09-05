

from project.supreme.src.settings import OPTIONAL_FEATURE_SELECTION, FEATURE_NETWORKS_INTEGRATION, NODE_NETWORKS, ADD_RAW_FEAT
import itertools
import pickle
import torch
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# Running Machine Learning for each possible combination of input network
# Input for Machine Learning algorithm is the concatanation of node embeddings (specific to each combination) and node features (if node feature integration is True)
def ml(save_path, dataset_name):
    NODE_NETWORKS2 = [NODE_NETWORKS[i] for i in trial_combs[trials]]
    netw_base = NODE_NETWORKS2[0]
    emb_file = save_path + "Emb_" + netw_base + ".pkl"
    with open(emb_file, "rb") as f:
        emb = pickle.load(f)

    if len(NODE_NETWORKS2) > 1:
        for netw_base in NODE_NETWORKS2[1:]:
            emb_file = save_path + "Emb_" + netw_base + ".pkl"
            with open(emb_file, "rb") as f:
                cur_emb = pickle.load(f)
            emb = torch.cat((emb, cur_emb), dim=1)

    if ADD_RAW_FEAT == True:
        is_first = 0
        addFeatures = FEATURE_NETWORKS_INTEGRATION
        for netw in addFeatures:
            file = BASE_PATH + "data/" + dataset_name + "/" + netw + ".pkl"
            with open(file, "rb") as f:
                feat = pickle.load(f)
            if is_first == 0:
                allx = torch.tensor(feat.values, device=device).float()
                is_first = 1
            else:
                allx = torch.cat(
                    (allx, torch.tensor(feat.values, device=device).float()), dim=1
                )

        if OPTIONAL_FEATURE_SELECTION == True:
            allx_flat = [item for sublist in allx.tolist() for item in sublist]
            allx_temp = robjects.FloatVector(allx_flat)
            robjects.globalenv["allx_matrix"] = robjects.r("matrix")(allx_temp)
            robjects.globalenv["allx_x"] = robjects.IntVector(allx.shape)
            robjects.globalenv["labels_vector"] = robjects.IntVector(labels.tolist())
            robjects.globalenv["top"] = BORUTA_TOP_FEATURES
            robjects.globalenv["maxBorutaRuns"] = BORUTA_RUNS
            robjects.r(
                """
                require(rFerns)
                require(Boruta)
                labels_vector = as.factor(labels_vector)
                allx_matrix <- Reshape(allx_matrix, allx_x[1])
                allx_data = data.frame(allx_matrix)
                colnames(allx_data) <- 1:allx_x[2]
                allx_data <- allx_data %>%
                    mutate('Labels' = labels_vector)
                boruta.train <- Boruta(allx_data$Labels ~ ., data= allx_data, doTrace = 0, getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
                thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
                boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
                    """
            )
            boruta_signif = robjects.globalenv["boruta_signif"]
            robjects.r.rm("allx_matrix")
            robjects.r.rm("labels_vector")
            robjects.r.rm("allx_data")
            robjects.r.rm("boruta_signif")
            robjects.r.rm("thr")
            topx = []
            for index in boruta_signif:
                t_index = re.sub("`", "", index)
                topx.append((np.array(allx).T)[int(t_index) - 1])
            topx = np.array(topx)
            emb = torch.cat((emb, torch.tensor(topx.T, device=device)), dim=1)
            print("Top " + str(BORUTA_TOP_FEATURES) + " features have been selected.")
        else:
            emb = torch.cat((emb, allx), dim=1)

    data = Data(x=emb, y=labels)
    train_mask = np.array([i in set(train_valid_idx) for i in range(data.x.shape[0])])
    data.train_mask = torch.tensor(train_mask, device=device)
    test_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
    data.test_mask = torch.tensor(test_mask, device=device)
    X_train = pd.DataFrame(data.x[data.train_mask].numpy())
    X_test = pd.DataFrame(data.x[data.test_mask].numpy())
    y_train = pd.DataFrame(data.y[data.train_mask].numpy()).values.ravel()
    y_test = pd.DataFrame(data.y[data.test_mask].numpy()).values.ravel()

    ml = MLModels(model=INT_MOTHOD, x_train=X_train, y_train=y_train)
    return ml.train_ml_model_factory()
    


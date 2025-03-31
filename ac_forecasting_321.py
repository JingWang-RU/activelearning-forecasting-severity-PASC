import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.optim as optim
# Define MLP classifier with sparse and mean mapping layers
class AttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=126):
        super(AttentionMLP, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)  # Per-feature attention
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim,         # Extract important features for each training example
        feature_importance_dict = {}        for i_train in range(len(filenames_train)):
            filename = filenames_train[i_train]
            feature_importance_dict[filename] = []

            for idx_train in top_features_per_sample[i_train]:
                event_idx = idx_train // total_feature_dim
                feature_pos = idx_train % total_feature_dim
                
                # Get event name
                if feature_pos < 32:
                    event_name = event_mapping.get(filename, ["Unknown"])[event_idx] if event_idx < len(event_mapping.get(filename, [])) else f"Unknown Event {event_idx}"
                else:
                    event_name = "Time"
                
                # Record the feature importance score
                feature_importance_score = feature_importance[i_train, idx_train]
                feature_importance_dict[filename].append((event_name, feature_importance_score))
        
        # Prepare data for saving
        for filename, feature_list in feature_importance_dict.items():
            for event, importance in feature_list:
                all_feature_data.append({
                    "filename": filename,
                    "event_name": event,
                    "importance": importance,
                    "sub_iteration": i_subiter
                })

    # Convert to DataFrame and save
    df_feature_importance = pd.DataFrame(all_feature_data)
    return df_feature_importance

print("finish loading")

# Load feature data
lc_files = ["PMC10008181", "PMC10077184", "PMC10129030", "PMC10173208", "PMC10284064", "PMC10469423", "PMC10476922", "PMC8132077", "PMC8511908", "PMC8606980", "PMC8850995", "PMC8958594", "PMC9066079", "PMC9451509", "PMC9514285", "PMC9633038", "PMC8405236","PMC9451509"]
des_path = "/data/wangj47/script/annote/activelearning/am_18_llm"
data_list = []
for lc_file in lc_files:
    file_path = os.path.join(des_path, lc_file + '.txt')
    df_temp = pd.read_csv(file_path, sep='\t', names=['event', 'time'], dtype={'event': str, 'time': float})
    df_temp['filename'] = lc_file
    data_list.append(df_temp)
df_features = pd.concat(data_list, ignore_index=True)

# Load label data
am_path = "/data/wangj47/script/annote/activelearning/"
df_labels = pd.read_csv(am_path+"am_risk_annote_18.csv")
print("load data")

# Initialize the reduction layer
reducer = EmbeddingReducer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer("neuml/pubmedbert-base-embeddings").to(device)
# Define a Linear Projection Layer
# Generate event embeddings
df_features["embedding"] = list(get_event_embeddings(df_features["event"].tolist()).detach().numpy())

# Normalize time feature
scaler = StandardScaler()
df_features["time_normalized"] = scaler.fit_transform(df_features[["time"]])
df_features['time_normalized'] = [[x] for x in df_features['time_normalized']]
print("feature generated")

# Set maximum number of event/time pairs per filename
MAX_EVENTS = 150
FEATURE_DIM = 32

# Aggregate features by filename
df_grouped = df_features.groupby("filename").agg({
    "embedding": lambda x: pad_or_truncate_embeddings(list(x), MAX_EVENTS, FEATURE_DIM),
    "time_normalized": lambda x: pad_or_truncate_time(list(x), MAX_EVENTS, 1)
}).reset_index()

df_merged = df_grouped.merge(df_labels, on="filename", how="left").fillna(0)
X = df_merged.apply(lambda row: np.hstack([row['embedding'], row['time_normalized']]), axis=1)
# Merge labels
y = df_merged["risk"].values
print("merged ")

filenames = df_merged["filename"].values  # Store filenames
num_train_test=[[11/18,3/7],[5/9,3/8],[1/2,1/3],[4/9,3/10]]
num_test = [4,5,6,7]
num_iterations = 20

for j, train_test_ratio in enumerate(num_train_test):
    test_size_folder = os.path.join(base_dir, f"test_size_{num_test[j]}")
    os.makedirs(test_size_folder, exist_ok=True)
    df_accuracy = pd.DataFrame()
    for iteration in range(num_iterations):
        print("iterat: ",iteration)
        iteration_folder = os.path.join(test_size_folder, f"iteration_{iteration+1}")
        os.makedirs(iteration_folder, exist_ok=True)
        acc_folder = iteration_folder#os.path.join(test_size_folder, f"accuracy")
        # os.makedirs(acc_folder, exist_ok=True)

        X_initial, X_pool, y_initial, y_pool, filenames_train, filenames_pool = train_test_split(
            X, y, filenames, test_size=train_test_ratio[0], random_state=42
        )
        
        X_test, X_train, y_test, y_train, filenames_test, filenames_train = train_test_split(
            X_initial, y_initial, filenames_train, test_size=train_test_ratio[1], random_state=42
        )
        X_train = list(X_train.values)
        y_train = list(y_train)
        X_test = list(X_test.values)
        y_test = list(y_test)
        X_pool = list(X_pool.values)
        # print("train ", len(y_train), "test ", len(y_test), "pool ", len(X_pool))
        feature_shape = len(X_train[0])
        original_X_train, original_y_train = X_train.copy(), y_train.copy()
        original_X_pool, original_y_pool = X_pool.copy(), y_pool.copy()
        original_filenames_train = filenames_train.copy()
        original_filenames_pool = filenames_pool.copy()
        
        X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
        X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long).to(device)
        y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long).to(device)

        result_al_acc, result_al_index = [],[] # Accuracy results for flag=0 (uncertainty)
        result_rd_acc, result_rd_index= [],[]  # Accuracy results for flag=1 (random)
        al_selected_sample, rd_selected_sample = [], []
        num_subiter = len(X_pool)

        for flag in range(2):  # flag=0: uncertainty sampling, flag=1: random sampling
            X_train = original_X_train.copy()
            y_train = original_y_train.copy()
            X_pool = original_X_pool.copy()
            y_pool = original_y_pool.copy()
            filenames_train = original_filenames_train.copy()
            filenames_pool = original_filenames_pool.copy()
            num_subiter = len(X_pool)
            n_train = 20
            model = AttentionMLP(input_dim=len(X_train[0])).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            df_features = pd.DataFrame()
            for i in range(num_subiter):
            # while len(X_pool) >= 1:
                # Train model
                X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long).to(device)
                
                for epoch in range(n_train):  
                    model.train()
                    optimizer.zero_grad()
                    outputs, _ = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()  # Adjust learning rate dynamically
            
                # Evaluate on the test set
                model.eval()
                # Evaluate on test set
                # print("train ", len(y_train))
                with torch.no_grad():
                    predictions, attn_weights = model(X_test_tensor)
                    accuracy = (predictions.argmax(dim=1) == y_test_tensor).float().mean().item()
            
                if flag == 0:
                    result_al_acc.append(accuracy)
                else:
                    result_rd_acc.append(accuracy)
                all_probs =[]
                # Select next sample
                if len(X_pool) > 1:
                    with torch.no_grad():
                        probs, _ = model(to_tensor(X_pool))
                        probs = probs.cpu().numpy()
                        uncertainty = np.abs(probs[:, 0] - probs[:, 1])  # Difference in probability
            
                    if flag == 0:  # Uncertainty Sampling
                        next_idx = np.argmin(uncertainty)
                        result_al_index.append(next_idx)                        
                    else:
                        next_idx = np.random.randint(len(X_pool))
                        result_rd_index.append(next_idx)
                    X_train = np.vstack([X_train, X_pool[next_idx]])
                    y_train = np.append(y_train, y_pool[next_idx])
                    filenames_train = np.append(filenames_train, filenames_pool[next_idx])
                    # Remove from pool
                    X_pool = np.delete(np.array(X_pool), next_idx, axis=0)
                    y_pool = np.delete(np.array(y_pool), next_idx)
                    filenames_pool = np.delete(filenames_pool, next_idx)
                elif len(X_pool) == 1:
                    print("Only one sample left in X_pool. Adding directly to training set.")
                    if isinstance(X_train, np.ndarray):
                        X_train = list(X_train)
                        y_train = list(y_train)
                    X_train.append(X_pool[0])
                    y_train.append(y_pool[0])
                    filenames_train = np.append(filenames_train, filenames_pool[0])
                    # Convert back to NumPy array
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    X_pool, y_pool, filenames_pool = [], [], []
                # Identify important features
                with torch.no_grad():
                    _, attn_weights = model(torch.tensor(np.array(X_train), dtype=torch.float32).to(device))
                    feature_importance = attn_weights.cpu().detach().numpy()
                top_features_per_sample = np.argsort(-feature_importance, axis=1)[:, :top_k]
                featuers_select = detect_feature_importance(iteration_folder, iteration, flag, filenames_train, feature_importance, total_feature_dim, event_mapping, num_subiter, top_k=5)
                df_features = pd.concat([df_features, featuers_select], ignore_index=True)

            feature_file = os.path.join(iteration_folder, f"feature_importance_flag_{flag}_iter_{iteration}.csv")
            df_features.to_csv(feature_file, index=False)
            print("feature importance saved as ", feature_file)

    # Save accuracy results
        acc_data = []
        acc_data.append({
                "iteration": iteration,
                "uncertainty_accuracy": result_al_acc,
                "uncertainty_selected":result_al_index,
                "random_accuracy": result_rd_acc,
                "random_selected": result_rd_index
            })
        iter_accuracy = pd.DataFrame(acc_data)
        df_accuracy = pd.concat([df_accuracy, iter_accuracy], ignore_index=True)

    accuracy_file = os.path.join(acc_folder, f"accuracy_results_split_{j}.csv")
    df_accuracy.to_csv(accuracy_file, index=False)

print("All results saved successfully.")
print(df_accuracy.iloc[0]['uncertainty_accuracy'])

num_iterations = 20

for j, train_test_ratio in enumerate(num_train_test):
    test_size_folder = os.path.join(base_dir, f"test_size_{num_test[j]}")
    os.makedirs(test_size_folder, exist_ok=True)
    df_accuracy = pd.DataFrame()
    for iteration in range(num_iterations):
        print("iterat: ",iteration)
        iteration_folder = os.path.join(test_size_folder, f"iteration_{iteration+1}")
        os.makedirs(iteration_folder, exist_ok=True)
        acc_folder = iteration_folder#os.path.join(test_size_folder, f"accuracy")
        # os.makedirs(acc_folder, exist_ok=True)

        X_initial, X_pool, y_initial, y_pool, filenames_train, filenames_pool = train_test_split(
            X, y, filenames, test_size=train_test_ratio[0], random_state=42
        )
        
        X_test, X_train, y_test, y_train, filenames_test, filenames_train = train_test_split(
            X_initial, y_initial, filenames_train, test_size=train_test_ratio[1], random_state=42
        )
        X_train = list(X_train.values)
        y_train = list(y_train)
        X_test = list(X_test.values)
        y_test = list(y_test)
        X_pool = list(X_pool.values)
        # print("train ", len(y_train), "test ", len(y_test), "pool ", len(X_pool))
        feature_shape = len(X_train[0])
        original_X_train, original_y_train = X_train.copy(), y_train.copy()
        original_X_pool, original_y_pool = X_pool.copy(), y_pool.copy()
        original_filenames_train = filenames_train.copy()
        original_filenames_pool = filenames_pool.copy()
        
        X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
        X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long).to(device)
        y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long).to(device)

        result_al_acc, result_al_index = [],[] # Accuracy results for flag=0 (uncertainty)
        result_rd_acc, result_rd_index= [],[]  # Accuracy results for flag=1 (random)
        al_selected_sample, rd_selected_sample = [], []
        num_subiter = len(X_pool)

        for flag in range(2):  # flag=0: uncertainty sampling, flag=1: random sampling
            X_train = original_X_train.copy()
            y_train = original_y_train.copy()
            X_pool = original_X_pool.copy()
            y_pool = original_y_pool.copy()
            filenames_train = original_filenames_train.copy()
            filenames_pool = original_filenames_pool.copy()
            num_subiter = len(X_pool)
            n_train = 20
            model = AttentionMLP(input_dim=len(X_train[0])).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            df_features = pd.DataFrame()
            for i in range(num_subiter):
            # while len(X_pool) >= 1:
                # Train model
                X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long).to(device)
                
                for epoch in range(n_train):  
                    model.train()
                    optimizer.zero_grad()
                    outputs, _ = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()  # Adjust learning rate dynamically
            
                # Evaluate on the test set
                model.eval()
                # Evaluate on test set
                # print("train ", len(y_train))
                with torch.no_grad():
                    predictions, attn_weights = model(X_test_tensor)
                    accuracy = (predictions.argmax(dim=1) == y_test_tensor).float().mean().item()
            
                if flag == 0:
                    result_al_acc.append(accuracy)
                else:
                    result_rd_acc.append(accuracy)
                all_probs =[]
                # Select next sample
                if len(X_pool) > 1:
                    with torch.no_grad():
                        probs, _ = model(to_tensor(X_pool))
                        probs = probs.cpu().numpy()
                        uncertainty = np.abs(probs[:, 0] - probs[:, 1])  # Difference in probability
            
                    if flag == 0:  # Uncertainty Sampling
                        next_idx = np.argmin(uncertainty)
                        result_al_index.append(next_idx)                        
                    else:
                        next_idx = np.random.randint(len(X_pool))
                        result_rd_index.append(next_idx)
                    X_train = np.vstack([X_train, X_pool[next_idx]])
                    y_train = np.append(y_train, y_pool[next_idx])
                    filenames_train = np.append(filenames_train, filenames_pool[next_idx])
                    # Remove from pool
                    X_pool = np.delete(np.array(X_pool), next_idx, axis=0)
                    y_pool = np.delete(np.array(y_pool), next_idx)
                    filenames_pool = np.delete(filenames_pool, next_idx)
                elif len(X_pool) == 1:
                    print("Only one sample left in X_pool. Adding directly to training set.")
                    if isinstance(X_train, np.ndarray):
                        X_train = list(X_train)
                        y_train = list(y_train)
                    X_train.append(X_pool[0])
                    y_train.append(y_pool[0])
                    filenames_train = np.append(filenames_train, filenames_pool[0])
                    # Convert back to NumPy array
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    X_pool, y_pool, filenames_pool = [], [], []
                # Identify important features
                with torch.no_grad():
                    _, attn_weights = model(torch.tensor(np.array(X_train), dtype=torch.float32).to(device))
                    feature_importance = attn_weights.cpu().detach().numpy()
                top_features_per_sample = np.argsort(-feature_importance, axis=1)[:, :top_k]
                featuers_select = detect_feature_importance(iteration_folder, iteration, flag, filenames_train, feature_importance, total_feature_dim, event_mapping, num_subiter, top_k=5)
                df_features = pd.concat([df_features, featuers_select], ignore_index=True)

            feature_file = os.path.join(iteration_folder, f"feature_importance_flag_{flag}_iter_{iteration}.csv")
            df_features.to_csv(feature_file, index=False)
            print("feature importance saved as ", feature_file)

    # Save accuracy results
        acc_data = []
        acc_data.append({
                "iteration": iteration,
                "uncertainty_accuracy": result_al_acc,
                "uncertainty_selected":result_al_index,
                "random_accuracy": result_rd_acc,
                "random_selected": result_rd_index
            })
        iter_accuracy = pd.DataFrame(acc_data)
        df_accuracy = pd.concat([df_accuracy, iter_accuracy], ignore_index=True)

    accuracy_file = os.path.join(acc_folder, f"accuracy_results_split_{j}.csv")
    df_accuracy.to_csv(accuracy_file, index=False)

print("All results saved successfully.")
print(df_accuracy.iloc[0]['uncertainty_accuracy'])




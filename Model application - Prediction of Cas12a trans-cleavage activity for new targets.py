import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from itertools import combinations
import joblib
from sklearn.preprocessing import StandardScaler

encoded_dict = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'C': [0, 0, 0, 1]
}

def one_hot_encode(sequence):
    return [encoded_dict.get(base, [0, 0, 0, 0]) for base in sequence]

def calculate_gc_content(sequence):
    g = sequence.count('G')
    c = sequence.count('C')
    return (g + c) / len(sequence) * 100

def get_reverse_complement(sequence):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement[base] for base in reversed(sequence))

file_path = '/result/crRNA_sequences.xlsx'
data = pd.read_excel(file_path, header=None)
gc_contents = []
encoded_data = []

sequence = ''.join(data.iloc[0, 0])

for i in range(len(sequence) - 23):
    segment = sequence[i:i + 24]
    front_20 = segment[:20]
    back_4 = segment[20:]

    reverse_complement = get_reverse_complement(back_4)
    new_sequence = front_20 + reverse_complement
    gc_content = calculate_gc_content(new_sequence)
    gc_contents.extend([gc_content] * 24)
    encoded_bases = one_hot_encode(new_sequence)
    encoded_data.extend(encoded_bases)

encoded_df = pd.DataFrame(encoded_data, columns=['A', 'T', 'G', 'C'])
gc_df = pd.DataFrame(gc_contents, columns=['GC_content'])
final_df = pd.concat([encoded_df, gc_df], axis=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

output_path = "/result"

file_path = os.path.join(output_path, "processed_crRNA_sequences.xlsx")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"file {file_path} not exist")

try:
    data = pd.read_excel(file_path)
    print(data.head())
except Exception as e:
    print(f"read Excel file error: {e}")

num_rows = data.shape[0]
if num_rows % 24 != 0:
    raise ValueError(f"line {num_rows} Not a multiple of 24, please check the data")

window_size = 24
crrna_data = []

for i in range(0, num_rows, window_size):
    group_data = data.iloc[i:i + window_size].copy()
    gc_content = group_data['GC_content'].mean()
    bases = group_data[['A', 'T', 'G', 'C']].values.flatten()
    row = np.concatenate([[gc_content], bases])
    crrna_data.append(row)

columns = (
        ['GC_content_mean'] +
        [f'Base_{i}_{b}' for i in range(1, window_size + 1) for b in ['A', 'T', 'G', 'C']]
)

crrna_df = pd.DataFrame(crrna_data, columns=columns)
print(crrna_df.head())

def add_polynomial_features(df, features, degree=2):
    new_features = df.copy()
    for feature in features:
        new_features[f'{feature}^2'] = new_features[feature] ** 2
    interactions = []
    for (feature1, feature2) in combinations(features, 2):
        interactions.append(new_features[feature1] * new_features[feature2])
    interaction_df = pd.concat(interactions, axis=1)
    interaction_df.columns = [f'{feature1}*{feature2}' for feature1, feature2 in combinations(features, 2)]
    new_features = pd.concat([new_features, interaction_df], axis=1)
    return new_features

base_features = [f'Base_{i}_{b}' for i in range(1, window_size + 1) for b in ['A', 'T', 'G', 'C']]
crrna_df_expanded = add_polynomial_features(crrna_df, base_features)


X_new_expanded = crrna_df_expanded[
    base_features + ['GC_content_mean'] +
    [col for col in crrna_df_expanded.columns if '^2' in col or '*' in col]
    ]

scaler_path = os.path.join(output_path, 'scaler_intermediate.pkl')
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"The normalizer file {scaler_path} does not exist, please check the path or file name")
scaler = joblib.load(scaler_path)
if not set(X_new_expanded.columns).issubset(set(scaler.feature_names_in_)):
    raise ValueError("The new feature does not match the feature of the normalizer. Check the feature name.")

X_new_scaled = scaler.transform(X_new_expanded)

# Define the improved neural network model (IntermediateNN model)
class ImprovedNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.bn3 = nn.BatchNorm1d(hidden_units)
        self.fc4 = nn.Linear(hidden_units, hidden_units)
        self.bn4 = nn.BatchNorm1d(hidden_units)
        self.fc5 = nn.Linear(hidden_units, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x

model_path_intermediate = os.path.join(output_path, "improved_nn_model.pth")
input_dim = X_new_expanded.shape[1]
output_dim = 168

hidden_units = 910
dropout_rate = 0.3087

model = ImprovedNN(input_dim=input_dim, output_dim=output_dim, hidden_units=hidden_units, dropout_rate=dropout_rate)
model.load_state_dict(torch.load(model_path_intermediate))
model.to(device)
model.eval()

X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    intermediate_predictions = model(X_new_tensor)
    predictions_df = pd.DataFrame(intermediate_predictions.cpu().numpy(), columns=[f'Feature_{i}' for i in range(output_dim)])

feature_names = [
    'van der Waals', 'Electrostatic', 'Polar Solvation',
    'Non-Polar Solv.', 'TOTAL', 'Hbond', 'RMSF'
]

new_column_names = [f"{name}_{i+1}_mean" for name in feature_names for i in range(24)]
predictions_df.columns = new_column_names
output_file_path = os.path.join(output_path, "predictions_for_crRNA_sequences.xlsx")
predictions_df.to_excel(output_file_path, index=False)
combined_df = pd.concat([crrna_df_expanded[['GC_content_mean'] + base_features], predictions_df], axis=1)
combined_output_file_path = os.path.join(output_path, "combined_crRNA_features_predictions.xlsx")
combined_df.to_excel(combined_output_file_path, index=False)
print(f"The combined features and predictions have been saved to: {combined_output_file_path}")

class ActivityNN(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate):
        super(ActivityNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 265
hidden_units = 128
dropout_rate = 0.002707760456745485

activity_model = ActivityNN(input_dim, hidden_units, dropout_rate).to(device)
model_save_path = "/result/activity_model.pth"
activity_model.load_state_dict(torch.load(model_save_path))
activity_model.eval()

new_data_path = "/result/combined_crRNA_features_predictions.xlsx"  # Change to the actual data path
new_data = pd.read_excel(new_data_path)
num_bases = 24
base_features = [f'Base_{i}_{b}' for i in range(1, num_bases + 1) for b in ['A', 'T', 'G', 'C']]
X_new = new_data[base_features + ['GC_content_mean'] + [f'{feat}_{i}_mean' for feat in ['van der Waals', 'Electrostatic', 'Polar Solvation', 'Non-Polar Solv.', 'TOTAL', 'Hbond', 'RMSF'] for i in range(1, num_bases + 1)]]

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    predictions = activity_model(X_new_tensor).squeeze().cpu().numpy()

output_path = "/result"
predictions_df = pd.DataFrame(predictions, columns=['Predicted Activity'])
predictions_df.to_excel(os.path.join(output_path, "new_predictions.xlsx"), index=False)
output_path = "result"
combined_file = os.path.join(output_path, "combined_crRNA_features_predictions.xlsx")
activity_predictions_file = os.path.join(output_path, "new_predictions.xlsx")
final_output_file = os.path.join(output_path, "final_combined_predictions.xlsx")
combined_df = pd.read_excel(combined_file)
activity_predictions_df = pd.read_excel(activity_predictions_file, names=["Predicted Activity"])

# Decode unique heat encoding into base sequences
def decode_sequence_from_one_hot(df, window_size=24):
    bases = []
    for i in range(1, window_size + 1):
        base = df[[f'Base_{i}_A', f'Base_{i}_T', f'Base_{i}_G', f'Base_{i}_C']].idxmax(axis=1).apply(lambda x: x[-1])
        bases.append(base)
    return pd.Series([''.join(bases_row) for bases_row in zip(*bases)], name='Decoded_Sequence')
combined_df['Decoded_Sequence'] = decode_sequence_from_one_hot(combined_df)
merged_data = pd.concat([combined_df, activity_predictions_df], axis=1)

# Form a reverse complementary sequence
def generate_complementary_sequence(sequence):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement[base] for base in sequence)

def get_reverse_complement(sequence):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement[base] for base in reversed(sequence))
ts_sequences = merged_data['Decoded_Sequence'].str[:20]
pam_nts_sequences = merged_data['Decoded_Sequence'].str[20:]
guide_sequences = ts_sequences.apply(get_reverse_complement)

sequence_df = pd.DataFrame({
    "5'-TS-3'": ts_sequences,
    "5'-guide-3'": guide_sequences,
    "5'-PAM-NTS-3'": pam_nts_sequences
})

with pd.ExcelWriter(final_output_file, engine='openpyxl') as writer:
    combined_df.to_excel(writer, sheet_name='CRRNA_Data_Expanded', index=False)
    activity_predictions_df.to_excel(writer, sheet_name='Activity_Predictions', index=False)
    merged_data.to_excel(writer, sheet_name='Merged_Data_with_Sequence', index=False)
    sequence_df.to_excel(writer, sheet_name='Split_Sequences', index=False)

print(f"All prediction results have been successfully saved to {final_output_file}")

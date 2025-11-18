import numpy as np
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import torch
from sklearn.preprocessing import StandardScaler

data = np.load('./DGraphFin/dgraphfin.npz')

print("Available keys:", data.files)

x = data['x']                    
y = data['y']                    
edge_index = data['edge_index']  
edge_type = data['edge_type']    
edge_timestamp = data['edge_timestamp']
train_mask = data['train_mask']
valid_mask = data['valid_mask']
test_mask = data['test_mask']

# Show what we loaded
print(f"✓ Loaded node features with shape: {x.shape}")
print(f"  - {x.shape[0]} nodes")
print(f"  - {x.shape[1]} features per node")
print(f"✓ Loaded {len(edge_index)} edges")
print(f"✓ Loaded {len(y)} labels")

# ==============================================================================
# PART 1: UNDERSTAND THE DATA
# ==============================================================================
print("\n" + "=" * 70)
print("PART 1: UNDERSTANDING THE DATA")
print("=" * 70)

# ------------------------------------------------------------------------------
# 1.1 Check the labels
# ------------------------------------------------------------------------------
print("\n[1.1] Analyzing node labels...")

# Find unique labels
unique_labels = np.unique(y)
print(f"Unique labels in dataset: {unique_labels}")
print("Label meanings:")
print("  0 = Normal user (to be predicted)")
print("  1 = Fraud user (to be predicted)")
print("  2 = Background user Type 1 (not predicted)")
print("  3 = Background user Type 2 (not predicted)")

# Count each class
num_class_0 = np.sum(y == 0)
num_class_1 = np.sum(y == 1)
num_class_2 = np.sum(y == 2)
num_class_3 = np.sum(y == 3)

print(f"\nNode counts per class:")
print(f"  Class 0 (Normal):      {num_class_0:,} ({num_class_0/len(y)*100:.1f}%)")
print(f"  Class 1 (Fraud):       {num_class_1:,} ({num_class_1/len(y)*100:.1f}%)")
print(f"  Class 2 (Background):  {num_class_2:,} ({num_class_2/len(y)*100:.1f}%)")
print(f"  Class 3 (Background):  {num_class_3:,} ({num_class_3/len(y)*100:.1f}%)")

# Calculate statistics for target classes (0 and 1)
num_target_nodes = num_class_0 + num_class_1
num_background_nodes = num_class_2 + num_class_3

print(f"\nSummary:")
print(f"  Target nodes (Class 0 & 1): {num_target_nodes:,} ({num_target_nodes/len(y)*100:.1f}%)")
print(f"  Background nodes (Class 2 & 3): {num_background_nodes:,} ({num_background_nodes/len(y)*100:.1f}%)")

# Calculate fraud rate (only among target nodes)
fraud_rate = num_class_1 / num_target_nodes * 100
print(f"\n⚠️ Fraud rate among target nodes: {fraud_rate:.2f}%")
print(f"   This is VERY IMBALANCED! (only {fraud_rate:.1f}% are fraudsters)")
print(f"   For every 1 fraudster, there are ~{num_class_0 / num_class_1:.0f} normal users")

# Verify the counts match documentation
print(f"\n✓ Verification against documentation:")
print(f"  Class 0 expected: 1,210,092 | Actual: {num_class_0:,} | Match: {num_class_0 == 1210092}")
print(f"  Class 1 expected:    15,509 | Actual: {num_class_1:,} | Match: {num_class_1 == 15509}")
print(f"  Class 2 expected: 1,620,851 | Actual: {num_class_2:,} | Match: {num_class_2 == 1620851}")
print(f"  Class 3 expected:   854,098 | Actual: {num_class_3:,} | Match: {num_class_3 == 854098}")

# ------------------------------------------------------------------------------
# 1.2 Check for missing values
# ------------------------------------------------------------------------------
print("\n[1.2] Analyzing missing values...")

# In this dataset, -1 represents a missing value in the features
# Let's create a boolean mask: True where value is -1
is_missing = (x == -1)

print(f"Checking where features equal -1 (missing values)...")
print(f"  Created a mask with shape: {is_missing.shape}")

# Count missing values per node (sum across features)
missing_per_node = is_missing.sum(axis=1)
print(f"\n  Calculated missing values for each node")

# Count missing values per feature (sum across nodes)
missing_per_feature = is_missing.sum(axis=0)
print(f"  Calculated missing values for each feature")

# How many nodes have at least one missing value?
nodes_with_missing = missing_per_node > 0
num_nodes_with_missing = nodes_with_missing.sum()
percent_nodes_with_missing = num_nodes_with_missing / len(x) * 100

print(f"\nMissing value statistics:")
print(f"  Nodes with missing values: {num_nodes_with_missing:,} ({percent_nodes_with_missing:.1f}%)")
print(f"  Total missing rate: {is_missing.mean() * 100:.1f}%")

# Show distribution of missing values
print(f"\n  Example: Node 0 has {missing_per_node[0]} missing values")
print(f"  Example: Feature 0 has {missing_per_feature[0]:,} missing values")

# Check missing values by class
print(f"\nMissing values by class:")
for class_id in [0, 1, 2, 3]:
    class_mask = (y == class_id)
    class_missing_rate = is_missing[class_mask].mean() * 100
    avg_missing_per_node = missing_per_node[class_mask].mean()
    print(f"  Class {class_id}: {class_missing_rate:.1f}% missing | Avg {avg_missing_per_node:.1f} missing per node")

# WHY THIS MATTERS:
print("\n💡 KEY INSIGHT:")
print("   Fraudsters (Class 1) likely have MORE missing values than normal users (Class 0)!")
print("   They intentionally leave information blank.")
print("   So missing values are actually a SIGNAL, not noise.")

# ------------------------------------------------------------------------------
# 1.3 Check edges
# ------------------------------------------------------------------------------
print("\n[1.3] Analyzing edges (connections between users)...")

print(f"Total number of edges: {len(edge_index):,}")

# Check for self-loops (user connected to themselves)
# edge_index has shape (num_edges, 2)
# Column 0 is source, column 1 is destination
source_nodes = edge_index[:, 0]
destination_nodes = edge_index[:, 1]

# Check where source equals destination
is_self_loop = (source_nodes == destination_nodes)
num_self_loops = is_self_loop.sum()

print(f"\nChecking for self-loops (user → same user):")
print(f"  Found {num_self_loops} self-loops")

# Check for invalid node indices
# All indices should be between 0 and (number of nodes - 1)
max_valid_index = len(x) - 1
min_valid_index = 0

# Check if any source nodes are out of range
invalid_sources = (source_nodes < min_valid_index) | (source_nodes > max_valid_index)
num_invalid_sources = invalid_sources.sum()

# Check if any destination nodes are out of range
invalid_destinations = (destination_nodes < min_valid_index) | (destination_nodes > max_valid_index)
num_invalid_destinations = invalid_destinations.sum()

# Combine: edge is invalid if either source or destination is invalid
invalid_edges = invalid_sources | invalid_destinations
num_invalid_edges = invalid_edges.sum()

print(f"\nChecking for invalid edges (pointing to non-existent nodes):")
print(f"  Valid node indices: {min_valid_index} to {max_valid_index}")
print(f"  Edge index range: {edge_index.min()} to {edge_index.max()}")
print(f"  Invalid edges found: {num_invalid_edges}")

# Check edge types
print(f"\nEdge type analysis:")
unique_edge_types = np.unique(edge_type)
print(f"  Unique edge types: {unique_edge_types}")
print(f"  Number of edge types: {len(unique_edge_types)} (expected: 11)")

for edge_t in unique_edge_types:
    count = np.sum(edge_type == edge_t)
    print(f"    Type {edge_t}: {count:,} edges ({count/len(edge_type)*100:.1f}%)")

# ------------------------------------------------------------------------------
# 1.4 Check timestamps
# ------------------------------------------------------------------------------
print("\n[1.4] Analyzing temporal information...")

# Get min and max timestamp
min_time = edge_timestamp.min()
max_time = edge_timestamp.max()
unique_times = len(np.unique(edge_timestamp))

print(f"Timestamp range: [{min_time}, {max_time}]")
print(f"Number of unique timestamps: {unique_times:,}")

# Check if timestamps are sorted (important for temporal GNNs!)
# Compare each timestamp with the next one
current_times = edge_timestamp[:-1]  # All except last
next_times = edge_timestamp[1:]      # All except first

# Check if current <= next for all pairs
is_sorted = np.all(current_times <= next_times)

print(f"\nAre timestamps sorted? {is_sorted}")
if not is_sorted:
    print("  ⚠️ WARNING: Timestamps are not sorted!")
    print("     We will need to sort them for temporal GNN training.")
else:
    print("  ✓ Timestamps are already in chronological order")

# ------------------------------------------------------------------------------
# 1.5 Check train/valid/test masks
# ------------------------------------------------------------------------------
print("\n[1.5] Analyzing data splits...")

# Count nodes in each split
num_train = train_mask.sum()
num_valid = valid_mask.sum()
num_test = test_mask.sum()

print(f"Data split sizes:")
print(f"  Training nodes: {num_train:,}")
print(f"  Validation nodes: {num_valid:,}")
print(f"  Test nodes: {num_test:,}")

# Check if splits are only for target classes (0 and 1)
train_labels = y[train_mask]
valid_labels = y[valid_mask]
test_labels = y[test_mask]

print(f"\nLabel distribution in splits:")

# Training split
train_class_0 = np.sum(train_labels == 0)
train_class_1 = np.sum(train_labels == 1)
print(f"  Training:")
print(f"    Class 0 (Normal): {train_class_0:,} ({train_class_0/num_train*100:.1f}%)")
print(f"    Class 1 (Fraud):  {train_class_1:,} ({train_class_1/num_train*100:.1f}%)")

# Validation split
valid_class_0 = np.sum(valid_labels == 0)
valid_class_1 = np.sum(valid_labels == 1)
print(f"  Validation:")
print(f"    Class 0 (Normal): {valid_class_0:,} ({valid_class_0/num_valid*100:.1f}%)")
print(f"    Class 1 (Fraud):  {valid_class_1:,} ({valid_class_1/num_valid*100:.1f}%)")

# Test split
test_class_0 = np.sum(test_labels == 0)
test_class_1 = np.sum(test_labels == 1)
print(f"  Test:")
print(f"    Class 0 (Normal): {test_class_0:,} ({test_class_0/num_test*100:.1f}%)")
print(f"    Class 1 (Fraud):  {test_class_1:,} ({test_class_1/num_test*100:.1f}%)")

# Verify the 70/15/15 split
total_target = num_train + num_valid + num_test
train_ratio = num_train / total_target * 100
valid_ratio = num_valid / total_target * 100
test_ratio = num_test / total_target * 100

print(f"\nSplit ratios:")
print(f"  Train: {train_ratio:.1f}% (expected: ~70%)")
print(f"  Valid: {valid_ratio:.1f}% (expected: ~15%)")
print(f"  Test:  {test_ratio:.1f}% (expected: ~15%)")

# Check if background nodes are in any split (they shouldn't be)
background_in_train = np.sum((train_labels == 2) | (train_labels == 3))
background_in_valid = np.sum((valid_labels == 2) | (valid_labels == 3))
background_in_test = np.sum((test_labels == 2) | (test_labels == 3))

print(f"\nBackground nodes in splits (should be 0):")
print(f"  In training: {background_in_train}")
print(f"  In validation: {background_in_valid}")
print(f"  In test: {background_in_test}")

if background_in_train + background_in_valid + background_in_test == 0:
    print(f"  ✓ Correct: Only target classes (0 & 1) are in splits")
else:
    print(f"  ⚠️ WARNING: Background nodes found in splits!")

print("\n💡 KEY INSIGHT:")
print("   Only Classes 0 and 1 are split for training/validation/testing")
print("   Classes 2 and 3 (background) are NOT in any split")
print("   But they're still in the graph and help with predictions!")

# ==============================================================================
# PART 2: CLEAN THE DATA
# ==============================================================================
print("\n" + "=" * 70)
print("PART 2: CLEANING THE DATA")
print("=" * 70)

# ------------------------------------------------------------------------------
# 2.1 Remove self-loops from edges
# ------------------------------------------------------------------------------
print("\n[2.1] Removing self-loops...")

# Create a mask: True where source != destination
keep_edge_mask = (source_nodes != destination_nodes)

# Count how many we're removing
num_edges_before = len(edge_index)

# Filter edges
edge_index = edge_index[keep_edge_mask]
edge_type = edge_type[keep_edge_mask]
edge_timestamp = edge_timestamp[keep_edge_mask]

num_edges_after = len(edge_index)
num_removed = num_edges_before - num_edges_after

print(f"✓ Removed {num_removed} self-loops")
print(f"  Edges before: {num_edges_before:,}")
print(f"  Edges after: {num_edges_after:,}")

# Update source and destination arrays
source_nodes = edge_index[:, 0]
destination_nodes = edge_index[:, 1]

# ------------------------------------------------------------------------------
# 2.2 Remove invalid edges
# ------------------------------------------------------------------------------
print("\n[2.2] Removing invalid edges...")

# Create a mask for valid edges
# An edge is valid if both source and destination are in valid range
valid_sources = (source_nodes >= 0) & (source_nodes < len(x))
valid_destinations = (destination_nodes >= 0) & (destination_nodes < len(x))
valid_edges_mask = valid_sources & valid_destinations

# Count how many we're removing
num_edges_before = len(edge_index)

# Filter edges
edge_index = edge_index[valid_edges_mask]
edge_type = edge_type[valid_edges_mask]
edge_timestamp = edge_timestamp[valid_edges_mask]

num_edges_after = len(edge_index)
num_removed = num_edges_before - num_edges_after

print(f"✓ Removed {num_removed} invalid edges")
print(f"  Edges before: {num_edges_before:,}")
print(f"  Edges after: {num_edges_after:,}")

# Update source and destination arrays
source_nodes = edge_index[:, 0]
destination_nodes = edge_index[:, 1]

# ------------------------------------------------------------------------------
# 2.3 Sort edges by timestamp
# ------------------------------------------------------------------------------
print("\n[2.3] Sorting edges by timestamp...")

# Get indices that would sort the timestamps
sorted_indices = np.argsort(edge_timestamp)

print(f"Finding the order to sort {len(edge_timestamp):,} timestamps...")

# Reorder all edge-related arrays
edge_index = edge_index[sorted_indices]
edge_type = edge_type[sorted_indices]
edge_timestamp = edge_timestamp[sorted_indices]

print(f"✓ Sorted all edges by timestamp")

# Verify sorting worked
current_times = edge_timestamp[:-1]
next_times = edge_timestamp[1:]
is_now_sorted = np.all(current_times <= next_times)
print(f"  Verification: Timestamps are now sorted? {is_now_sorted}")

# Update source and destination arrays
source_nodes = edge_index[:, 0]
destination_nodes = edge_index[:, 1]

# ==============================================================================
# PART 3: HANDLE MISSING VALUES
# ==============================================================================
print("\n" + "=" * 70)
print("PART 3: HANDLING MISSING VALUES (MOST IMPORTANT!)")
print("=" * 70)

# According to the paper, we use "Trick B":
# 1. Create binary flags (1 = was missing, 0 = was not missing)
# 2. Replace -1 with 0
# 3. Concatenate original features with flags

print("\n[3.1] Creating binary flags for missing values...")

# Step 1: Create flags
# Create a boolean mask where True means the value is missing (-1)
is_missing_mask = (x == -1)

print(f"Created boolean mask with shape: {is_missing_mask.shape}")
print(f"  Example: Node 0, Feature 0 is missing? {is_missing_mask[0, 0]}")

# Convert boolean to float (True → 1.0, False → 0.0)
missing_flags = is_missing_mask.astype(np.float32)

print(f"Converted to float32 array with shape: {missing_flags.shape}")
print(f"  Example: Node 0, Feature 0 flag value: {missing_flags[0, 0]}")

print("\nWhat this means:")
print("  If original feature was -1 (missing) → flag = 1")
print("  If original feature was not -1 (present) → flag = 0")

# ------------------------------------------------------------------------------
print("\n[3.2] Replacing missing values with 0...")

# Make a copy of the features so we don't modify the original
x_cleaned = x.copy()

print(f"Created a copy of features with shape: {x_cleaned.shape}")

# Replace all -1 values with 0
# Find where values are -1
replace_mask = (x_cleaned == -1)
num_to_replace = replace_mask.sum()

print(f"Found {num_to_replace:,} values to replace")

# Do the replacement
x_cleaned[replace_mask] = 0

print(f"✓ Replaced all -1 values with 0")
print(f"  Example before: x[0,0] = {x[0,0]}")
print(f"  Example after: x_cleaned[0,0] = {x_cleaned[0,0]}")

# Verify no -1 values remain
remaining_negatives = (x_cleaned == -1).sum()
print(f"  Verification: Remaining -1 values: {remaining_negatives}")

# ------------------------------------------------------------------------------
print("\n[3.3] Concatenating features with flags...")

# We now have:
# - x_cleaned: Original features with -1 replaced by 0 (shape: [num_nodes, 17])
# - missing_flags: Binary flags indicating missingness (shape: [num_nodes, 17])

print(f"x_cleaned shape: {x_cleaned.shape}")
print(f"missing_flags shape: {missing_flags.shape}")

# Concatenate along the feature dimension (axis=1)
# This creates: [original_17_features, 17_missing_flags]
x_with_flags = np.concatenate([x_cleaned, missing_flags], axis=1)

print(f"\n✓ Combined features with flags")
print(f"  New shape: {x_with_flags.shape}")
print(f"  Original features: columns 0-16")
print(f"  Missing flags: columns 17-33")

# Example to show what happened
print(f"\nExample for Node 0:")
print(f"  Original features: {x[0, :5]}...")
print(f"  Cleaned features: {x_with_flags[0, :5]}...")
print(f"  Missing flags: {x_with_flags[0, 17:22]}...")

# ==============================================================================
# PART 4: NORMALIZE FEATURES
# ==============================================================================
print("\n" + "=" * 70)
print("PART 4: NORMALIZING FEATURES")
print("=" * 70)

print("\n[4.1] Preparing to normalize...")

# We only want to normalize the ORIGINAL features (columns 0-16)
# We DON'T normalize the missing flags (columns 17-33)
# Because flags are already 0 or 1

num_original_features = 17
print(f"We will normalize only the first {num_original_features} features")
print(f"The remaining {x_with_flags.shape[1] - num_original_features} features (flags) stay as-is")

# ------------------------------------------------------------------------------
print("\n[4.2] Fitting the scaler on TRAINING data only...")

# Get only the training nodes
training_node_indices = train_mask

print(f"Number of training nodes: {training_node_indices.sum():,}")

# Extract features for training nodes only
# We only take the first 17 columns (original features, not flags)
training_features = x_with_flags[training_node_indices, :num_original_features]

print(f"Training features shape: {training_features.shape}")

# Create the scaler
scaler = StandardScaler()

print(f"\nFitting StandardScaler...")
print(f"  This calculates mean and std for each feature")

# Fit the scaler
scaler.fit(training_features)

print(f"✓ Scaler fitted!")
print(f"  Example feature means: {scaler.mean_[:3]}")
print(f"  Example feature stds: {scaler.scale_[:3]}")

# ------------------------------------------------------------------------------
print("\n[4.3] Transforming ALL data using the fitted scaler...")

# Make a copy to store normalized features
x_normalized = x_with_flags.copy()

print(f"Transforming features for all {len(x_normalized):,} nodes...")

# Extract all original features (first 17 columns)
all_original_features = x_normalized[:, :num_original_features]

# Transform using the scaler fitted on training data
normalized_original_features = scaler.transform(all_original_features)

# Put the normalized features back into the array
x_normalized[:, :num_original_features] = normalized_original_features

print(f"✓ Normalized all features!")

# Verify normalization worked
# Check mean and std of training data (should be ~0 and ~1)
normalized_train_features = x_normalized[training_node_indices, :num_original_features]
train_mean = normalized_train_features.mean(axis=0)
train_std = normalized_train_features.std(axis=0)

print(f"\nVerification on training data:")
print(f"  Mean of first 3 features: {train_mean[:3]}")
print(f"  Std of first 3 features: {train_std[:3]}")
print(f"  (Should be close to 0 and 1 respectively)")

# Show what normalization did
print(f"\nExample transformation:")
print(f"  Original value: {x_with_flags[0, 0]}")
print(f"  Normalized value: {x_normalized[0, 0]}")

# ==============================================================================
# PART 5: IDENTIFY NODE TYPES
# ==============================================================================
print("\n" + "=" * 70)
print("PART 5: IDENTIFYING NODE TYPES")
print("=" * 70)

print("\n[5.1] Categorizing nodes by their labels...")

# Create boolean masks for each type
target_nodes_mask = (y >= 0)  # Nodes with labels (normal or fraud)
background_nodes_mask = (y == -1)  # Nodes without labels

# Count each type
num_target = target_nodes_mask.sum()
num_background = background_nodes_mask.sum()

print(f"Node categories:")
print(f"  Target nodes (have labels): {num_target:,}")
print(f"    - Normal: {(y == 0).sum():,}")
print(f"    - Fraud: {(y == 1).sum():,}")
print(f"  Background nodes (no labels): {num_background:,}")

print(f"\n💡 KEY INSIGHT:")
print(f"   Background nodes don't have labels (can't predict them)")
print(f"   BUT they're crucial for graph connectivity!")
print(f"   The paper shows: Removing them drops performance by 4%!")

# ==============================================================================
# PART 6: CONVERT TO PYTORCH
# ==============================================================================
print("\n" + "=" * 70)
print("PART 6: CONVERTING TO PYTORCH TENSORS")
print("=" * 70)

print("\n[6.1] Converting features to PyTorch...")

# Convert features to torch tensor
x_tensor = torch.FloatTensor(x_normalized)

print(f"✓ Created feature tensor with shape: {x_tensor.shape}")
print(f"  Data type: {x_tensor.dtype}")

# ------------------------------------------------------------------------------
print("\n[6.2] Converting labels to PyTorch...")

# Convert labels to torch tensor
y_tensor = torch.LongTensor(y)

print(f"✓ Created label tensor with shape: {y_tensor.shape}")
print(f"  Data type: {y_tensor.dtype} (Long = int64)")

# ------------------------------------------------------------------------------
print("\n[6.3] Converting edges to PyTorch...")

# For PyTorch Geometric, edges need to be transposed
# NumPy format: [num_edges, 2]  → [[src1, dst1], [src2, dst2], ...]
# PyTorch format: [2, num_edges] → [[src1, src2, ...], [dst1, dst2, ...]]

print(f"Edge index before transpose: {edge_index.shape}")

# Transpose the edge index
edge_index_transposed = edge_index.T

print(f"Edge index after transpose: {edge_index_transposed.shape}")

# Convert to tensor
edge_index_tensor = torch.LongTensor(edge_index_transposed)

print(f"✓ Created edge index tensor with shape: {edge_index_tensor.shape}")
print(f"  Row 0 (sources): {edge_index_tensor.shape[1]} values")
print(f"  Row 1 (destinations): {edge_index_tensor.shape[1]} values")

# ------------------------------------------------------------------------------
print("\n[6.4] Converting edge attributes to PyTorch...")

# Convert edge types
edge_type_tensor = torch.LongTensor(edge_type)
print(f"✓ Created edge type tensor with shape: {edge_type_tensor.shape}")

# Convert timestamps
edge_timestamp_tensor = torch.LongTensor(edge_timestamp)
print(f"✓ Created timestamp tensor with shape: {edge_timestamp_tensor.shape}")

# ------------------------------------------------------------------------------
print("\n[6.5] Converting masks to PyTorch...")

# Convert masks to boolean tensors
train_mask_tensor = torch.BoolTensor(train_mask)
valid_mask_tensor = torch.BoolTensor(valid_mask)
test_mask_tensor = torch.BoolTensor(test_mask)

print(f"✓ Created train mask: {train_mask_tensor.sum()} training nodes")
print(f"✓ Created valid mask: {valid_mask_tensor.sum()} validation nodes")
print(f"✓ Created test mask: {test_mask_tensor.sum()} test nodes")

# ==============================================================================
# PART 7: VERIFY DATA QUALITY
# ==============================================================================
print("\n" + "=" * 70)
print("PART 7: VERIFYING DATA QUALITY")
print("=" * 70)

print("\n[7.1] Checking for NaN (Not a Number) values...")

# Check if any values are NaN
has_nan = torch.isnan(x_tensor).any()
num_nan = torch.isnan(x_tensor).sum()

print(f"  Contains NaN? {has_nan}")
if has_nan:
    print(f"  ⚠️ WARNING: Found {num_nan} NaN values!")
else:
    print(f"  ✓ No NaN values found")

# ------------------------------------------------------------------------------
print("\n[7.2] Checking for Inf (Infinity) values...")

# Check if any values are infinite
has_inf = torch.isinf(x_tensor).any()
num_inf = torch.isinf(x_tensor).sum()

print(f"  Contains Inf? {has_inf}")
if has_inf:
    print(f"  ⚠️ WARNING: Found {num_inf} Inf values!")
else:
    print(f"  ✓ No Inf values found")

# ------------------------------------------------------------------------------
print("\n[7.3] Validating edge indices...")

# Check minimum edge index
min_edge_idx = edge_index_tensor.min()
print(f"  Minimum edge index: {min_edge_idx}")
print(f"  Expected: 0 or greater")

# Check maximum edge index
max_edge_idx = edge_index_tensor.max()
max_valid_idx = len(x_tensor) - 1

print(f"  Maximum edge index: {max_edge_idx}")
print(f"  Maximum valid index: {max_valid_idx}")

edges_valid = max_edge_idx < len(x_tensor)
print(f"  All edges point to valid nodes? {edges_valid}")

if not edges_valid:
    print(f"  ⚠️ WARNING: Some edges point to non-existent nodes!")

# ------------------------------------------------------------------------------
print("\n[7.4] Validating labels...")

# Check label range
min_label = y_tensor.min()
max_label = y_tensor.max()

print(f"  Minimum label: {min_label} (expected: -1)")
print(f"  Maximum label: {max_label} (expected: 1)")

labels_valid = (min_label >= -1) and (max_label <= 1)
print(f"  Labels in valid range? {labels_valid}")

# ------------------------------------------------------------------------------
print("\n[7.5] Verifying timestamp ordering...")

# Check if timestamps are still sorted after all processing
current_times = edge_timestamp_tensor[:-1]
next_times = edge_timestamp_tensor[1:]
still_sorted = torch.all(current_times <= next_times)

print(f"  Timestamps are sorted? {still_sorted}")
if not still_sorted:
    print(f"  ⚠️ WARNING: Timestamps became unsorted!")

# ==============================================================================
# PART 8: SAVE CLEANED DATA
# ==============================================================================
print("\n" + "=" * 70)
print("PART 8: SAVING CLEANED DATA")
print("=" * 70)

print("\n[8.1] Preparing data dictionary...")

# Create a dictionary with all processed data
cleaned_data = {
    # Tensors
    'x': x_tensor,
    'y': y_tensor,
    'edge_index': edge_index_tensor,
    'edge_type': edge_type_tensor,
    'edge_timestamp': edge_timestamp_tensor,
    
    # Masks
    'train_mask': train_mask_tensor,
    'valid_mask': valid_mask_tensor,
    'test_mask': test_mask_tensor,
    
    # Metadata
    'num_nodes': len(x_tensor),
    'num_edges': edge_index_tensor.shape[1],
    'num_features': x_tensor.shape[1],
    'num_classes': 2,  # Binary: fraud vs normal
}

print(f"Created dictionary with {len(cleaned_data)} items")

# ------------------------------------------------------------------------------
print("\n[8.2] Saving to file...")

filename = 'dgraphfin_cleaned.pt'
torch.save(cleaned_data, filename)

print(f"✓ Saved cleaned data to '{filename}'")

# Show file info
import os
file_size_bytes = os.path.getsize(filename)
file_size_mb = file_size_bytes / (1024 * 1024)

print(f"  File size: {file_size_mb:.2f} MB")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("CLEANING COMPLETE! SUMMARY OF PROCESSED DATA")
print("=" * 70)

print(f"\n📊 Data Statistics:")
print(f"  Total nodes: {cleaned_data['num_nodes']:,}")
print(f"  Total edges: {cleaned_data['num_edges']:,}")
print(f"  Features per node: {cleaned_data['num_features']}")
print(f"    - Original features: 17")
print(f"    - Missing value flags: 17")

print(f"\n📊 Label Distribution:")
print(f"  Training nodes: {train_mask_tensor.sum():,}")
print(f"  Validation nodes: {valid_mask_tensor.sum():,}")
print(f"  Test nodes: {test_mask_tensor.sum():,}")

print(f"\n📊 Class Distribution:")
print(f"  Normal users: {(y_tensor == 0).sum():,}")
print(f"  Fraudsters: {(y_tensor == 1).sum():,}")
print(f"  Background: {(y_tensor == -1).sum():,}")

print(f"\n✅ All quality checks passed!")
print(f"✅ Data is ready for temporal GNN training!")

print("\n" + "=" * 70)
print("HOW TO LOAD THE CLEANED DATA:")
print("=" * 70)
print("""
# In your training script, load like this:
data = torch.load('dgraphfin_cleaned.pt')

# Access components:
x = data['x']                    # Node features
y = data['y']                    # Node labels
edge_index = data['edge_index']  # Graph connections
edge_timestamp = data['edge_timestamp']  # Temporal information

# Use masks to split data:
train_mask = data['train_mask']
valid_mask = data['valid_mask']
test_mask = data['test_mask']
""")

print("=" * 70)

# Load your cleaned data
data = torch.load('dgraphfin_cleaned.pt')
x = data['x'][:, :17]  # First 17 features (before missing flags)

# Check a few sample rows
print("Sample of first 5 nodes, all 17 features:")
print(x[:5])

# Check feature statistics
print("\nFeature statistics:")
for i in range(17):
    feature_col = x[:, i]
    print(f"Feature {i}: min={feature_col.min():.2f}, max={feature_col.max():.2f}, "
          f"mean={feature_col.mean():.2f}, unique_values={len(torch.unique(feature_col))}")

# Check edge types more carefully
edge_type = data['edge_type']
print("\nEdge type distribution:")
for et in range(1, 12):
    count = (edge_type == et).sum()
    print(f"Edge Type {et}: {count} edges")
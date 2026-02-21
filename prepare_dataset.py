import os
import shutil
import pandas as pd

# CSV Paths
train_csv = "archive/Disease_Grading/Groundtruths/IDRiD_Disease Grading_Training Labels.csv"
test_csv = "archive/Disease_Grading/Groundtruths/IDRiD_Disease Grading_Testing Labels.csv"

# Image Paths
train_images_path = "archive/Disease_Grading/Original_Images/Training Set"
test_images_path = "archive/Disease_Grading/Original_Images/Testing Set"

# Output Dataset Paths
output_train = "dataset/train"
output_test = "dataset/test"

# Create class folders (0–4)
for i in range(5):
    os.makedirs(os.path.join(output_train, str(i)), exist_ok=True)
    os.makedirs(os.path.join(output_test, str(i)), exist_ok=True)

# -----------------------
# PROCESS TRAINING DATA
# -----------------------
train_df = pd.read_csv(train_csv)

for index, row in train_df.iterrows():
    image_name = row["Image name"] + ".jpg"
    label = str(row["Retinopathy grade"])

    src = os.path.join(train_images_path, image_name)
    dst = os.path.join(output_train, label, image_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

# -----------------------
# PROCESS TEST DATA
# -----------------------
test_df = pd.read_csv(test_csv)

for index, row in test_df.iterrows():
    image_name = row["Image name"] + ".jpg"
    label = str(row["Retinopathy grade"])

    src = os.path.join(test_images_path, image_name)
    dst = os.path.join(output_test, label, image_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("✅ Dataset prepared successfully!")
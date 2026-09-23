import os
import shutil
import kagglehub

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(current_dir, "..", "dataset"))

    os.makedirs(dataset_dir, exist_ok=True)

    print("Preuzimanje dataseta sa Kaggle-a...")
    download_path = kagglehub.dataset_download("vedikagupta0/youtube-top-100-songs-2025-dataset")
    print("Path to downloaded files:", download_path)

    for item in os.listdir(download_path):
        s = os.path.join(download_path, item)
        d = os.path.join(dataset_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print(f"💾 Svi fajlovi dataseta su uspešno premešteni u: {dataset_dir}")
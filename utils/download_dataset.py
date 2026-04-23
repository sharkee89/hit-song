import kagglehub

# Download latest version
path = kagglehub.dataset_download("asmonline/spotify-song-performance-dataset")

print("Path to dataset files:", path)
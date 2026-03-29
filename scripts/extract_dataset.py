import zipfile
import os
import shutil

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    
    if os.path.exists(extract_to):
        print(f"Cleaning existing directory: {extract_to}")
        shutil.rmtree(extract_to)
    
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files
        all_files = zip_ref.infolist()
        total_files = len(all_files)
        
        for i, file in enumerate(all_files):
            # Skip __MACOSX and hidden files
            if '__MACOSX' in file.filename or '/.' in file.filename or file.filename.startswith('.'):
                continue
            
            # Skip symlinks (external attributes bit 28-31)
            # On Windows, zipfile might extract symlinks as text files.
            # We want to ensure we extract actual files.
            
            try:
                zip_ref.extract(file, extract_to)
            except Exception as e:
                print(f"Error extracting {file.filename}: {e}")
            
            if i % 1000 == 0:
                print(f"Progress: {i}/{total_files} files processed...")

    print("Extraction complete.")

if __name__ == "__main__":
    zip_file = "archive.zip"
    target_dir = "data/raw"
    
    if os.path.exists(zip_file):
        extract_zip(zip_file, target_dir)
        
        # Verify a sample file
        # We need to find where it actually extracted to, 
        # usually zips have a top-level folder.
        print("\nVerifying extraction...")
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_path = os.path.join(root, f)
                    size = os.path.getsize(sample_path)
                    print(f"Sample file: {sample_path} (Size: {size / 1024:.2f} KB)")
    else:
        print(f"Error: {zip_file} not found.")

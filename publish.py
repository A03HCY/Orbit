import os
import sys
import subprocess
import glob
from builder import clean, build

def check_twine():
    '''Checks if twine is installed, installs if not.
    
    Verifies if 'twine' package is available in the current environment.
    If not found, attempts to install it using pip.
    '''
    print('Checking for twine...')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'show', 'twine'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('Twine is already installed.')
    except subprocess.CalledProcessError:
        print('Twine not found. Installing twine...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'twine'])
            print('Twine installed successfully.')
        except subprocess.CalledProcessError as e:
            print(f'Failed to install twine: {e}')
            sys.exit(1)

def upload():
    '''Uploads the built package to PyPI using twine.
    
    Finds all files in 'dist/' directory and uploads them using 'twine upload'.
    '''
    print('Uploading package to PyPI...')
    
    # Find the dist files
    dist_files = glob.glob('dist/*')
    if not dist_files:
        print('No distribution files found in dist/')
        sys.exit(1)
    
    # Run twine upload
    cmd = [sys.executable, '-m', 'twine', 'upload'] + dist_files
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print('Upload finished successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Upload failed: {e}')
        sys.exit(1)

def main():
    '''Main execution flow.'''
    try:
        clean()
        build()
        check_twine()
        upload()
        clean()
        print('All publish tasks completed successfully.')
    except KeyboardInterrupt:
        print('\nOperation cancelled by user.')
        sys.exit(1)
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()

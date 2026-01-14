import os
import shutil
import subprocess
import glob
import sys

def clean():
    '''Cleans up build artifacts.

    Removes 'build', 'dist' directories and any '*.egg-info' directories found
    in the current working directory.
    '''
    print('Cleaning up...')
    dirs_to_remove = ['build', 'dist']
    # Find egg-info directories
    dirs_to_remove.extend(glob.glob('*.egg-info'))

    for d in dirs_to_remove:
        if os.path.exists(d):
            print(f'Removing {d}...')
            shutil.rmtree(d)
    print('Clean finished.')

def build():
    '''Builds the package using setup.py.

    Runs 'python setup.py sdist bdist_wheel' to generate source distribution
    and wheel package.
    '''
    print('Building package...')
    subprocess.check_call([sys.executable, 'setup.py', 'sdist', 'bdist_wheel'])
    print('Build finished.')

def install():
    '''Installs the built package.

    Uninstalls the existing 'orbit-torch' package and installs the newly built
    wheel file from the 'dist' directory.
    '''
    print('Installing package...')
    
    # Uninstall existing package
    # We use call instead of check_call because uninstall might fail if package is not installed,
    # which is fine.
    subprocess.call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'orbit-torch'])

    # Find the wheel file
    whl_files = glob.glob('dist/*.whl')
    if not whl_files:
        print('No wheel file found in dist/')
        return
    
    # Install the new wheel
    # Pick the latest one just in case, though clean() should ensure only one exists
    latest_whl = max(whl_files, key=os.path.getctime)
    print(f'Installing {latest_whl}...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', latest_whl])
    print('Install finished.')

def main():
    '''Main execution flow.'''
    try:
        clean()
        build()
        install()
        clean()
        print('All tasks completed successfully.')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()

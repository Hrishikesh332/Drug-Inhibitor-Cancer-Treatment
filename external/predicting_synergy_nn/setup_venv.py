import os
import subprocess
import sys
import platform

def run_command(command):

    print(f"Running: {command}")
    try:
        process = subprocess.run(command, shell=True, check=True)
        print(f"Command completed with exit code {process.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return False

def main():

    python_exe = sys.executable
    print(f"Using Python: {python_exe}")

    venv_name = "synergy_env"
    
    if os.path.exists(venv_name):
        print(f"Warning: Virtual environment '{venv_name}' already exists.")
        response = input("Do you want to remove and recreate it? (y/n): ")
        if response.lower() == 'y':
            print(f"Removing existing environment...")
            import shutil
            shutil.rmtree(venv_name)
        else:
            print("Using existing environment.")
            
    if not os.path.exists(venv_name):
        print(f"Creating virtual environment '{venv_name}'...")
        if not run_command(f"{python_exe} -m venv {venv_name}"):
            print("Failed to create virtual environment. Exiting.")
            return
    
    if platform.system() == "Windows":
        activate_script = f"{venv_name}\\Scripts\\activate"
        pip_command = f"{venv_name}\\Scripts\\pip"
    else:
        activate_script = f"source {venv_name}/bin/activate"
        pip_command = f"{venv_name}/bin/pip"
    
    print("Upgrading pip...")
    if platform.system() == "Windows":
        if not run_command(f"{pip_command} install --upgrade pip"):
            print("Warning: Failed to upgrade pip.")
    else:
        if not run_command(f"{activate_script} && pip install --upgrade pip"):
            print("Warning: Failed to upgrade pip.")

    print("Installing requirements...")
    if platform.system() == "Windows":
        if not run_command(f"{pip_command} install -r requirements.txt"):
            print("Failed to install requirements. Exiting.")
            return
    else:
        if not run_command(f"{activate_script} && pip install -r requirements.txt"):
            print("Failed to install requirements. Exiting.")
            return
    
    print("\n" + "="*60)
    print("Virtual environment setup completed successfully!")
    print("="*60)
    print("\nTo activate the virtual environment, run:")
    if platform.system() == "Windows":
        print(f"{venv_name}\\Scripts\\activate")
    else:
        print(f"source {venv_name}/bin/activate")
    
    print("\nTo deactivate when you're done, simply run:")
    print("deactivate")

if __name__ == "__main__":
    main()
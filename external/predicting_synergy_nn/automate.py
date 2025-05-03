import os
import subprocess
import sys
import venv


project_dir = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(project_dir, 'venv')
requirements_file = os.path.join(project_dir, 'requirements.txt')


if not os.path.exists(venv_dir):
    print(f"Creating virtual environment at {venv_dir}...")
    venv.create(venv_dir, with_pip=True)

python_executable = os.path.join(venv_dir, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_dir, 'bin', 'python')

print("Upgrading pip...")
subprocess.check_call([python_executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

print("Installing required packages...")
subprocess.check_call([python_executable, '-m', 'pip', 'install', '-r', requirements_file])

training_scripts = [
    os.path.join(project_dir, 'train_final_models', 'train_fold1', 'train_model1.py'),
    os.path.join(project_dir, 'train_final_models', 'train_fold2', 'train_model2.py'),
    os.path.join(project_dir, 'train_final_models', 'train_fold3', 'train_model3.py'),
]

for script in training_scripts:
    script_dir = os.path.dirname(script)
    print(f"Running {script} in {script_dir}...")
    subprocess.check_call([python_executable, os.path.basename(script)], cwd=script_dir)

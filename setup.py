import subprocess

def SettingEnvironment():
    #set_up_for_DEEPHOUGH
    directory_path = 'DeepHough/model/_cdht/'
    subprocess.call(["python", "setup.py", "install"], cwd=directory_path)

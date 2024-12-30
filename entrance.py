import inquirer
import logging, functools
import yaml
import requests
from git import repo
import os
import subprocess
import time

with open("./settings.yaml") as f:
    settings = yaml.safe_load(f)
with open("./secret.yaml") as f:
    secret_dict = yaml.safe_load(f)
settings["github_token"]=secret_dict["github_token"]
log_path = "./log/info.log"
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))
logging.basicConfig(level = logging.INFO,filename=log_path,filemode="a",format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"running {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} DONE.")
            return result
        except Exception as e:
            logger.exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
            raise e
    return wrapper
def log_print(message:str,level=20): 
    """level=20(info) by default"""
    logger.log(level=level,msg=message)
    print(message)

@log
def main():
    # update_check()
    login()
    # choose mode
    mode = inquirer.list_input("Which service would you start?",
        choices=[("Run OurBOT",0),("Fine-tune OurBOT",1),("Run webui",2),("exit",-1)])
    if mode==0:
        # start a Bot server
        start_server()
    elif mode==1:
        # start fine-tune
        pass
    elif mode==2:
        # start webui locally
        settingfile = "config/qwen25_lora_run.yaml"
        log_print(f"Running webui with {settingfile}")
        runner = subprocess.Popen(
            ["llamafactory-cli","webchat", settingfile]
        )
        time.sleep(5)
        while inquirer.text("Enter 'exit' to stop server")!="exit":
            pass
        runner.terminate()
        log_print("Local webui terminated.")
    elif mode==3:
        # evaluate
        pass
    elif mode==-1:
        return 0

@log
def update_check():
    print("Connecting to Github...")
    url = "https://api.github.com/repos/PandragonXIII/DistributedChatbot/releases/latest"
    response = requests.get(url,headers={"Authorization":settings["github_token"]})
    if response.status_code == 200:
        release_info = response.json()
        latest_release = release_info["tag_name"]
        this_repo = repo.Repo()
        present_release = this_repo.tags[-1].name
        if latest_release!=present_release:
            # require an update
            if inquirer.confirm(f"Found latest release: {latest_release}, update now?",default=False): 
                # perform auto update
                # TODO
                logger.warning("auto update not implemented now.")
                pass
        else:
            msg = f"Present version {present_release}, Everything up-to-date."
            log_print(msg)
            return 0
    else:
        logger.warning(f"GitHub response code: {response.status_code}\nskipping update process.")
        print(f"Connection error, skipping update process...")
        return -1
@log
def login():
    print("User system not ready yet.")
    logger.info("skip login")
    pass
@log
def fine_tune():
    # check existing task.

    # search checkpoint

    # start fine-tune

    # save checkpoint path to task yaml

    # merge weight?

    # upload weight

    pass
@log
def start_server():
    raise NotImplementedError

def install_OurBOT():
    pass

if __name__=="__main__":
    ret = main()
    logger.info(f"Programme exited with code: {ret}")
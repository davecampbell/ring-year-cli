import warnings

# this suppresses from warnings about torchvision
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from fastai.vision.all import *
    import torch

import prettyprinter as pp
import time
import argparse

import os
import subprocess

# set up the expected command line arguments
parser = argparse.ArgumentParser(description='Predict/Recognize the year on a class ring image.')
parser.add_argument(
    "-i", "--img_path",
    type=str,
    help='The path to the image to be predicted.'
)

parser.add_argument(
    "-m", "--mode",
    type=str,
    help='The mode of running this program. Options: "random" or "look".'
)

# parse the command line
args = parser.parse_args()
# get a count of the number of non-None arguments
num_args = sum(
        1 for arg in vars(args).values() if arg not in (None, False)
    )

# function to pick a random file from a folder
# if there is only one, it will return it
def pick_random_file(folder_path):
    try:
        # Get a list of all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if not files:
            # print("No files found in the folder.")
            return None
        # Pick a random file
        random_file = random.choice(files)
        return random_file
    except FileNotFoundError:
        print("The specified folder does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_file_flag(file_path: str):
    """
    Reads a flag in a file.

    :param file_path: Path to the file containing the flag.
    :return: value of the flag.
    """
    with open(file_path, 'r') as file:
        for line in file:
            return line.strip()

def is_file_open(file_path):
    """
    Check if a file is open by any process using lsof.
    :param file_path: Path to the file
    :return: True if the file is open, False otherwise
    """
    try:
        result = subprocess.run(['lsof', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0  # If lsof returns 0, the file is open
    except Exception as e:
        # print(f"Error checking file with lsof: {e}")
        return False

def get_top_classes_by_confidence(df, top_x):
    """
    Returns a dictionary of the top `top_x` classes by confidence from a DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame with 'class' (str) and 'confidence' (float) columns.
    - top_x (int): The number of top classes to return.

    Returns:
    - dict: A dictionary with class names as keys and confidence as values, sorted by confidence in descending order.
    """
    if 'class' not in df.columns or 'confidence' not in df.columns:
        raise ValueError("DataFrame must have 'class' and 'confidence' columns.")
    
    top_classes = df.nlargest(top_x, 'confidence')
    return dict(zip(top_classes['class'], top_classes['confidence']))


# this is a custom function that was used in the datablock of the learner
# need to define it here so it can be monkey-patched into the model after
# the exported version
def custom_labeller(fname):
    # Assuming filenames are like "classX_imageY.jpg"
    # return fname.name.split('_')[0]  # Extract class name based on your logic
    pattern = '(?<=\d{2})\d{2}(?=_)'
    digits = re.search(pattern, str(fname)).group()
    klasses = list(digits)
    klasses.append(digits)
    return klasses

# the monkey-patch
globals()['custom_labeller'] = custom_labeller

# the exported model location
model_path = 'models/model-single-both.pkl'

# Load the model
learner = load_learner(model_path)

# folder locations
folder_path = 'images/ring-scans/'
look_folder_path = 'images/look/'
output_path = 'output/ring_pred.json'
flag_file_path = 'output/flag.txt'

while True:

    while read_file_flag(flag_file_path) == 'GO':

        # response output
        output = {}

        # determine which image to predict on 
        if num_args == 0 or args.mode == 'random':
            folder_path = 'images/ring-scans/'
            img = pick_random_file(folder_path)
            img_path = folder_path + img

        elif args.mode == 'look':
            folder_path = 'images/look/'
            img = pick_random_file(folder_path)
            while img is None:
                # print(f"no image in {folder_path} - sleeping 5 seconds...")
                # time.sleep(5)
                img = pick_random_file(folder_path)
            img_path = folder_path + img

        elif len(args.img_path) > 0:
            img_path = args.img_path

        else:
            print("confused state")
            break

        # print("predicting: " + img_path)
        output["img_path"] = img_path

        # predict classes for the image, suppressing progress output
        with learner.no_bar():
            x = learner.predict(img_path)

        # put results into a dataframe to do better manipulation

        # confidence numbers from tensors into a list
        conf = [tensor.item() for tensor in x[2]]
        # make the dataframe with the classes and confidences
        df = pd.DataFrame(zip(learner.dls.vocab, conf), columns=['class', 'confidence'])
        # filter for 2-digit classes
        two_digit_classes = df[df['class'].str.match(r'^\d{2}$')]
        # Find the class with the highest confidence and return just the string
        highest_conf_class = two_digit_classes.loc[two_digit_classes['confidence'].idxmax(), 'class']

        # get some of the top classes to evaluate

        top_x = 5
        top_all = get_top_classes_by_confidence(df, top_x)

        top_x = 3
        top_2_digit = get_top_classes_by_confidence(two_digit_classes, top_x)

        # output that one class
        # print(highest_conf_class)

        output["class"] = highest_conf_class

        # diagnostic stuff if needed

        # output the top classes
        # pp.pprint(top_all)
        output["top_all_classes"] = top_all

        # pp.pprint(top_2_digit)
        output["top_2_digit_classes"] = top_2_digit

        # print(output)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with open(output_path, "w") as file:
            json.dump(output, file, indent=4)  # `indent` makes the JSON more readable

import os
import argparse

import tkinter
from tkinter import filedialog


def ask_user_specify_checkpoints_location(gui: bool = True):
    # Asks the user (via command line or gui) to select one random model checkpoint,
    # and writes the folder of that checkpoint into the
    # ./../checkpoints/checkpoints_location.txt

    # Get location of file path holder file
    this_file_folder = os.path.dirname(os.path.abspath(__file__))
    checkpoints_location_file = os.path.join(this_file_folder, "..", "checkpoints", "checkpoints_location.txt")
    # Ask directory
    if gui:
        root = tkinter.Tk()
        root.withdraw()
        pth_file = filedialog.askopenfilename(
            initialdir=os.path.dirname(checkpoints_location_file),
            title="Select a checkpoint (.pth file)",
            filetypes=[("Checkpoint files", "*.pth", "*.pt"), ("All files", "*.*")]
        )
        if pth_file:
            checkpoint_folder = os.path.dirname(pth_file)
        else:
            print("No checkpoint file selected.")
            return
    else:
        checkpoint_folder = input("Please enter the full path to the folder containing the DINOv3 model checkpoints (.pth files): ").strip()
        checkpoint_folder = checkpoint_folder.replace('"', '').replace("'", "")
    # Now that we have a folder, we normalize the path
    checkpoint_folder = os.path.normpath(checkpoint_folder)
    # test this location
    pth_files_list = [f for f in os.listdir(checkpoint_folder) if f.endswith(".pth")]
    if len(pth_files_list) == 0:
        print(f"No .pth files found in the specified checkpoint folder:\n\t{checkpoint_folder}")
        return
    # write it down
    with open(checkpoints_location_file, "w") as f:
        f.write(str(checkpoint_folder))
    # Confirm success
    print("Success!!")
    print(f"Checkpoint folder set to:\n\t{checkpoint_folder}\nContaining {len(pth_files_list)} .pth files.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Set the location of the DINOv3 model checkpoints.")
    parser.add_argument("--no-gui", action="store_true", help="Use command line input instead of GUI to specify the checkpoint folder.")
    args = parser.parse_args()

    ask_user_specify_checkpoints_location(gui=not args.no_gui)
import os
import shutil
import logging


def create_directory(directory):
    # Check if the directory already exists
    if os.path.exists(directory):
        # Prompt the user for a decision
        response = (
            input(
                f"The directory '{directory}' already exists. Do you want to delete it? (y/n): "
            )
            .strip()
            .lower()
        )
        if response == "y":
            # Delete the directory if the user confirms
            shutil.rmtree(directory)  # This removes an empty directory
            logging.info(f"Deleted the directory: {directory}")
            # Re-create the directory
            os.makedirs(directory)
            logging.info(f"Created the directory: {directory}")
        elif response == "n":
            logging.info("Continuing without deleting the directory.")
        else:
            logging.info("Invalid input. Exiting.")
            return
    else:
        # Create the directory if it does not exist
        os.makedirs(directory)
        logging.info(f"Created the directory: {directory}")
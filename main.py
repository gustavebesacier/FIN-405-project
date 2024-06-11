from trash.Data_handler import get_data

import os

def main():
    # Download the data
    print(get_data())

    # Prepare the folder for the figures
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

if __name__ == "__main__":
    main()
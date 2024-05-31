from Data_handler import get_data

import os

def main():
    # print(get_data())
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

if __name__ == "__main__":
    main()
import os
import webbrowser

BOX_FOLDER = "https://berkeley.box.com/s/6l0lr3qfuz3i67act3kphquf21c2zm0m"

def main():
    print("\n📦 Box does not support direct file downloads from shared folders.")
    print("Please download the following files manually from the browser and move the files into the empty data folder:")
    print("  - cook_county_train.csv")
    print("  - cook_county_train_val.csv")
    print("  - cook_county_test.csv")
    print("  - cook_county_contest_test.csv")
    print("  - codebook.txt")
    print("\nOpening Box folder in your browser now...\n")
    webbrowser.open(BOX_FOLDER)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()

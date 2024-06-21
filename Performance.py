import re
import pandas as pd

def load_factors(data):
    #We need to load the data from the 12 industry portfolio return and the Fama/French 5 factors
    famafrench = pd.read_csv("Data/F-F_Research_Data_5_Factors_2x3.CSV", skiprows=3)
    #Industry_weights_12 = pd.read_csv("Data/12_Industry_Portfolios.CSV", skiprows=11)
    # Load the entire CSV file
    with open("Data/12_Industry_Portfolios.CSV", 'r') as file:
        lines = file.readlines()

    # Find the start and end of the first table
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if 'Average Value Weighted Returns -- Monthly' in line:
            start_index = i + 1  # The data starts after this line
        elif start_index is not None and line.strip() == '':
            end_index = i
            break

    # Extract the lines for the first table
    table_lines = lines[start_index:end_index]

    # Create a new DataFrame from the extracted lines
    Industry_weights_12 = pd.DataFrame([x.strip().split(',') for x in table_lines[1:]], columns=table_lines[0].strip().split(','))

    # Convert appropriate columns to numeric
    Industry_weights_12 = Industry_weights_12.apply(pd.to_numeric, errors='ignore')
    with open("Data/Siccodes12.txt", 'r') as file:
        Industries = file.read()

    # Regular expression to match each class and its indices
    pattern = r'(\d+)\s+(\w+)\s+([^\n]+)\n((?:\s+\d+-\d+\n?)+)'

    # Find all matches in the content
    matches = re.findall(pattern, Industries)

    # Initialize an empty dictionary to hold the classes and their indices
    classes = {}

    # Iterate over each match and populate the dictionary
    for match in matches:
        class_number = int(match[0])
        class_code = match[1]
        class_description = match[2].strip()
        indices_raw = match[3].strip().split('\n')

        # Extract index ranges and convert them to tuples of integers
        indices = []
        for index_range in indices_raw:
            start, end = map(int, index_range.strip().split('-'))
            indices.append((start, end))

        # Add the class to the dictionary
        classes[class_number] = {
            'code': class_code,
            'description': class_description,
            'indices': indices
        }

    # Display the classes dictionary
    for class_number, class_info in classes.items():
        print(f"Class {class_number} ({class_info['code']}): {class_info['description']}")
        for index_range in class_info['indices']:
            print(f"  Index range: {index_range[0]}-{index_range[1]}")

    def get_class_for_siccd(siccd):
        for class_number, class_info in classes.items():
            for index_range in class_info['indices']:
                if index_range[0] <= int(siccd) <= index_range[1]:
                    return class_number
        return 12

    # Apply our rule to find classes
    data['class'] = data['siccd'].apply(get_class_for_siccd)

    print(data.head())

    return famafrench, Industry_weights_12, data
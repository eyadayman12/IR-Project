import re


list_of_documents = []
counter = 1


def take_filepath_input():
    global counter
    filepath = input(f"Enter file path number {counter}: ")
    return filepath

def read_file():
    filepath = take_filepath_input()
    if filepath == re.findall('[A-z|0-9]*.txt', filepath)[0]:
        with open(filepath) as file:
            lines = file.read()
            list_of_documents.append(lines)
    return filepath

        
def error_message():
    print("\nThe file could not be read, try again")
    print("Make sure that the file path is correct, make sure that the file extension is written and make sure that it is a .txt file\n")

    
    
def read10documents():
    global counter
    while counter<=10:
        try:
            filepath = read_file()
            counter+=1
        except:
            error_message()
    return list_of_documents
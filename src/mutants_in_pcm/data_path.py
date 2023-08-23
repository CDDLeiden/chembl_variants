import os

def get_data_path():
    data_dir = 'C:\\Users\\gorostiolam\\Documents\\Gorostiola Gonzalez, ' \
           'Marina\\PROJECTS\\6_Mutants_PCM\\PROTOCOLS-SCRIPTS\\Python\\mutants_in_pcm\\data'
    data_dir = os.path.abspath(data_dir)
    if os.path.exists(data_dir):
        return data_dir
    else:
        print('The data directory does not exist. Please check the path in data_path.py')

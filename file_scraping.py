import os
import glob
import shutil as sh
import pandas as pd


def find_files(pattern, search_path):
    """
    pattern: [str] to be specified as pattern already (e.g. *individual*")
    search path: [str] path where to search with "\\" at the end

    path subdirectory delimiter "\\" Windows or "/" Linux/Mac
    """

    # find file -> list of paths
    file_list = glob.glob(search_path+pattern)

    if len(file_list) == 0:
        print("no file names match the pattern")
        return []
    else:
        return file_list


def copy_files(original_path, target_path):
    """
    original_path: [str] complete path to file to copy (including filename.extension)
    target_path: [str] complete path to copy the file to (including filename.extension)

    path subdirectory delimiter "\\" Windows or "/" Linux/Mac
    """

    subdir = "\\".join(target_path.split("\\")[:-1])

    # creates the new subdirectory (+ intermediates) if it does not exist already
    try:
        os.makedirs(subdir)
    except FileExistsError:
        pass

    sh.copyfile(original_path, target_path)


def individuals_list(path_to_data):
    """
    path_to_data: [str] path to the file containing the individual codes (must be .csv)

    path subdirectory delimiter "\\" Windows or "/" Linux/Mac
    """

    ind_data = pd.read_csv(path_to_data)

    # if more than just one col containing the codes must specify col num.
    # creates a set in case of duplicates
    return set(["*"+code+"*" for code in ind_data.iloc[:, 0].tolist()])


if __name__ == "__main__":
    # selection of paths to search in (** := search in all subdirectories)
    paths_list = [

        "\\\\ieu-fsgr.d.uzh.ch\Manser\Shared_Callfiles\meerkats\\y2Annotations_General_combined_converted\\converted_txt_to_audition\\"
    ]

    # individuals' codes list
    ind_list = individuals_list("selection_dfs\\selected_individuals.csv")

    files = []

    for path in paths_list:

        for ind_name in ind_list:

            # gathers all files in one list
            files = files + find_files(pattern=ind_name, search_path=path)

    print("FINISHED SEARCHING\nBEGIN COPYING")

    for name in files:

        # creeates target path (first part is own path, second is part of the original)
        # in this case I keep from \\CC\\ onwards for the original path
        target = "C:\\Users\\adelu\\Documents\\UZH\\Thesis\\Shared_CallFiles\\" + \
            "\\".join(name.split("\\")[6:])

        copy_files(original_path=name, target_path=target)

        print("copied successfully")

    print("DONE")

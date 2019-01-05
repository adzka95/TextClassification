from pathlib import Path


def find_categories(filename, path):
    text = Path(path + "/" + filename).read_text()
    res = [int(i) for i in text.split()]
    return find_categories_with_array(res)


def find_categories_with_array(array):
    current_text_categories = []
    file = open("finalCategories.txt", "r")
    all_categories = [i for i in file.read().split()]
    for idx, value in enumerate(array):
        if value == 1:
            current_text_categories.append(all_categories[idx])

    return current_text_categories

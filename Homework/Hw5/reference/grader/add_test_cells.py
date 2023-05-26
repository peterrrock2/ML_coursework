import json
import os

submissions_folder = "/repos/CSX622_F21/grader/PS5/submissions"
modified_folder = "/repos/CSX622_F21/grader/PS5/submissions_mod"


def merge_lists(list1, list2, pos):
    return list1[:pos] + list2 + list1[pos:]

def add_cells(to_add, cell_id, submissions_folder, modified_folder):
    to_add = json.load(open(to_add, 'r'))

    for file in filter(lambda x: x[-5:] == "ipynb", os.listdir(submissions_folder)):
        print(file)
        with open(os.path.join(submissions_folder, file),'r') as fopen:
            notebook = json.load(fopen)
            fopen.close()
            pos = 0
            for cell in notebook['cells']:
                meta = cell['metadata']
                if "nbgrader" in meta:
                    if meta["nbgrader"]['grade_id'] == cell_id:
                        break
                pos += 1
            notebook["cells"] = merge_lists(notebook['cells'], to_add["cells"], pos)

        with open(os.path.join(modified_folder, file), 'w') as fopen:
            json.dump(notebook, fopen)
            fopen.close()

add_cells("/repos/CSX622_F21/grader/PS5/PS5_1_tests.ipynb", "q1_8",submissions_folder, modified_folder)

add_cells("/repos/CSX622_F21/grader/PS5/PS5_2_tests.ipynb", "deep_intro", modified_folder, modified_folder)
add_cells("/repos/CSX622_F21/grader/PS5/save_results.ipynb", "deep_intro", modified_folder, modified_folder)


#
#
#
# old = list(filter(lambda x: x[-5:] == "ipynb", os.listdir("/repos/CSX622_F21/grader/PS5/submissions")))
# new = list(filter(lambda x: x[-5:] == "ipynb", os.listdir("/repos/CSX622_F21/grader/PS5/submissions(1)")))
# for n in new:
#     if n not in old:
#         print(n)
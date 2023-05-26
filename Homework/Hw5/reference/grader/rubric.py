

class Rubric(object):

    def __init__(self, assignment_name):
        self.name = assignment_name
        self.questions = {}
        self.tests = {}

    def add_question(self, question, max_points):
        self.questions[question] = {"max": max_points, "tests": []}

    def add_test(self, question, test_function, grade):
        if question not in self.questions:
            raise AssertionError("Question $s is not in questions list, add it first" % question)
        self.questions[question]["tests"].append((test_function, grade))

    def sanity_check(self):
        for question in self.questions:
            graded_total = sum([grade for test, grade in self.questions[question]["tests"]])
            if graded_total != self.questions[question]["max"]:
                raise AssertionError("Question %s: Mismatch between tests total %f and max grade %f" %
                                     (question, graded_total, self.questions[question]["max"]))
        return True

    def run(self):
        self.sanity_check()
        grades = {}
        for question in self.questions:
            grades[question] = {"max": self.questions[question]["max"], "obtained": 0, "comments": []}
            for test, grade in self.questions[question]["tests"]:
                success, comment = test()
                passed = "Fail"
                if success:
                    grades[question]["obtained"] += grade
                    passed = "Pass"
                    comment = str(grade) + " pts. " + comment
                else:
                    comment = "-%i pts. " % grade + comment
                    grades[question]["comments"].append(comment + ":[%s]" % passed)

        self.show(grades)

    @staticmethod
    def show(grades):
        for question in grades:
            print(question, ":",grades[question]["obtained"],"/",grades[question]["max"])
            for comment in grades[question]["comments"]:
                print("\t", comment)

    @staticmethod
    def get_values(grades):
        grades_dict = {}
        comments = ""
        for question in grades:
            grades_dict[question] = grades[question]["obtained"]
            comments = comments + question + ":" + str(grades[question]["obtained"])+ "/" + str(grades[question]["max"])
            comments = comments + "\n"
            if len(grades[question]["comments"]) > 0:
                for comment in grades[question]["comments"]:
                    comments += "\t" + comment + "\n"

        return grades_dict, comments

    def run_and_save(self, notebook, df):
        self.sanity_check()
        grades = {}
        for question in self.questions:
            grades[question] = {"max": self.questions[question]["max"], "obtained": 0, "comments": []}
            for test, grade in self.questions[question]["tests"]:
                success, comment = test()
                passed = "Fail"
                if success:
                    grades[question]["obtained"] += grade
                    passed = "Pass"
                    comment = str(grade) + " pts. " + comment
                else:
                    comment = "-%i pts. " % grade + comment
                    grades[question]["comments"].append(comment + ":[%s]" % passed)

        grades_dict, comments = self.get_values(grades)
        for index, row in df.iterrows():
            if row["notebook"] == notebook:
                for k in grades_dict:
                    df.loc[index, k] = int(grades_dict[k])
                #df.loc[index, "comments"] = comments
        return grades_dict, comments

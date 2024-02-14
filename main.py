from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC


from SmellDetector import SmellDetector


def run_cli(datasets, classifiers):
    print("Choose smell dataset")
    print("1: data class")
    print("2: god class")
    print("3: long method")
    print("4: feature envy")
    selected_dataset = int(input())
    if 1 <= selected_dataset <= 4:
        dataset_name = datasets[selected_dataset - 1]
        print(f"dataset:{dataset_name}")

        print('Choose your classifier:')
        for index in range(len(classifiers)):
            print(f"{index + 1}: {classifiers[index][0].__class__.__name__}")
        selected_classifier = int(input())
        if 1 <= selected_classifier <= len(classifiers) + 1:
            (model, param_grid) = classifiers[selected_classifier - 1]
            scaling = str(model.__class__.__name__).startswith('SVC')
            smell_detector = SmellDetector(dataset_name, model, param_grid=param_grid, scaling=scaling)
            commands = [smell_detector.train_classifier, smell_detector.test_classifier,
                        smell_detector.get_global_importance, 'exit']
            exiting = False
            while not exiting:
                print("1: train a model")
                print("2: test  model")
                print("3: explain model")
                print("4: exit")
                selected_command_index = int(input())
                if 1 <= selected_command_index <= len(commands):
                    selected_command = commands[selected_command_index - 1]
                    if selected_command == 'exit':
                        exiting = True
                        continue
                    selected_command()
                    # smell_detector.get_global_importance()
                else:
                    print("invalid input")
        else:
            print("invalid input")
    else:
        print("invalid input")


if __name__ == '__main__':
    datasets = ['data-class', 'god-class', 'long-method', 'feature-envy']
    classifiers = [(DecisionTreeClassifier(), {'criterion': ['log_loss', 'gini'],
                                               'max_depth': list(range(1, 101)) + [None]}),
                   (RandomForestClassifier(),
                    {'criterion': ['log_loss', 'gini'], 'max_depth': list(range(1, 101, 10)),
                     'n_estimators': list(range(1, 200, 20))}),
                   (HistGradientBoostingClassifier(), {
                       'max_depth': list(range(1, 101, 10)), 'max_iter': list(range(1, 200, 20))
                   }),
                   (SVC(kernel='rbf'), {
                       'C': [float(x) for x in range(200, 2001, 200)],
                       'gamma': [float(i) for i in range(0, 100, 10)] + ['scale', 'auto']
                   })]

    run_cli(datasets, classifiers)

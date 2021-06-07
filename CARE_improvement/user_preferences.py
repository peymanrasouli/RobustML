from utils import *

def userPreferences(dataset, x_ord):

    x_org = ord2org(x_ord, dataset)

    print('\n')
    print('----- possible values -----')
    for f_val in dataset['feature_values']:
        print(f_val)

    print('\n')
    print('----- instance values -----')
    for i, f in enumerate(dataset['feature_names']):
        print(f+':', x_org[i])

    ## discrete constraints = {'fix', {v1, v2, v3, ...}}
    ## continuous constraints = {'fix', 'l', 'g', 'le', 'ge', [lb, ub]}
    ## constraints = {feature_name_1: (constraint, importance), feature_name_2: (constraint, importance), ...}

    ## Adult data set
    if dataset['name'] == 'adult':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., age) and features with set {} values (e.g., work-class) are discrete

        # {'age': [17, 90]}
        # {'fnlwgt': [13769, 1484705]}
        # {'capital-gain': [0, 99999]}
        # {'capital-loss': [0, 4356]}
        # {'hours-per-week': [1, 99]}
        # {'work-class': {' Self-emp-not-inc', ' Local-gov', ' State-gov', ' Self-emp-inc', ' Without-pay',
        #                 ' Federal-gov', ' Private'}}
        # {'education-num': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
        # {'education': {' 12th', ' Some-college', ' Assoc-acdm', ' Doctorate', ' Assoc-voc', ' Prof-school',
        #                ' Preschool', ' HS-grad', ' 7th-8th', ' 1st-4th', ' 9th', ' 10th', ' 11th', ' Masters',
        #                ' Bachelors', ' 5th-6th'}}
        # {'marital-status': {' Married-AF-spouse', ' Divorced', ' Married-civ-spouse', ' Married-spouse-absent',
        #                     ' Widowed', ' Never-married', ' Separated'}}
        # {'occupation': {' Farming-fishing', ' Handlers-cleaners', ' Craft-repair', ' Machine-op-inspct',
        #                 ' Transport-moving', ' Other-service', ' Prof-specialty', ' Armed-Forces', ' Priv-house-serv',
        #                 ' Adm-clerical', ' Exec-managerial', ' Sales', ' Tech-support', ' Protective-serv'}}
        # {'relationship': {' Husband', ' Not-in-family', ' Own-child', ' Other-relative', ' Wife', ' Unmarried'}}
        # {'race': {' Asian-Pac-Islander', ' Other', ' White', ' Amer-Indian-Eskimo', ' Black'}}
        # {'sex': {' Male', ' Female'}}
        # {'native-country': {' France', ' Japan', ' Iran', ' Taiwan', ' Hungary', ' Trinadad&Tobago',
        #                     ' Holand-Netherlands', ' Honduras', ' Outlying-US(Guam-USVI-etc)', ' Vietnam', ' Canada',
        #                     ' Italy', ' South', ' Jamaica', ' Mexico', ' Philippines', ' Ecuador', ' Greece',
        #                     ' Nicaragua', ' Portugal', ' Columbia', ' Scotland', ' Yugoslavia', ' Dominican-Republic',
        #                     ' England', ' Guatemala', ' United-States', ' Peru', ' Laos', ' Germany', ' Hong',
        #                     ' El-Salvador', ' Ireland', ' Haiti', ' India', ' Poland', ' Cambodia', ' Puerto-Rico',
        #                     ' Thailand', ' Cuba', ' China'}}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'age': ('ge',1),
                   'sex': ('fix', 1),
                   'race': ('fix', 1),
                   'native-country': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## Credit card default data set
    elif dataset['name'] == 'credit-card-default':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., LIMIT_BAL) and features with set {} values (e.g., SEX) are discrete

        # {'LIMIT_BAL': [10000, 1000000]}
        # {'AGE': [21, 79]}
        # {'BILL_AMT1': [-165580, 964511]}
        # {'BILL_AMT2': [-69777, 983931]}
        # {'BILL_AMT3': [-157264, 1664089]}
        # {'BILL_AMT4': [-170000, 891586]}
        # {'BILL_AMT5': [-81334, 927171]}
        # {'BILL_AMT6': [-339603, 961664]}
        # {'PAY_AMT1': [0, 873552]}
        # {'PAY_AMT2': [0, 1684259]}
        # {'PAY_AMT3': [0, 896040]}
        # {'PAY_AMT4': [0, 621000]}
        # {'PAY_AMT5': [0, 426529]}
        # {'PAY_AMT6': [0, 528666]}
        # {'SEX': {1, 2}}
        # {'EDUCATION': {0, 1, 2, 3, 4, 5, 6}}
        # {'MARRIAGE': {0, 1, 2, 3}}
        # {'PAY_0': {0, 1, 2, 3, 4, 5, 6, 7, 8, -2, -1}}
        # {'PAY_2': {0, 1, 2, 3, 4, 5, 6, 7, 8, -2, -1}}
        # {'PAY_3': {0, 1, 2, 3, 4, 5, 6, 7, 8, -2, -1}}
        # {'PAY_4': {0, 1, 2, 3, 4, 5, 6, 7, 8, -2, -1}}
        # {'PAY_5': {0, 2, 3, 4, 5, 6, 7, 8, -1, -2}}
        # {'PAY_6': {0, 2, 3, 4, 5, 6, 7, 8, -1, -2}}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'AGE': ('ge', 1),
                   'SEX': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## Heart disease data set
    elif dataset['name'] == 'heart-disease':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., age) and features with set {} values (e.g., sex) are discrete

        # {'age': [29, 77]},
        # {'trestbps': [94, 200]},
        # {'chol': [126, 564]},
        # {'thalach': [71, 202]},
        # {'oldpeak': [0.0, 6.2]},
        # {'sex': {0, 1}},
        # {'cp': {1, 2, 3, 4}},
        # {'fbs': {0, 1}},
        # {'restecg': {0, 1, 2}},
        # {'exang': {0, 1}},
        # {'slope': {1, 2, 3}},
        # {'ca': {0.0, 1.0, 2.0, 3.0}},
        # {'thal': {3.0, 6.0, 7.0}}

        print('\n')
        print('---- user-specified constraints -----')
        constraints = {'age': ('ge', 1),
                   'sex': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## Iris data set
    elif dataset['name'] == 'iris':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., CRIM) and features with set {} values (e.g., CHAS) are discrete

        # {'sepal length (cm)': [4.3, 7.9]},
        # {'sepal width (cm)': [2.0, 4.4]},
        # {'petal length (cm)': [1.0, 6.9]},
        # {'petal width (cm)': [0.1, 2.5]}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## Diabetes data set
    elif dataset['name'] == 'diabetes':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., CRIM) and features with set {} values (e.g., CHAS) are discrete

        # {'age': [-0.107225631607358, 0.110726675453815]},
        # {'sex': [-0.044641636506989, 0.0506801187398187]},
        # {'bmi': [-0.0902752958985185, 0.17055522598066]},
        # {'bp': [-0.112399602060758, 0.132044217194516]},
        # {'s1': [-0.126780669916514, 0.153913713156516]},
        # {'s2': [-0.115613065979398, 0.198787989657293]},
        # {'s3': [-0.10230705051742, 0.181179060397284]},
        # {'s4': [-0.076394503750001, 0.185234443260194]},
        # {'s5': [-0.126097385560409, 0.133598980013008]},
        # {'s6': [-0.137767225690012, 0.135611830689079]}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'age': ('ge', 1),
                   'sex': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    ## Boston house price data set
    elif dataset['name'] == 'boston-house-prices':
        ## Feature names and their possible values
        ## Features with range [] values are continuous (e.g., CRIM) and features with set {} values (e.g., CHAS) are discrete

        # {'CRIM': [0.00632, 88.9762]}
        # {'ZN': [0.0, 100.0]}
        # {'INDUS': [0.46, 27.74]}
        # {'NOX': [0.385, 0.871]}
        # {'RM': [3.5610000000000004, 8.78]}
        # {'AGE': [2.9, 100.0]}
        # {'DIS': [1.1296, 12.1265]}
        # {'RAD': [1, 24]}
        # {'TAX': [187, 711]}
        # {'PTRATIO': [12.6, 22.0]}
        # {'BLACK': [0.32, 396.9]}
        # {'LSTAT': [1.73, 37.97]}
        # {'CHAS': {0, 1}}

        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'AGE': ('ge', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

    print('\n')
    print('N.B. preferences are taken into account when ACTIONABILITY=True!')
    print('\n')

    preferences = {'constraint': constraint,
                   'importance': importance}

    return preferences
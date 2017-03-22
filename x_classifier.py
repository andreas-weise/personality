# classify something

from sklearn.model_selection import cross_val_score
from sklearn import svm
import feature_extractor as fe
import csv
import datetime


def main():
    """ main function, called if module is run as main program"""

    # load raw csv file and extract relevant lines (marked by 'GEN' for general)
    with open('data/sarcasm_v2.csv') as datafile:
        raw_data = list(csv.reader(datafile))
        data = [line[-1] for line in raw_data if line[0] == 'GEN']
        labels = [line[1] for line in raw_data if line[0] == 'GEN']

    # load config file
    with open('conf.txt') as conffile:
        conf_all = set(line.strip() for line in conffile)

    # compute score, for each line in the config individually and all together
    confs = [line for line in conf_all]
    confs.append([line for line in conf_all])
    for conf in confs:
        print('computing score for: %s... ' % conf, end='')
        features = fe.extract_features(data, conf)
        score = cross_val_score(svm.SVC(), features, labels,
                                scoring='accuracy', cv=10).mean()
        score = round(score, 3)
        print(score)
        with open('experiments3.csv', 'a') as f:
            f.write(";".join(
                ('{:%Y-%m-%d;%H:%M:%S}'.format(datetime.datetime.now()),
                 str(conf), str(score), '\n')))

if __name__ == "__main__":
    main()

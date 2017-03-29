# label a status update for personality

from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn import svm, preprocessing, metrics
import feature_extractor as fe
import numpy
import re
import sys
import csv
import codecs
import argparse
import pickle
import operator
import itertools
import os
from subprocess import call

class_idx = [7, 8, 9, 10, 11]

def load_data(datafile):
	with codecs.open(datafile, encoding="latin-1") as f:
		reader = csv.reader(f)
		return next(reader), list(reader)

def load_conf_file(conffile):
	conf = list(line.strip() for line in open(conffile))
	return conf

def predict_trait(X, Y):
	scores = cross_val_score(svm.SVC(), X, Y, scoring='accuracy', cv=10)
	return scores.mean()

def handle_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--datafile', help='file containing data for training or testing', required=True, dest='datafile')
	parser.add_argument('-c', '--conffile', help='file containing list of features to be extracted', dest='conffile', required=True)
	parser.add_argument('-e', '--expdir', help='directory for storing conf file and models associated with this experiment', required=True)
	parser.add_argument('-l', '--load', action='store_true', help='include to load models from <expdir> instead of training new')
	return vars(parser.parse_args())

if __name__ == "__main__":
	args = handle_args()
	print(args)
	header, data = load_data(args['datafile'])
	labels = numpy.asarray([[line[i] for i in class_idx] for line in data]).T.tolist()

	conf = load_conf_file(args['conffile'])

	# look through each of these feature sets and pick the best
	# for each personality trait
	'''feat_list = [ ['bag_of_pos_trigrams','bag_of_pos_bigrams','bag_of_pos_unigrams'],
		['bag_of_trigrams','bag_of_bigrams','bag_of_unigrams'],
		['characters_per_word','unique_words_ratio','words_per_sentence'],
		['char_unigram','char_bigram','char_trigram']
		] + [[x] for x in conf]'''
	feat_list = [['char_bigram'], ['word2vec_avg'], ['char_unigram','char_bigram','char_trigram']]

	# concat statuses by user id
	user_ids = []
	source_texts = []
	for line in data:
	    if len(user_ids) == 0 or line[0] != user_ids[-1]:
	        source_texts.append(line[1])
	    else:
	        source_texts[-1] = source_texts[-1] + '\n\n' + line[1]
	    user_ids.append(line[0])

	if not args['load']:

		# for each personality set
		for i in range(len(class_idx)):
			trait = header[class_idx[i]]

			# run each feature set through the ring
			models = []
			for feat_set in feat_list:

				# check if we've already done this experiment before (mostly for debugging)
				if not os.path.exists('pickle_models/%s/%s.p' % (trait, " ".join(feat_set))):

					# evaluate with individuals
					feat_mat = fe.extract_features([line[1] for line in data], feat_set, 1)

					accs = cross_val_score(svm.SVC(class_weight='balanced'), feat_mat, labels[i], cv=10, n_jobs=-1)
					acc = numpy.mean(accs)
					clf = svm.SVC(class_weight='balanced').fit(feat_mat, labels[i])

					models.append((acc, feat_set, clf))

					print("%s, %s: %.2f" % (", ".join(feat_set), header[class_idx[i]], acc))

					# pickle the model (for debugging)
					with open("pickle_models/%s/%s.p" % (trait, " ".join(feat_set)), 'wb') as f:
						pickle.dump((acc, feat_set, clf), f)
				else:
					with open('pickle_models/%s/%s.p' % (trait, " ".join(feat_set)), 'rb') as f:
						acc, _, clf = pickle.load(f)

					models.append((acc, feat_set, clf))

					print("%s, %s: %.2f" % (", ".join(feat_set), header[class_idx[i]], acc))

			# pick the best model
			best_model = max(models, key=operator.itemgetter(0))
			print("final model, %s, %s: %.2f" % (", ".join(best_model[1]), header[class_idx[i]], best_model[0]))

			# pickle that model (along with a list of features)
			with open("%s/%s.p" % (args['expdir'], trait), 'wb') as f:
				pickle.dump((best_model[1], best_model[2]), f)

	else:
		for i in range(len(class_idx)):

			trait = header[class_idx[i]]

			with open("%s/%s.p" % (args['expdir'], trait), 'rb') as f:
				feat_set, clf = pickle.load(f)

			feat_mat = fe.extract_features([line[1] for line in data], feat_set, 1)

			print("%s, %s: %.2f" % (", ".join(feat_set), trait, clf.score(feat_mat, labels[i])))

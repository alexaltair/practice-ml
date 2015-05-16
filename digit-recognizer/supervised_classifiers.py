from matplotlib import pyplot
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn import tree, svm, naive_bayes, linear_model
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import numpy

SUBMITTING = False
NUMBER_OF_EXAMPLES = 42000  # Max 42000
SCALE = False
DECOMPOSITION = True

def load_and_split_data(proportion):
    print("Loading data...")
    skiprows = 42000-NUMBER_OF_EXAMPLES+1
    all_data = numpy.loadtxt('train.csv', skiprows=skiprows, delimiter=",")
    if SUBMITTING:
        train_data = all_data
        cv_data = all_data
    else:
        split = len(all_data)/proportion
        train_data = all_data[split:]
        cv_data = all_data[:split]
    return train_data, cv_data


def display_samples(data, predicted):
    samples = data[:8]
    for index in range(len(samples)):
        image = samples[index].reshape((28, 28))
        pyplot.subplot(2, 4, index)
        pyplot.axis('off')
        pyplot.imshow(image, cmap=pyplot.cm.gray_r, interpolation='nearest')
        pyplot.title('Prediction: %i' % predicted[index])

    pyplot.show()


def print_results(expected, predicted):
    print("Classification report:\n%s\n"
          % metrics.classification_report(expected, predicted))
    # print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


def write_submission(test_predicted):
    print("Writing submission file...")
    with open('submission.csv', "w") as f:
        f.write('"ImageId","Label"\n')
        for index in range(len(test_predicted)):
            f.write('%s,"%d"\n' % (index+1, test_predicted[index]))


train_data, cv_data = load_and_split_data(proportion=5)

# Split the labels from the data and scale data
train_labels = train_data[:, 0:1]
cv_labels = cv_data[:, 0:1]

if SCALE:
    print("Scaling data...")
    train_data[:, 1:] = preprocessing.scale(train_data[:, 1:])
    cv_data[:, 1:] = preprocessing.scale(cv_data[:, 1:])

if DECOMPOSITION:
    print("Analyzing PCA...")
    pca_estimator = PCA(n_components=80).fit(train_data[:, 1:])

    transformed_train = pca_estimator.transform(train_data[:, 1:])
    train_data = numpy.hstack((train_labels, transformed_train))

    transformed_cv = pca_estimator.transform(cv_data[:, 1:])
    cv_data = numpy.hstack((cv_labels, transformed_cv))

train_images = train_data[:, 1:]
cv_images = cv_data[:, 1:]

# Define the classifier and fit it to the training data
classifier = naive_bayes.BernoulliNB(alpha=1, binarize=127, fit_prior=False)  # <-- 84%, no preprocessing
# classifier = naive_bayes.MultinomialNB(alpha=1, fit_prior=False)  # <-- 84%, no preprocessing
# classifier = svm.SVC(gamma=0.0017, cache_size=1000)  # <-- 96%, scaling
# classifier = svm.LinearSVC(C=0.0000003, dual=False)  # <-- 91%, scaling
# classifier = linear_model.LogisticRegression(dual=False)  # <-- 84%, no preprocessing
# classifier = tree.DecisionTreeClassifier(criterion='entropy')  # <-- 86%, no preprocessing

# classifier = RandomForestClassifier(n_estimators=28, criterion='entropy')  # <-- 96%, no proprocessing
# classifier = BaggingClassifier(classifier,  # svm.SVC, 93%
#     n_estimators=45,
#     max_samples=0.1,
#     max_features=0.3,
# )

print("Training classifier...")
classifier.fit(train_images, train_labels)


# Predict the labels on the CV data.
expected = cv_labels
print("Predicting CV labels...")
predicted = classifier.predict(cv_images)

print(classifier)
print_results(expected, predicted)

# display_samples(cv_images, predicted)

if SUBMITTING:
    print("Loading test data...")
    test_data = numpy.loadtxt('test.csv', skiprows=1, delimiter=",")
    if SCALE:
        test_data = preprocessing.scale(test_data)
    print("Predicting test labels...")
    test_predicted = classifier.predict(test_data)
    write_submission(test_predicted)

print("Done!")

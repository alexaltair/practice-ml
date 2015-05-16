from matplotlib import pyplot
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy
from scipy.stats import mode

SUBMITTING = False
NUMBER_OF_EXAMPLES = 10000  # Max 42000
SCALE = False
DECOMPOSITION = False

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
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


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
    n_components = 200

    transformed_train = PCA(n_components=n_components).fit_transform(train_data[:, 1:])
    train_data = numpy.hstack((train_labels, transformed_train))

    transformed_cv = PCA(n_components=n_components).fit_transform(cv_data[:, 1:])
    cv_data = numpy.hstack((cv_labels, transformed_cv))

train_images = train_data[:, 1:]
cv_images = cv_data[:, 1:]


# Define the estimator and fit it to the training data
estimator = KMeans(n_clusters=10)

print("Training estimator...")
estimator.fit(train_images)

# Determine mapping between clusters and labels
cluster_of_label = {}
import pdb; pdb.set_trace()
for label in range(10):
    matches = train_data[train_data[:, 0] == label]
    predicted = estimator.predict(matches[:, 1:])
    cluster = mode(predicted)[0][0]
    cluster_of_label[label] = cluster


# Remap labels to clusters
cv_clusters = []
for i in range(len(cv_labels)):
    label = int(cv_labels[i])
    cv_clusters.append(cluster_of_label[label])

# Predict the labels on the CV data.
expected = cv_clusters
print("Predicting CV labels...")
predicted = estimator.predict(cv_images)

print(estimator)
print_results(expected, predicted)

# display_samples(cv_images, predicted)

if SUBMITTING:
    print("Loading test data...")
    test_data = numpy.loadtxt('test.csv', skiprows=1, delimiter=",")
    if SCALE:
        test_data = preprocessing.scale(test_data)
    print("Predicting test labels...")
    test_predicted = estimator.predict(test_data)
    write_submission(test_predicted)

print("Done!")

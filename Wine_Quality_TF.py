import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


set_data_flag = 'normalized_data'
#set_data_flag = 'unnormalized_data'
#Reading the pre-processed data csv into a data frame

print('\n\n#################### Running with {} ####################\n\n'.format(set_data_flag))

if(set_data_flag == 'normalized_data'):
    wine_data = pd.read_csv('/Users/abhave/PycharmProjects/BMC/Secops/ML/winedata_pre_processed.csv', sep=',')
    # creating the label and feature list for TF.
    features_list = wine_data.columns.values[0:-2]
    labels_list = wine_data.columns.values[-1]
    #Creating inputs and outputs based on labels and converting them to a matrix that can be used.
    outputs = [int(0) if item == 'Good' else int(1) for item in wine_data['quality_label']]
    inputs = wine_data.iloc[:,0:-2].get_values()
else:
    # Reading the pre-processed data csv into a data frame
    wine_data = pd.read_csv('/Users/abhave/PycharmProjects/BMC/Secops/ML/winequality-red.csv', sep=';')
    # creating the label and feature list for TF.
    features_list = wine_data.columns.values[0:-1]
    labels_list = wine_data.columns.values[-1]
    # Creating inputs and outputs based on labels and converting them to a matrix that can be used.
    outputs = [int(0) if item > 7 else int(1) for item in wine_data['quality']]
    inputs = wine_data.iloc[:, 0:-1].get_values()

print('\nThe list of features we have available for this model are: \n{}\n'.format(features_list))
print('\n\nSimilarly the label columns we are using is: \n{}\n\n'.format(labels_list))


#Convert the single point output into a 2x1 matrix for Tensorflow. The idea is to use array's and indices for arrays to manipulate the
#data, such that the y with a value of 1 will show up as [1, 0] and y with value of 0 will show up as [0, 1]
def convert_output_to_Matrix(labels, num_classes):
    converted_matrix = []
    for label in labels:
        index = [1]*num_classes
        index[label] = 0
        converted_matrix.append(index)

    return(converted_matrix)

y_matrix = convert_output_to_Matrix(outputs, 2)

# As per the requirements creating a test data set of 30% proportion of the entire dataset available.
#12 will be used as the int seed to the random number generator to select test data randomply fromt he avialable data.
X_train, X_test, Y_train, Y_test = train_test_split(inputs, y_matrix, test_size = 0.3, random_state=12)
print('Size of training data is: ', len(X_train))
print('Size of testing data is: ', len(X_test))

#Model

#Setting the learning rate at 0.001
learning_rate = .5
#Setting the batch size for model training and testing.
batch_size = X_train.shape[0] // 10
#Programatically identifing the number of features based on data and training.
number_of_features = X_train.shape[1]
#Defining the total numnber of classifications to be generated.
number_of_classifications = 2
#Defining the # of iterations that will be run.
epochs = 1000
#Setting a variable to print the loss every 100 iterations
epoch_hunders_identifier = epochs//10
neurons_in_hidden_layer = 10


#Creating tensors to hold inputs and output.
input_placeholder = tf.placeholder("float32", [None, number_of_features])
output_placeholder = tf.placeholder("float32", [None, number_of_classifications])


def make_batch(input, output, batch_size):
    output_size = len(output)
    index_sample = np.random.choice(output_size, batch_size, replace=False)
    output_array = np.array(output)
    X_batch = input[index_sample]
    Y_batch = output_array[index_sample]
    return X_batch, Y_batch


def hidden_layer(X_tensor, neurons_in_hidden_layer):
    Weight = tf.Variable(tf.random_uniform([number_of_features, neurons_in_hidden_layer]))
    Bias = tf.Variable(tf.zeros([neurons_in_hidden_layer]))
    output = tf.nn.softmax(tf.matmul(X_tensor, Weight) + Bias)
    return output


def output_layer(output_from_hidden_layer, number_of_classifications):
    num_inputs = output_from_hidden_layer.get_shape()[1].value
    Weight = tf.Variable(tf.zeros([num_inputs, number_of_classifications]))
    Bias = tf.Variable(tf.zeros([number_of_classifications]))
    output = tf.nn.softmax(tf.matmul(output_from_hidden_layer, Weight) + Bias)
    return output

def training(loss, learning_rate):
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return training_step

def calculate_loss_function(output, y_tensor, batch_size):
    loss = -tf.reduce_sum(y_tensor * tf.log(output)/ batch_size)
    return loss

def compute_accuracy(output, y_tensor):
    prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_tensor, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, 'float'))
    return accuracy


def run_model():
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        sum_of_loss = 0
        for i in range(epochs):
            X_batch, Y_batch = make_batch(X_train, Y_train, batch_size)
            feed_dict = {input_placeholder: X_batch, output_placeholder: Y_batch}
            _, current_loss = sess.run([training_step, loss], feed_dict)
            sum_of_loss += current_loss

            if i % epoch_hunders_identifier == 99:
                average_loss = sum_of_loss / epoch_hunders_identifier
                print('Epoch: {:4d}, avg_loss: {:0.3f}'.format(i + 1, average_loss))
                sum_of_loss = 0

        print('Model fitting is complete.......')

        # Calclulate final accuracy
        X_batch, Y_batch = make_batch(X_test, Y_test, batch_size)
        feed_dict = {input_placeholder: X_batch, output_placeholder: Y_batch}
        print('Final accuracy achieved by this model is: {:0.2%}\n\n'.format(sess.run(accuracy, feed_dict)))


def singleLayer_model():
    print('Running singleLayer Model')
    with tf.name_scope("output") as scope:
        y_output = output_layer(input_placeholder, number_of_classifications)

    with tf.name_scope("Calculate_Loss") as scope:
        global loss
        loss = calculate_loss_function(y_output, output_placeholder, batch_size)

    with tf.name_scope('training') as scope:
        global training_step
        training_step = training(loss, learning_rate)

    with tf.name_scope('accuracy') as scope:
        global accuracy
        accuracy = compute_accuracy(y_output, output_placeholder)

    run_model()

def multiLayer_model():
    print('Running multiLayer Model')
    #Hidden layer model run
    with tf.name_scope("hidden_layer") as scope:
        y_relu = hidden_layer(input_placeholder, neurons_in_hidden_layer)

    #Output Layer model run
    with tf.name_scope("output") as scope:
        y_output = output_layer(y_relu, number_of_classifications)

    #Caculating the loss
    with tf.name_scope("Calculate_loss") as scope:
        global loss
        loss = calculate_loss_function(y_output, output_placeholder, batch_size)
        #tf.summary.scalar('loss', loss)

    #Define training steps
    with tf.name_scope('training') as scope:
        global training_step
        training_step = training(loss, learning_rate)

    #Calculate accuracy
    with tf.name_scope('accuracy') as scope:
        global accuracy
        accuracy = compute_accuracy(y_output, output_placeholder)
        #tf.summary.scalar('accuracy', accuracy)

    run_model()

multiLayer_model()
singleLayer_model()


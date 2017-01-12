#Carter Johnson
#TensorFlow Linear Model Tutorial
#Wide_Tutorial.py

#We solve a binary classification problem: 
#Given census data about a person such as age, gender, education 
#and occupation (the features), we will try to predict whether or 
#not the person earns more than 50,000 dollars a year (the target label).
# We will train a logistic regression model, and given an individual's 
#information our model will output a number between 0 and 1, which can be
# interpreted as the probability that the individual has an annual income 
#of over 50,000 dollars.

import pandas as pd
import tensorflow as tf

import tempfile
import urllib
# ------------ READING THE CENSUS DATA --------------------------------

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

#read census data in Pandas dataframes
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

#for binary classification, construct a label column named Label whose value is 1
#if income is over 500k, 0 otherwise
LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

#label columns as categorical or continuous
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

# ------------ CONVERTING DATA INTO TENSORS --------------------------------

# input data is specified by the InputBuilder function
# builder function will not be called until passed into TF.Learn methods like fit, evaluate
# InputBuilder returns feature_cols (dictionary from feature column names to Tensors/Sparse)
# and label (tensor containing the label column (500k or not))

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)


# --------- Selecting and Engineering Features for the Model ------------

#Base Categorical Feature Columns
#set categorical feature values an integer key
gender = tf.contrib.layers.sparse_column_with_keys(
  column_name="gender", keys=["Female", "Male"])

#use hash bucket to hash an integer ID to each new value in the feature column as encountered
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
race = tf.contrib.layers.sparse_column_with_hash_bucket("race", hash_bucket_size=100)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

#Base Continuous Feature Columns
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

#bucketization to divide continuous feature into set of buckets - converts to categorical with hashing
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

#Crossed feature columns - take into account correlation between two features
#e.g. education and occupation correlate more together towards income earning than separately
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))

#crossed column over caterogical, bucketized real-valued feature, or another cross_column
age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column(
  [age_buckets, education, occupation], hash_bucket_size=int(1e6))

# --------- Defining the Logistic Regression Model --------------------

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race,
  age_buckets, education_x_occupation, age_buckets_x_education_x_occupation],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=0.5,
    l2_regularization_strength=1.0),
  model_dir=model_dir)


# --------- Training and Evaluating our Model --------------------

#Train
m.fit(input_fn=train_input_fn, steps=200)

#evaluate predictive accuracy
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print(key, results[key])

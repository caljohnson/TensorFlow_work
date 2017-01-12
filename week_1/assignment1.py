#Carter Johnson
#Andrew Ng's Coursera course on ML
#Assignment 1 - Linear Regression

import tensorflow
import pandas as pd
import matplotlib.pyplot as plt

def run_linear_regression():
  columns = ["population", "profit"]
  df_train = pd.read_csv("ex1data1.txt", names=columns, skipinitialspace=True)

  plt.figure()
  plt.plot(df_train.population, df_train.profit, 'x')
  plt.xlabel('Population of City in 10,000s')
  plt.ylabel('Profit in $10,000s')
  plt.show()
  plt.close()

def main():
	run_linear_regression()

if __name__ == '__main__':	
	main()

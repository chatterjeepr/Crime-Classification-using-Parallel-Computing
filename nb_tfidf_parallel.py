
import pyspark
spark = pyspark.sql.SparkSession.builder.appName("clipper-pyspark").getOrCreate()

sc = spark.sparkContext

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
np.random.seed(60)

import time

#Read the data into spark datafrome
from pyspark.sql.functions import col, lower
df = spark.read.format('csv')\
          .option('header','true')\
          .option('inferSchema', 'true')\
          .option('timestamp', 'true')\
          .load('/content/train.csv')

data = df.select("*")
df.show(5)
df.count()

def top_n_list(df,var, N):
    '''
    This function determine the top N numbers of the list
    '''
    print("Total number of unique value of"+' '+var+''+':'+' '+str(df.select(var).distinct().count()))
    print(' ')
    print('Top'+' '+str(N)+' '+'Crime'+' '+var)
    df.groupBy(var).count().withColumnRenamed('count','totalValue')\
    .orderBy(col('totalValue').desc()).show(N)


top_n_list(data, 'Category',10)
print(' ')
print(' ')
top_n_list(data,'Description',10)

data.select('Category').distinct().count()

training, test = data.randomSplit([0.7,0.3], seed=60)
#trainingSet.cache()
print("Training Dataset Count:", training.count())
print("Test Dataset Count:", test.count())

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, OneHotEncoder, StringIndexer, VectorAssembler, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes

#----------------Define tokenizer with regextokenizer()------------------
regex_tokenizer = RegexTokenizer(pattern='\\W')\
                  .setInputCol("Description")\
                  .setOutputCol("tokens")

#----------------Define stopwords with stopwordsremover()---------------------
extra_stopwords = ['http','amp','rt','t','c','the']
stopwords_remover = StopWordsRemover()\
                    .setInputCol('tokens')\
                    .setOutputCol('filtered_words')\
                    .setStopWords(extra_stopwords)

#-----------Using TF-IDF to vectorise features instead of countVectoriser-----------------
hashingTf = HashingTF(numFeatures=10000)\
            .setInputCol("filtered_words")\
            .setOutputCol("raw_features")

#Use minDocFreq to remove sparse terms
idf = IDF(minDocFreq=5)\
        .setInputCol("raw_features")\
        .setOutputCol("features")

#-----------Encode the Category variable into label using StringIndexer-----------
label_string_idx = StringIndexer()\
                  .setInputCol("Category")\
                  .setOutputCol("label")

#---------Define classifier structure for Naive Bayes----------
nb = NaiveBayes(smoothing=1)

def metrics_ev(labels, metrics):
    '''
    List of all performance metrics
    '''
    # Confusion matrix
    print("---------Confusion matrix-----------------")
    print(metrics.confusionMatrix)
    print(' ')
    # Overall statistics
    print('----------Overall statistics-----------')
    print("Precision = %s" %  metrics.precision())
    print("Recall = %s" %  metrics.recall())
    print("F1 Score = %s" % metrics.fMeasure())
    print(' ')
    # Statistics by class
    print('----------Statistics by class----------')
    for label in sorted(labels):
       print("Class %s precision = %s" % (label, metrics.precision(label)))
       print("Class %s recall = %s" % (label, metrics.recall(label)))
       print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
    print(' ')
    # Weighted stats
    print('----------Weighted stats----------------')
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

# Measure time for model prediction
start_time = time.time()
pipeline_idf_nb = Pipeline().setStages([regex_tokenizer,stopwords_remover,hashingTf, idf, label_string_idx, nb])
model_idf_nb = pipeline_idf_nb.fit(training)
predictions_idf_nb = model_idf_nb.transform(test)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator_idf_nb = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_idf_nb)
print(' ')
print('-----------------------------Accuracy-----------------------------')
print(' ')
print('                          accuracy:{}:'.format(evaluator_idf_nb))
end_time = time.time()
elapsed_time_prediction1 = end_time - start_time
# Print Time taken for Model Prediction
print(f"Time taken for Model Prediction: {elapsed_time_prediction1:.2f} seconds")

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create a MulticlassClassificationEvaluator
evaluator_idf_nb = MulticlassClassificationEvaluator(metricName='accuracy', labelCol='label', predictionCol='prediction')

# Define the parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(hashingTf.numFeatures, [10000, 50000, 100000]) \
    .addGrid(idf.minDocFreq, [1, 5, 10]) \
    .addGrid(nb.smoothing, [0.1, 0.5, 1.0]) \
    .build()

# Create a cross-validator
crossval = CrossValidator(estimator=pipeline_idf_nb,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator_idf_nb,
                          numFolds=3)

# Fit the cross-validator to the data
cv_model = crossval.fit(x_train['Descript'], y_train)

# Make predictions on the test set using the best model
best_model = cv_model.bestModel
predictions = best_model.transform(x_test['Descript'])

# Evaluate the best model
best_accuracy = evaluator_idf_nb.evaluate(predictions)

print(' ')
print('-----------------------------Best Model Accuracy-----------------------------')
print(' ')
print('                          accuracy:{}:'.format(best_accuracy))
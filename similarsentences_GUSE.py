#Import the libraries
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

#Finding top 10 similar sentences
def run_and_extract(session_, input_tensor_, messages_, encoding_tensor,similaritydf):
  message_embeddings_ = session_.run(encoding_tensor, feed_dict={input_tensor_: messages_})
  print(message_embeddings_)
  
  corr=np.inner(message_embeddings_, message_embeddings_)
  
  
  row=0
  
  cosines = []
  
  for i in range(len(corr)):
       
          sorted_similarity=sorted(corr[i])
          cosines = sorted_similarity[len(sorted_similarity)-11:len(sorted_similarity)-1]
          cosines.reverse()
          
          closestIdx = corr[i].argsort()
          similar=['']*len(closestIdx)
          for j in range(0, len(closestIdx)): 
              similar[j]= str(messages[closestIdx[j]])
          similar.reverse()
          similar=similar[1:11]
          for k in range(len(similar)):
              similaritydf.set_value(row,'Input sentence',messages[i])
              similaritydf.set_value(row,'Similar sentence found',similar[k])
              similaritydf.set_value(row,'Similarity score',cosines[k])
              row=row+1
          print(corr[i])
          print("**************")
          print(len(similaritydf))
  return similaritydf

# read input sentences text file
import pandas as pd
df=pd.read_csv('excavator.txt',header=None)
df.drop_duplicates(inplace=True)
messages = [line for line in df[0]]


similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  similaritydf=pd.DataFrame()
  similarityresults=run_and_extract(session, similarity_input_placeholder, messages,similarity_message_encodings,similaritydf)
  
similarityresults.to_csv('SimilarityResultsfinal.csv',index=False)

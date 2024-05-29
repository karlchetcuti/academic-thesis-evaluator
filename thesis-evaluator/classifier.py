from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from mistralai import exceptions
import os, csv, ast
import utils as ut
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

load_dotenv()
ut.configure_logging()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
logger = ut.get_logger()

#Text splitter to split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 6500,
    chunk_overlap = 20,
    length_function = len,
)

#Load docs and add to list
def load_docs(folder_path, docs, grade):
    #Loading and labelling docs
    loaders = [PyPDFLoader(os.path.join(folder_path, file)) for file in os.listdir(folder_path)]
    for loader in loaders:
        docs.append((loader.load(), grade))
    logger.info("Successfully loaded documents.")
    return docs

#Create dataset and split documents into chunks
def create_dataset(docs, text_splitter):
    data = []
    for doc in docs:
        pages = ""
        #Concatenating all pages in document together
        for page in doc[0]:
            pages += page.page_content
        #Splitting pages into chunks of 6.5k tokens (Max tokens for mistral-embed is 8k)
        pages = text_splitter.split_text(pages)
        #Append data
        for page in pages:
            data.append((doc[1], page)) #where doc[1] is the grade
    logger.info("Successfully created dataset.")
    return data

#Create csv file from dataset
def create_csv(fields, data, filename):
    with open(filename, 'w', newline='', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(data)
        logger.info("Successfully created CSV file.")
    return

#Generating embeddings for each chunk
def get_embeddings(data):
    logger.info("Getting embeddings...")
    all_embeddings = []
    for chunk in data:
        try:
            all_embeddings += embeddings.embed_documents([chunk])
        except TypeError:
            logger.error(f"Error. The chunk {chunk} returned a Type Error.")
            break
        except exceptions.MistralAPIException:
            logger.error(f"Error. Too many tokens in chunk {chunk}.")
        except Exception as e:
            logger.error(f"Unexpected error occured with {chunk}. Error: {str(e)}")
    logger.info("Successfully generated embeddings.")
    return [e for e in all_embeddings]

#Visualise embeddings using t-SNE
def plot_tsne(df):
    tsne = TSNE(n_components=2, random_state=0).fit_transform(np.array(df['Embeddings'].to_list()))
    ax = sns.scatterplot(x=tsne[:, 0],
                         y=tsne[:, 1],
                         hue=np.array(df['Label'].to_list()))
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
    plt.savefig('tsne.png')
    logger.info("Successfully plotted t-SNE.")
    return

def train_classifier(df):
    #Splitting dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        df['Embeddings'], 
        df['Label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['Label']
    )
    #Standardizing features
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train.to_list())
    # x_test = scaler.transform(x_test.to_list())
    logger.info("Training classifier...")
    #Training classification model
    clf = LogisticRegression(random_state=0, C=1.0, max_iter=500).fit(
        x_train.to_list(), y_train.to_list()
    )
    logger.info("Successfully trained classifier.")
    #Testing model
    x_test = x_test.to_list()
    y_test = y_test.to_list()
    y_pred = clf.predict(x_test)
    clf.score(x_test, y_test)
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    #Save model
    pkl.dump(clf, open('classifier.pkl', 'wb'))
    return x_test, y_test, clf

#Preparing document for classification testing
def prepare_test_doc(path, file_name):
    path = os.path.join(path, file_name)
    loader = PyPDFLoader(path)
    doc = loader.load()
    pages = ""
    for page in doc:
        pages += page.page_content
    pages = text_splitter.split_text(pages)
    embed_pages = get_embeddings(pages)
    return embed_pages

#Predicting grade for an input document
def grade_doc(doc, clf):
    A_counter = 0
    not_A_counter = 0
    total_prob = 0
    #Predict grade for each chunk of doc
    for embedding in doc:
        probability = clf.predict_proba([embedding])[0]
        if probability[1] >= 0.5:
            not_A_counter += 1
            total_prob += probability[1]
        else:
            A_counter += 1
    #Return 'A' only if majority of chunks are 'A'
    if A_counter > not_A_counter:
        return 'A'
    #These are the grade splittings based on probability:
    #80-100: F
    #70-80: D
    #60-70: C
    #50-60: B
    probability = total_prob / not_A_counter
    if probability < 0.65:
        return 'B'
    elif probability < 0.8:
        return 'C'
    elif probability < 0.95:
        return 'D'
    return 'F'

#Loading and labelling 'A' and 'Not A' docs
# docs = []
# docs = load_docs('dataset/A', docs, 'A')
# docs = load_docs('dataset/Not A', docs, 'Not A')
# docs = load_docs('dataset/Test', docs, 'Not A') #<-- TESTING ONLY

#Creating labelled dataset with docs and split them into chunks for embedding later
# data = create_dataset(docs, text_splitter)

#Creating csv file for dataset
# fields = ['Label', 'Text', 'Embeddings']
# create_csv(fields, data, 'dataset.csv')
# df = pd.read_csv('dataset.csv', encoding='utf-8') #cp1252
#Remove empty rows due to images.
# df = df.dropna(subset=['Text'])
# print(df)

#Generate embeddings for each chunk of text
# df['Embeddings'] = get_embeddings(df['Text'].tolist())
# print(df)

#Updating csv file with embeddings to store data
# df.to_csv('dataset.csv', index=False)

#Use already existing embeddings
# df = pd.read_csv('dataset.csv', encoding='utf-8') #cp1252
#Convert from string to list
# df['Embeddings'] = df['Embeddings'].apply(ast.literal_eval)
# print(df)

#Plotting t-SNE scatter plot
# plot_tsne(df)

#Training Logistic Regression classifier
# x_test, y_test, clf = train_classifier(df)
# print(f"Precision: {100*np.mean(clf.predict(x_test) == y_test):.2f}%")

#Loading existing classifier
# clf = pkl.load(open('classifier.pkl', 'rb'))

#Using classifier
# input_doc = prepare_test_doc('static/files/')
# grade = grade_doc(input_doc, clf)
# print(grade)
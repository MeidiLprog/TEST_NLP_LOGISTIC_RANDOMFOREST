import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')

from PIL import Image # convert scanned to images

#Part 1, have to use os lib

import pytesseract # useful, if the pdf file is scanned, we convert it to images and apply some text recognition
import json #useful to save my chunks in a json file
#tesseract didn't work because it wasn't found, by adding the path explicitely I'm assuring a success
_PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = _PATH_TO_TESSERACT
'''
print("Let us see where we are \n")
#trying a few things here and there, not relevant to the tp
print(os.environ.get("PATH"),end="\n")
for key, value in os.environ.items():
    print(key,":",value)
print(f"\nWhere are we ? {os.getcwd()}\n")
'''

#ocr function

def ocr_string_return(path : str) -> str:

    if not isinstance(path,str):
        raise ValueError("Is not a string")
    if not os.path.isfile(path):
        raise ValueError("Doesn't lead anywhere ! \n")
    if os.path.getsize(path) == 0: raise ValueError("An image cannot be empty ! \n")
    img = Image.open(path)
    return pytesseract.image_to_string(img)



#let us get the content of our repository

#we need to store the path and the content of our file to be able build a dataframe based on it


file_path = r"C:\Users\MLSD24\Desktop\tp3_reco\tobacco"

path : list = []
name_folder : list = []
content : list = []

for truc in os.listdir(file_path):
    folder = os.path.join(file_path,truc) #build the whole path
    
    if not os.path.isdir(folder): continue #check whether it's a folder
    for images in os.listdir(folder):
        complete_path = os.path.join(folder,images)
        if images.lower().endswith((".jpg",".jpeg",".png",".tiff")):
            path.append(complete_path)
            name_folder.append(truc)
            


print(f" files : {path} \n")



data = pd.DataFrame({

    'path' : path,
    'name' : name_folder,

})


print(data)

#let us check the distribution

info = data["name"].value_counts() #thanks to value_counts I can retrieve the name of the folder + the values, useful to create my barplot
'''
plt.bar(info.index,info.values)
plt.xlabel("Classes")
plt.ylabel("Values")
plt.title("Distribution of our data")
plt.xticks(rotation= 45)
for i,v in enumerate(info):
    plt.text(i,v,str(v),ha="center")
plt.show()
'''


#part with OCR
#computer not powerful enough, I shall do an OCR on 5 images per 
#data["text_ocr"] = data["path"].apply(ocr_string_return)

sample = data.groupby("name").head(5).copy()

sample["text_ocr"] = sample["path"].apply(ocr_string_return)


#View some images
#Identify potential errors:



print(sample.head(10))

for i,rows in sample.iterrows():

    img = Image.open(rows["path"])

    plt.imshow(img, cmap="gray")
    plt.title(f"Class: {rows['name']}")
    plt.axis("off")
    plt.show()

    print(rows["text_ocr"])

#Q4 erreurs OCR

'''
Il semblerait que nous ayons quelques soucis avec les text et apparitions de signes non demandé ex: —~-Original Message———*
ae BST7 ray ye rico) nest |        ici: clairement ma au lieu de me, BTS7 ? rico ? surement remplacé par hasard pour correspondre à la transcription textuelle mais 
sémantiquement et grammaticalement/ortographiquement incorrecte 

Associates, i... ici il manque des caractères

true January 1974 - 2nd Cover ici il manque des mots

On a des sauts de lignes, des caractères étranges/manquants

'''

#TF-IDF part 

stop_words = set(stopwords.words("english"))

delete_space = re.compile("\s+")
delete_punc = re.compile(f"[{re.escape(string.punctuation)}]")
def normalize_text(text : str):
    if not isinstance(text,str):
        raise ValueError("Is to be a string \n")
    if len(text) == 0:
        print("It cannot be empty \n")
        return -1
    
    text = text.lower()
    text = delete_punc.sub(" ",text)
    text = re.sub(r"\d+"," ",text) #numerics
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if not i in stop_words]
    
    text = " ".join(tokens)
    return delete_space.sub(" ",text).strip()

sample["ready_to_use_text"] = sample["text_ocr"].apply(normalize_text)


#Build TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1,1))

X_v = vectorizer.fit_transform(sample["ready_to_use_text"])

voc_size = len(vectorizer.vocabulary_)
print(f"Size of vocab {voc_size}")


#training data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,ConfusionMatrixDisplay,confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns
y = sample["name"]
X = X_v
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2 ,random_state=42,stratify=y)

model_log = LogisticRegression(max_iter=300)
model_log.fit(X_train,y_train)

y_pred_log = model_log.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test,y_pred_log)}")
print(f"Accuracy: {f1_score(y_test,y_pred_log,average='macro')}")

cm = confusion_matrix(y_test,y_pred_log)
plt.figure(figsize=(6,10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_log.classes_)
disp.plot(cmap="Blues")
plt.show()


from sklearn.ensemble import RandomForestClassifier


X_rdf = X_v #our vector

model_rdf = RandomForestClassifier(n_estimators=200,criterion="gini",max_depth=10,random_state=42)

model_rdf.fit(X_train,y_train)
y_pred_rf = model_rdf.predict(X_test)
print("Accurary + F1 for random forest \n ")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Macro-F1:", f1_score(y_test, y_pred_rf, average="macro"))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf, cmap="Blues"
)
plt.title("Confusion Matrix — Random Forest")
plt.show()








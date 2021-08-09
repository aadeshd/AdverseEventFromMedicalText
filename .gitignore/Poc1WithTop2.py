import spacy
import pandas as pd
import random
from spacy.training.example import Example

def AE(sampleText):
    nlp = spacy.load("C:\ProgramData\Anaconda3\Lib\site-packages\en_ner_bc5cdr_md\en_ner_bc5cdr_md-0.4.0")
    doc = nlp(sampleText)
    #displacy_image = displacy.render(doc, jupyter = True, style = 'ent')
    entities=[]
    labels=[]
    for ent in doc.ents:
        entities.append(ent)
        labels.append(ent.label_)
    df= pd.DataFrame({'Entities':entities,'Labels':labels})
    return df

def dataPreparation(trainingData):
    data=list()
    for index, row in trainingData.iterrows():
        temp=row["textCptured"]
        sym=temp.split(',')
        txt=str(row["SYMPTOM_TEXT"])
        flag2=0
        t=list()
        #print(txt)
        if ((txt)!=0):
            for i in range(len(sym)):
                #print(sym[i],txt)
                alert=0
                if(len(t)>0):
                    for k in t:
                        if (txt.find(sym[i])) in k:
                            alert=1
                            break
                        else:
                            alert=0
                if alert ==0:
                    if sym[i] in txt:
                        flag2=flag2+1
                        t.append((txt.find(sym[i]),(txt.find(sym[i])+len(sym[i])+1),"AE_Captured"))
        dic={'entities':t}
        if(len(t)!=0):
            tempData=(txt,dic)
        if(len(tempData)!=0):
            data.append(tempData)
    return (data)

def trainModel(trainData, iterations):
    TRAIN_DATA=trainData
    nlp = spacy.load("C:\ProgramData\Anaconda3\Lib\site-packages\en_ner_bc5cdr_md\en_ner_bc5cdr_md-0.4.0")
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe("ner")
    else:
        ner= nlp.get_pipe("ner")
    for _,annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        #optimizer=nlp.begin_training()
        for itn in range(iterations):
            print("starting iterartion"+str(itn))
            random.shuffle(TRAIN_DATA)
            losses={}
            for text,annotations in TRAIN_DATA:
                # create Example
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example],drop=0.2,losses=losses)
            print(losses)
    return (nlp)

def savingmodel(filepath,iterations):
    x=pd.read_csv(filepath)
    dataInFormat= dataPreparation(x)
    NewTrainedModel = trainModel(dataInFormat,iterations)
    NewTrainedModel.to_disk(r"E:\Modelwith2AE2")

def testmodel(testData):
    newnlp=spacy.load(r"E:\Modelwith2AE")
    newdoc=newnlp(testData)
    result=""
    for ent in newdoc.ents:
        print("hello, this is a new models result")
        result=(ent.text,ent.label_)
    return result

def customNER(filepath):
    savingmodel(filepath,30)
    print("Succesfully trained and saved the model ")

customNER("DataToBeUsed.csv")

testtext="Adverse Events: Inflammation in the eye, confusion, headaches, inflammation in ears, cold chills, shivering, and fever like symptoms  Treatments: Primary care physician ran a series of bloodwork and found that after Flu shot I had big drop in white blood cell count and referred me to ophthalmologist and otolaryngologist  ophthalmologist  prescribed Cequa to treat the inflammation in eyes along with fortified caster oil.  otolaryngologist prescribed Prednisone to treat the inflammtion  Time course: Still having adverse events"

print(testmodel(testtext))
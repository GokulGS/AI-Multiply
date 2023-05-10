# Gensim
import gensim
import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pickle
import pandas as pd
import numpy as np
import math
import seaborn as sns
from wordcloud import WordCloud

# pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

from os import path
import os

from Constants import *

#######################################


### Create a word cloud with calculated association strength (idf * relative weights)

def createWordClouds(df_idf, topics_count, maxwords = 10):
    # Wordcloud of Top N LTCs for each topic based on association strength

    i = 1
    col_offset = topics_count +3  # first column in dataframe containing relative_weight
    
    for j in df_idf.columns[col_offset: col_offset+topics_count]:
        
        print("column: {0}".format(j))

        ltc_topic = pd.Series(df_idf[j].values,index=df_idf.MLTC).sort_values(ascending = False)
        
        ltc_dic = ltc_topic.to_dict()
        
#         print(ltc_topic[:20])

        colrs = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        cloud = WordCloud(background_color='white',
                        width=2500,
                        height=1800,
                        max_words=maxwords,
                        colormap='tab10',
                        color_func=lambda *args, **kwargs: colrs[i],
                        prefer_horizontal=1.0)

        topics = df_idf[[j]] # Since we have fixed the topics to 4, we can change this to make it dynamic

        fig, axs = plt.subplots(1)

        topic_words = ltc_dic
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        titl = 'Topic ' + str(j[-1])
        plt.gca().set_title(titl, fontdict=dict(size=16))
        plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()

        ## suffix the base file name
        TARGET_DIR = TOPICS_COUNTS_PATH + str(topics_count)
        
        if not os.path.exists(TARGET_DIR):
            os.mkdir(TARGET_DIR)
            
        WORDCLOUD_SAVE_TO =  TARGET_DIR + '/' + path.basename(WORDCLOUD_FILES).split('.')[0]+ "_"+ str(i) 
        TERMS_LIST_SAVE_TO = TARGET_DIR + '/' + path.basename(TERMS_TOPICS_FILES).split('.')[0]+ "_"+ str(i) + '.' + path.basename(TERMS_TOPICS_FILES).split('.')[1]

        
        fig.savefig(WORDCLOUD_SAVE_TO, dpi=60)
        print('saved to {0}'.format(WORDCLOUD_SAVE_TO))

        ltc_topic[:20].to_csv(TERMS_LIST_SAVE_TO)
        
        i = i+1
        
        

#####
## from `ltc_patients` create `list_of_patients` if it does not exist
## return list of BOW, one per patient
## return list of unique disease terms
def create_bows(bin_matrix):
    list_of_patients = []


    patients = bin_matrix['patient_id'].unique()
    ltcs = bin_matrix.drop('patient_id', axis = 1)
    index = 0

    # Iterate through patients
    for patient in patients:

        # Start with empty list of LTCs for each patient
        patient_ltcs = []

        # Iterate through each LTC for patient
        for ltc in ltcs:

            # Check if patient has LTC
            if bin_matrix.at[index, ltc] == 1:

                # If LTC present, add to list of patient LTCs
                patient_ltcs.append(ltc)   

        # Add list of patient LTCs to list of patients        
        list_of_patients.append(patient_ltcs)

        # Increment index by 1
        index+=1

    ## cache for future use
    with open(BOWs, 'wb') as f:
        pickle.dump(list_of_patients, f)

    with open(LTCs, 'wb') as f:
            pickle.dump(ltcs, f)
            
        
    return list_of_patients, ltcs.columns







        
# Topics generation

# in: bow is the list of bag of words
# in: topics_count is the number of topics to be generated
# returns lda-model
## this method saves the model as a pickle file -- using topics_count as suffix to separate different topics configurations
## parameter alpha is fixed but perhaps this could change??


def bagOfWords2Topics(bow, topics_count):
    id2word = corpora.Dictionary(bow)

    corpus = []
    for text in bow:
        new = id2word.doc2bow(text)
        corpus.append(new)

    lda_model =  gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topics_count,
                                           workers=3,
                                           random_state=100,
                                           chunksize=500,
                                           passes=10,
                                           alpha="asymmetric",
                                           eta=1.0)        
    return lda_model


# compute the probabilities of each bow relative to each topic

def compute_all_bow_probabilities(lda_model):
    # Will store topic similarity score for each of the patients in our corpus
    lda_res_each_ptnt = []

    ## for each bag of words used to generate the topics:
    for i in bows:
        bow = id2word.doc2bow(i)
        lda_res_each_ptnt.append(lda_model[bow])  ## these are the probabilities for this document for each topic
        
    return lda_res_each_ptnt


# input: list of disease terms
# input: corpus_size = number of bows
# return list of ids ordered by the same order as the input terms
def calculate_idf(terms, corpus_size, term_occur):
    
    idf = list()
    for i in range (len(terms)):
        idf.append(math.log10(corpus_size / (term_occur[terms[i]] + 1)))
    return idf


## return a dict term:<number of occurrences>
def term_occurrences(bows):
    
    term_occur = dict()    
    for bow in bows:
        for t in bow:
            try:
                term_occur[t] = term_occur[t] + 1
            except:
                term_occur[t] = 1
    return term_occur


def genTopicsColumns(topics_count):
    return [ "topics_"+str(i) for i in range(topics_count)]

def genWeightedTopicsColumns(topics_count):
    return [ "weighted_topics_"+str(i) for i in range(topics_count)]



# Build a dataframe with each term's relative weight within each topic
# in: lda_model

## input:
## LDA model computed in the previous step
## number of topics in the model

## 1 get terms  native weights from lda_model: weight(term, topic)
## 2 calculate idf for each term in the corpus idf(term)
## to eeach term in topic assign weight  weight(term, topic) * idf(term)
## return dataframe df_idf with schema ['MLTC', topic1, ..., topicl]
def compute_terms_topics_associations(lda_model, topics_count, ltc_names, bows):

     ## init dataframe with schema ['MLTC', 'topic_1',  'topic_2', ..., 'topic_k'] where k = topics_count
    topics_columns = genTopicsColumns(topics_count)

    ## the terms in term_weights are sorted by rank in each topic, so the order is different in each of them
    ## in the following we need to pick a common sorting order, we use alphabetical 
    sorted_terms = sorted(ltc_names)


    cols = ['MLTC']
    [ cols.append(t) for t in topics_columns]
    weighted_topics_df = pd.DataFrame(columns=cols)
    weighted_topics_df['MLTC'] = sorted_terms

    ## get native term weights from LDA
    term_weights = lda_model.show_topics(num_words=300, formatted=False)

#     print(term_weights)
    
#     for t in range(len(topics_columns)):
#         print("topic {}".format(t))
#         for  (name, x) in term_weights[t][1]:
#             print("name: {} weight {}".format(name,x))

    ## step 1: populate weighted_topics_df with native LDA term weight

    ## for each topic:
    for t in range(len(topics_columns)):
        weights_for_topic = [ (name, x) for (name, x) in term_weights[t][1]]  ## list of weights, one for each term, for topic t
        weights_for_topic_sorted = sorted(weights_for_topic, key=lambda t: t[0])  ##  alpha sorting to be consistent across the columns    
        values = [  v for (name, v) in weights_for_topic_sorted ]
        weighted_topics_df[topics_columns[t]] = values     ## assign the list to column topic_t 

#     print(weighted_topics_df)

    ##############
    ## step 2: calculate idf for each term, add idf as new column, calculate weighted terms and add them as new columns
    term_occur = term_occurrences(bows) ## term --> number of occurrences

    idf = calculate_idf(sorted_terms, len(bows), term_occur)

    weighted_topics_df['idf'] = idf

    sum = 0
    for i in range(len(topics_columns)):
        sum += weighted_topics_df[topics_columns[i]]

    weighted_topics_df['sums'] = sum

    weighted_topics_columns = genWeightedTopicsColumns(topics_count)

    for i in range( len(topics_columns)):
        weighted_topics_df[weighted_topics_columns[i]] = weighted_topics_df[cols[i+1]] / weighted_topics_df['sums']  #* idf

    weighted_topics_df['term_occurrences'] = term_occur.values()
    
    return weighted_topics_df, weighted_topics_columns



### example allSequences records:
# patient_id, LTC_abbrev
# 1000014,PMR
# 1000014,glaucoma
# 1000059,OA
# 1000059,OA
# 1000059,OA
# 1000059,skin_ulcer
# 1000059,dermatitis
# 1000062,spondylosis
# 1000062,obesity
# 1000062,urine_incont

def collectStages(LTCSeq):
    stages = []
    for i in range(len(LTCSeq)):
        stage = stages.append(LTCSeq[0:i+1])
        
#     print("collected stages: {}".format(stages))
    return stages


## return dict: {patientID:stages)}
## where stages is a list of lists
## example:
## {  1000014: [ [PMR], [PMR, glaucoma]], 1000059: [['OA'], ['OA', 'skin_ulcer'], ['OA', 'skin_ulcer', 'dermatitis']] , ... }
def computeStagesPerPatient(allSequences):

    attrs = ['patient_id', 'LTC_abbrev']
    simplified_sequences = allSequences[attrs]
    
    stagesPerPatient = dict()
    singleDiagnosis = 0  ## number of patients with 1 diagnosis only -- reported as these cannot be used
    
    ## construct stages for each patient -- assume records for same patient are consecutive
    last = len(simplified_sequences)  ## total records to scan
    print("{} records to scan".format(last))

    i = 0
    currPatientID = simplified_sequences.iloc[i]['patient_id']
    LTCSeq = []
    while i<last-1:   ## for each record
        LTC = simplified_sequences.iloc[i]['LTC_abbrev']

        ## avoid duplicates
        if not LTC in LTCSeq:
            LTCSeq.append(LTC)        
        
        i = i + 1
        newPatientID = simplified_sequences.iloc[i]['patient_id']
        
        if newPatientID != currPatientID:
#             print("\ncompleted sequence for patient {}: {}".format(currPatientID, LTCSeq))
            stagesPerPatient[currPatientID] = collectStages(LTCSeq)
            
            if (len(stagesPerPatient[currPatientID]) == 1):
                singleDiagnosis += 1 
                
            currPatientID = newPatientID
            LTCSeq = []

    print(" {} patients have one single stage:".format(singleDiagnosis))

    return stagesPerPatient


##########################
### compute the tensor that holds the patient-topic association __for incremental bag of terms in the patient's history__
##########################

# look up each term in bowStage. add up all partial association strength for each topic, generating an array of size K = number of topic
def assoc(bowstage, terms_topics_df, weighted_cols_names, topics_count):

    assocVector = np.zeros(topics_count)
    
#     print("associating stage {} with topics:".format(bowstage))
        
    for term in bowstage:
        
        ## find row in term_topics_df for this <term>
        row = terms_topics_df.loc[terms_topics_df['MLTC'] == term]
        
        ## <row> looks like 
        ## 0	MLTC	topics_0	topics_1	topics_2	topics_3	idf	sums	weighted_topics_0	weighted_topics_1	weighted_topics_2	weighted_topics_3	term_occurrences
        ## 3	CCD	0.000003	0.004850	0.000008	0.000018	2.002031	0.004879	0.000548	0.994170	0.001565	0.003717	9786
        
#         print("term {} found in row\n {}".format(term, row[weighted_cols_names]))
                 
        for i in range( len(weighted_cols_names)):
            assocVector[i] += row[weighted_cols_names[i]]
            
#         print("assoc vector: {}".format(assocVector))

    return assocVector


## corrected to work on actual sequences not BOWs!!

## stagesPerPatient looks like this
## dict: {patientID:stages)}
## where stages is a list of lists
## example:
## {  1000014: [ [PMR], [PMR, glaucoma]], 1000059: [['OA'], ['OA', 'skin_ulcer'], ['OA', 'skin_ulcer', 'dermatitis']] , ... }

## main method to compute the 'tensor' as a nested dictionary:
# bows = list(bow)
# bow = list(bowStage)
# bowStage = list(term)
# term --> association vector of size K = number of topics
# so:
#    all_patients_traj = { patientID: one_patient_trajectory}
#    one_patient_trajectory = { id(bowStage): <assoc vector>}
#  to use hashing we need to create one id for each stage in that patient's history.
#
# return all_patients_traj
#
def computeTrajectoryAssociations(stagesPerPatient, terms_topics_df, weighted_topics_columns, topics_count, limit=1000000):
    
    all_patients_traj = dict()  ## top level dict
    
    i = 0
    for patientID in stagesPerPatient.keys():   ## for each patient
        
        if i>=limit: 
            break
            
        singlePatientTrajectory = dict()  ## individual trajectory is itself a dict()
        
        bowStageId = 1
        stages = stagesPerPatient[patientID]  ## this is a nested list, each element is a BOW
        for stage in stages:   ## process one stage
            singlePatientTrajectory[bowStageId]  = assoc(stage, terms_topics_df, weighted_topics_columns, topics_count)
            bowStageId += 1
        all_patients_traj[patientID] = singlePatientTrajectory

#         print("vectors for patient {}".format(patientID))
#         pprint(singlePatientTrajectory)
        i += 1
        if i % 1000 == 0:
            print("{0} patients processed".format(i))

    return all_patients_traj





## computes the "resultant" of forces represented by the gravitational pull when the topics are represented as a "constellation" around the patient
## input assoc_vector v
## input number of topics topics_count
## return resultant magnitude R
## return resultant angle theta (in radians)

from math import sin, cos, degrees

## min len(v) = 2
def resultant(v):
    Rx = Ry = 0   
    N = len(v)
    alpha = 360 / N
    
#     print("alpha: {}".format(alpha))
    
    for i in range(len(v)):
        x = v[i] * cos(math.radians(alpha) * i)
        y = v[i] * sin(math.radians(alpha) * i)
    
#         print("x = {0:.2f} y= {1:.2f}\n".format(x,y))

        Rx = Rx + x
        Ry = Ry + y
        
#     print("Rx = {0:.2f}, Ry ={1:.2f}".format(Rx, Ry))

    if abs(Rx) < 0.001: 
        Rx = 0
    if abs(Ry) < 0.001:
        Ry = 0 

    return math.sqrt(Rx**2 + Ry**2), degrees(math.atan2(Ry,Rx))
    

## report difference between two vectors

## assume len(v1) == len(v2)
def diff(v1, v2):
    v = []
    for  i in range(len(v1)):
        if (v1[i]==0):
            v.append(v2[i])
        else:
            v.append( (v2[i]-v1[i])/v1[i])
    return v

## example 
# diff( [1,3,0], [1.5,4,2])

def pprintAssocVector(vec):
    res = "["
    for i in range(len(vec)):
        res = res + "{0:.2}".format(vec[i])+ " "
    res = res +"]"
    return res



def pprint(trajectory):
    for bowStageId in trajectory.keys():
        print("stage {0}: {1}".format(bowStageId, trajectory[bowStageId]))
        
        



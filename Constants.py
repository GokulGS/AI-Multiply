## adjust this to your FS
HOME = '/Users/npm65/OneDrive - Newcastle University/NIHR-AI-MLTC-M/CODE/DPTA/'

DATA_PATH = HOME + 'data/'

# LTC_BINARY =  'https://newcastle-my.sharepoint.com/:u:/g/personal/npm65_newcastle_ac_uk/EVT0-AfnQclHg7dWUHCoNr0BaH5o1N8HV42LF5Z3jaPrYA?e=FEVcEO'
LTC_BINARY = DATA_PATH+'reference/ltc_matrix_binary_mm4.tsv'

RAW_MLTC_SEQUENCES = DATA_PATH+'reference/ltc_events_all_patients_ukbb45840.tsv'

## generated cached data -- common to all topic choices
TERMS_REL_WEIGHTS_IDF = DATA_PATH +'/terms_rel_weights_idf.csv'
BOWs = DATA_PATH + 'generated/BOWs.pkl'
LTCs = DATA_PATH + 'generated/LTCSs.pkl'
STAGES_PER_PATIENT = DATA_PATH + 'generated/SPP.pkl'

## these are specific for each number of clusters
LDA_MODEL = DATA_PATH + 'generated/lda_model.pkl'
ALL_TRAJECTORIES = DATA_PATH + 'generated/trajectories.pkl'
BOWID2BOW  = DATA_PATH + 'generated/BowId2bow.pkl'

TOPICS_COUNTS_PATH = DATA_PATH + 'generated/'
WORDCLOUD_FILES= 'wordCloud.png'
TERMS_TOPICS_FILES = 'termsForTopic.csv'

STAGES_PER_PATIENT = DATA_PATH + 'generated/SPP.pkl'

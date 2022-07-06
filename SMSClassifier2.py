from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
import nltk
import SMSModel as model

#============================================================================#

ridge_clf = RidgeClassifier(alpha=0.05)
logreg_clf = LogisticRegression(C=100, max_iter=50000, n_jobs=-1)
bern_nb_clf = BernoulliNB(alpha=0.05)
knn_clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
randfor_clf = RandomForestClassifier(min_samples_split=5, min_samples_leaf=2, n_jobs=-1)
nltk.download('stopwords')
        
#============================================================================#

def set_random_state(clf):
    try:
        r_s = int(input('Enter your favorite number: '))
    except ValueError:
        print('I specifically asked for a number, you must be a real nerd if you didn\'t type an integer.')
    clf.random_state = r_s
    return clf

def main(clf):
    sms = model.SMSModel('spam.csv', clf)
    sms.explore_data()
    input()
    sms.prepare_data([2, 3, 4])
    input()
    sms.explore_data()
    input()
    sms.visualize_data()
    input()
    sms.drop_redundant(400)
    input()
    sms.visualize_data()
    input()
    sms.prepare_text()
    input()
    sms.split_data()
    input()
    sms.train_model()
    input()
    
    del sms
    
if __name__ == "__main__":
    clf = set_random_state(randfor_clf)
    main(clf)    
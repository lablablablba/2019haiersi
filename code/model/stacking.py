from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import os

class Stacker(object):
    """
    """
    def __init__(self, X_train_path, y_train_path, X_test_path):
        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.X_test_path = X_test_path
        
    def _load(self,path):
        return joblib.load(path)
    
    def load_data_from_path(self, path):
        return np.array(self._load(path))
    
    def load_data(self):
        self.y_train = self.load_data_from_path(self.y_train_path)
        self.labels = set(self.y_train)
        self.n_classes = len(self.labels)
        
        pathes = os.listdir(self.X_train_path)
        self.X_train = np.zeros((len(self.y_train), len(pathes), self.n_classes))
        for i, path in enumerate(pathes):
            self.X_train[:,i,:] = self.load_data_from_path(self.X_train_path+path)
            
        pathes = os.listdir(self.X_test_path)
        n_test = len(self.load_data_from_path(self.X_test_path+pathes[0]))
        self.X_test = np.zeros((n_test, len(pathes), self.n_classes))
        for i, path in enumerate(pathes):
            self.X_test[:,i,:] = self.load_data_from_path(self.X_test_path+path)
            
        self.X_train = self.X_train.reshape((len(self.y_train), -1))
        self.X_test = self.X_test.reshape(n_test, -1)
    
    def get_result(self,X,y):
        print(classification_report(y, X, target_names=self.labels))
    
    def stack(self, clf, split=0.7, reload=True):
        if reload:
            self.load_data()            
        sep = int(len(self.y_train)*0.7)
        clf.fit(self.X_train[:sep], self.y_train[:sep])
        self.get_result(clf.predict(self.X_train[sep:]), self.y_train[sep:])
    
    def predict(self, clf, reload=False):
        if reload:
            self.load_data()
        clf.fit(self.X_train, self.y_train)
        return np.array(clf.predict(self.X_test))

if __name__ == '__main__':
    st = Stacker('./cache/stack/train/','cache/y_data.pkl','./cache/stack/test/')
    clf = ExtraTreesClassifier()
    st.stack(clf)
    preds = st.predict(clf)

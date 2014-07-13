#access to database

import tndj.setup
from tndj.scrapyitems.models import SelItem
from django.db.models import Max

import numpy as np
from sklearn.naive_bayes import *

from collections import defaultdict

#Helper class to manage indexes
class AutoInc(object):
    counter = -1
    action = None

    def __init__(self,action = None):
        self.action = action

    def __call__(self):
        self.counter+=1
        if self.action:
            self.action()
        return self.counter

#Class that hold the Classifier
class NBTrainedModel(object):
    def add_freq(self):
        self.freq += [[0] * self.nk]

    @property
    def nk(self):
        return len(self.kwset)

    @property
    def nc(self):
        return len(self.categories)

    def getCategory(self,index):
        for key,value in self.categories.iteritems():
            if index == value:
                return key
        return None

    def __init__(self,nbclass=MultinomialNB,nbargs={"alpha": 0.001}):
        self.categories = defaultdict(AutoInc(self.add_freq))
        self.kwset = defaultdict(AutoInc())
        self.catcount = defaultdict(int)
        self.freq = []
        self.nbclass = nbclass
        self.nbargs  = nbargs

    def classify(self, t,count_f=None,other_kw=10000):
        if count_f is None:
            count_f = lambda x,y:1

        nc = self.nc

        kws = t.selitemkeyword_set.all()
        nk1 = len(kws)
        # kws = [k for k in kws if k.DictWord_id in self.kwset]
        if not [k for k in kws if k.DictWord_id in self.kwset]:
            return None,0
        freq1 = [[0]*(nk1+1) for i in range(nc)]
        T = [0] * (nk1+1)
        p = 0
        for kw in kws:
            if kw.DictWord_id in self.kwset:
                i = self.kwset[kw.DictWord_id]
                for j in range(nc):
                    freq1[j][p] = self.freq[j][i]
            T[p] = count_f(t,kw)
            p += 1
        for j in range(nc):
            #freq1[j][nk1] = sum(freq[j])-sum(freq1[j])
            freq1[j][nk1] = other_kw(self,j) if callable(other_kw) else other_kw

        freq1 = np.array(freq1)
        print freq1,T
        X = np.array(reduce(lambda x, y: x+y, [list(np.eye(nk1+1)) for i in range(0, nc)]))
        Y = reduce(lambda x, y: x+y, [[i]*(nk1+1) for i in range(0, nc)])
        W = freq1.flat
        clf = self.nbclass(**self.nbargs)
        clf.fit(X, Y, W)
        cat_id = clf.predict(T)[0]
        p = clf.predict_proba(T)[0][cat_id]
        cat = self.getCategory(cat_id)
        return cat, p

# # #Create the training set and the test set given the need_id and the size of each set
# def makeSets(need_id,training,test=1000):
#     s = SelItem.objects.filter(
#         spiders__spidermaker__need_id = need_id,
#         spiders__spidermaker__Category1__gt=0
#         ).distinct()
#     size = s.count()
#     if size < training:
#         raise Exception("Not enough data (only %d headline)" % size)
#     test = min(size-training, test)
#     return s[0:training], s[training:training+test]

def makeSets(need_id,training,test=1000): #this code is for news with only one category.(replaced with the function above)
    s = SelItem.objects.filter(
        spiders__spidermaker__need_id = need_id,
        spiders__spidermaker__Category1__gt=0
        ).distinct()
    size = s.count()
    if size < training:
        raise Exception("Not enough data (only %d headline)" % size)
    test_s = min(size-training, test*2)         #test_s = 2000
    test_set = list(s[training:training+test_s]) # len(test_set) = 2000
    test_set = filter(lambda x:len(x.categories)==1,test_set)[:test]
    return s[0:training], test_set    



#Train the model given the training set and the need_id. Additionally get a count function
def SelectionCategorizeTraining(training, need_id,count_f=None):
    if count_f is None:
        count_f = lambda x,y : 1

    result = NBTrainedModel()
    # for i in training:
    #     for kw in i.keyword.all():
    #         if kw not in kwset:
    #           kwset[kw] = counter
    #           counter += 1

    c = 0
    for i in training:
        for kw in i.selitemkeyword_set.all():
            if count_f(i,kw):
                result.kwset[kw.DictWord_id]
        c+=1
        if c%50 == 0:
            print("Step1: Done %d"%c)
            
    #kwset = list(kwset) #create a list again because set has no indexing
    
    c=0
    for t in training:
        for cat in t.categories:
            if cat/10000 != (1200+need_id):
                continue
            result.catcount[cat] += 1 #add number of cat. to dictionary
            j = result.categories[cat]
            for kw in t.selitemkeyword_set.all():
                value = count_f(t,kw)
                if value:
                    i = result.kwset[kw.DictWord_id]
                    result.freq[j][i] += value
        c+=1
        if c%50 == 0:
            print("Step2: Done %d"%c)

    print(result.catcount)

    return result

#Run a test on the test sets (to implement calc on precision and recall)
def SelectionCategorizeTesting(test, model, count_f=None,other_kw=10000):
    catcount = defaultdict(int)
    #success = 0
    success = []
    fail = []
    fail2 = []

    for t in test:
        for cat in t.categories:
            catcount[cat] += 1
        cat, _ = model.classify(t, count_f, other_kw)
        if cat in t.categories:
            print ("Success %s -> %s" % (cat, t.categories))
            #success += 1
            success += [t]
        else:
            print ("Fail %s -> %s" % (cat, t.categories))
            fail += [t]
            fail2 += [cat]

    print(catcount)     

    #return success, len(fail), fail
    return len(success), success, len(fail), fail, fail2

#Additional functions
#Count functions
def count_relevant(n,kw):
    if kw.status == 1:
        return 1
    else:
        return 0

def count_rank(n,kw):
    return kw.rank

def count_rank_norm(n,kw):
    if not hasattr(n,'kw_rank_max'):
        n.kw_rank_max = n.selitemkeyword_set.aggregate(max=Max('rank'))['max']
    return 10.*kw.rank / n.kw_rank_max

#Other Keyword functions
def other_kw(model,cat):
    cat = model.getCategory(cat)
    return model.catcount[cat]

# #Examples

# # create the set for training Sports (need_id = 13) 
# # with 10000 element in the training set and 1000 in the test set
#training, test = makeSets(13,10000,1000)

# # train one plain model and one using the rank
#model      = SelectionCategorizeTraining(training,13)
# model_rank = SelectionCategorizeTraining(training,13,count_rank)
# model_rank = SelectionCategorizeTraining(training,13,count_rank_norm)


# # Check the models (don't mix differnt param)
# # failed_list contains the failed items
#success, fails, failed_list = SelectionCategorizeTesting(test, model)
# success, fails, failed_list = SelectionCategorizeTesting(test, model_rank, count_rank)
# success, fails, failed_list = SelectionCategorizeTesting(test, model_rank, count_rank_norm)
#success, fails, failed_list = SelectionCategorizeTesting(test, model, other_kw = other_kw)

# #to check the single item (e.g: t)
# model.classify(t)

def precision_recall (success_list , failed_list, model ):
    TP = 0
    FP = 0
    #TP = []
    #FP = []

    for cat in model.categories.keys():
        cat = set([cat])

        for i in success_list:
            if i.categories == cat: #success_list[i].categories is a set so , cat must also be a set
                TP += 1

        for i in failed_list:
            if i.categories == cat:
                FP += 1
    return TP, FP


#########added parts later########

 catcount = set([]) #create empty set

s = SelItem.objects.filter(
         spiders__spidermaker__need_id = need_id,
         spiders__spidermaker__Category1__gt=0
         ).distinct()

#getting the categories
 for i in s:
    for cat in i.categories:
        if cat / 10000 == 1213:
               catcount.add(cat)

#create a dictionary of categories
catcount_dict = {}

#create a lisr of catcount
catcount_list = list(catcount)


for cat in catcount_list:
    catcount_dict[cat] = []

#if you want to add to the dictionary later
#for cat in catcount_list:
#    catcount_dict[cat].append(2)
#deleting the dictionary
#del dict["name"]
#dict.clear()               #clear the values
#del dict

#getting each category as a vector in the dictionary
for i in s:
    for cat in i.categories:
        for eachcategory in catcount_list:
            if cat == eachcategory:
                catcount_dict[eachcategory].append(i)

#get the number of items in each category
for cat in catcount_list:
    print "sub-category",cat , "-->", len(catcount_dict[cat] )


#import random
#from copy import copy

# first step         y = copy(x)

# second Step        random.choice(y)

# third step         vector_name.remove()

#this is an example about taking random examples and removing from the list

# x = [1,2,3,4,5,6,7,8]
# y = copy(x)
# y
#  [1, 2, 3, 4, 5, 6, 7, 8]
# z = [] #create empty list
# z.append(random.choice(y))
# z
# [4]
# y.remove(z[0])
# y
# [1, 2, 3, 5, 6, 7, 8]
#  z.append(random.choice(y))
# z
#  [4, 8]
# y.remove(z[1])
# y
#  [1, 2, 3, 5, 6, 7]



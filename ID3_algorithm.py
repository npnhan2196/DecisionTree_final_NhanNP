'''
Created on Nov 3, 2022

@author: NhanNP
'''

import numpy as np
import pandas as pd

eps = np.finfo(float).eps

df = pd.read_csv('C:\\resources\\data_forecast1.csv')
# df = pd.read_csv('C:\\resources\\data_gold.csv')
# print(df)


##1. calculate entropy of the whole dataset

# entropy_node = 0  #Initialize Entropy
# values = df.Play.unique()  #Unique objects - 'Yes', 'No'
# for value in values:
#     fraction = df.Play.value_counts()[value]/len(df.Play)  #tỉ lệ trên tổng
#     print("Ti le phần tử", value, "trong collum target la: ",fraction)
#     entropy_node += -fraction*np.log2(fraction)
#
# print(f'Values cua attribute target la: {values}')
# print(f'entropy_node: {entropy_node}') 
#
#
# #2. Now define a function {ent} to calculate entropy of each attribute :
# def ent(df,attribute):
#     target_variables = df.Play.unique()  #This gives all 'Yes' and 'No'
#     variables = df[attribute].unique()   #This gives different features in that attribute 
#     entropy_attribute = 0
#     for variable in variables:
#         entropy_each_feature = 0
#         for target_variable in target_variables:
#             num = len(df[attribute][df[attribute]==variable][df.Play ==target_variable]) #Tử số
#             den = len(df[attribute][df[attribute]==variable]) #Mẫu số
#             fraction = num/(den+eps)  #pi
#             entropy_each_feature += -fraction*np.log2(fraction+eps)
#             # fraction = num/(den)  #pi
#             # entropy_each_feature += -fraction*np.log2(fraction) 
#         fraction2 = den/len(df)
#         entropy_attribute += -fraction2*entropy_each_feature   #Sums up all the entropy ETaste
#         # print("entropy của phần tử",variable, "trong atrribute",attribute, "la:", abs(entropy_attribute))
#     return(abs(entropy_attribute))
#
# a_entropy = {k:ent(df,k) for k in df.keys()[:-1]}
# print("Entropy cua tung attribute la: ",a_entropy)

print("Info tong quat phia duoi!!!==============================================================================================\n\n")


#1. Find the Entropy of whole dataset
def find_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    # print("values la: ", values)
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction) # -(play/tong)*log2(play/tong)+ -(don't play/tong)*log2(dont' play/tong)
    return entropy

#2. Define a function to calculate entropy of each attribute :
def find_entropy_attribute(df,attribute):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()  #This gives all 'Play' and 'Don't Play
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable]) #Numerator
            den = len(df[attribute][df[attribute]==variable]) #Denominator
            fraction = num/(den+eps)
            entropy += -fraction*np.log2(fraction+eps)
        fraction2 = den/len(df)
        entropy2 += -fraction2*entropy
    return (abs(entropy2))

def ig(e_dataset,e_attr):
    return(e_dataset-e_attr)

#4.Find attribute which has maximum IG:
def find_winner(df):
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    max = df.keys()[:-1][np.argmax(IG)]
    return max

def get_subtable(df, node,value):
    return df[df[node] == value].reset_index(drop=True)

# Make a DEcision Tree visualize
def buildTree(df,tree=None): 
    Class = df.keys()[-1]
    #Get attribute with maximum information gain
    node = find_winner(df)
    # node_root_temp =[]
    # node_root_temp.append(find_winner(df)) 
    attValue = np.unique(df[node])
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    #Make loop to construct a tree by calling this function recursively. 
    #Check if the subset is pure and stops if it is pure. 
    for value in attValue:
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable[Class],return_counts=True)                        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:
            tree[node][value] = buildTree(subtable) #Calling the function recursively   
            
    return tree



print("\n\nGiá Trị của Entropy_node", df.keys()[-1], "là:",find_entropy(df),"\n")
a_entropy2 = {k:find_entropy_attribute(df,k) for k in df.keys()[:-1]}
print("Entropy cua tung attribute la: ",a_entropy2,"\n")
IG = {k:ig(find_entropy(df),a_entropy2[k]) for k in a_entropy2}
print("Information Gain cua tung attribute là" , IG,"\n")
print("\n") 

t = buildTree(df)
import pprint
pprint.pprint(t)
print("\n")

#Write result to txt file
f = open("resultID3_v2.txt", "w")
f.write(str(f'entropy_node la: \n'))
f.write(str(find_entropy(df)))
f.write("\n\n")
f.write("Entropy cua tung attribute la: \n")
f.write(str(a_entropy2))
f.write("\n\n")
f.write("Information Gain là: \n")
f.write(str(IG))
f.write("\n\n")
f.write("Tree: \n")
f.write(str(t))
f.write("\n\n")

#Prediction######################################################################################################
dt2 = buildTree(df)
df2 = pd.DataFrame(data=[['rain', 'hot', 'high', 'strong']],columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])
# df2 = pd.DataFrame(data=[['CAO', 'ON DINH', 'THAP', 'THAP']],columns=['USD', 'Inflat (CPI)', 'Quantity Supply', 'Quantity exploit'])

def fun(d, t):
    """
    d -- decision tree dictionary
    t -- testing examples in form of pandas dataframe
    """
    res = []
    for _, e in t.iterrows():
        res.append(predict(d, e))
    return res

def predict(d, e):
    """
    d -- decision tree dictionary
    e -- a testing example in form of pandas series
    """
    current_node = list(d.keys())[0]
    current_branch = d[current_node][e[current_node]]
    # if leaf node value is string then its a decision
    if isinstance(current_branch, str):
        return current_branch
    # else use that node as new searching subtree
    else:
        return predict(current_branch, e)

print("Dự đoán quyết định của một điều kiện thời tiết: \n",df2 ,"\n\n Kết quả dự đoán là:",fun(dt2, df2))

f.write("Data used for prediction is: \n")
f.write(str(df2))
f.write("\n\n Prediction is: \n")
f.write(str(fun(dt2, df2)))
f.close()

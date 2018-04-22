import json
import pickle
import numpy as np
a = json.load(open("download.json"))
def processTitle():
	global a 
	arr = []
	for i in a["res"]:
		loc = i["location"].lower().split()
		name =  i["agent_name"].lower().split()
		print(loc)
		loc = [str(j).translate(None,'!, ') for j in loc]
		name = [str(k).translate(None,'!, ') for k in name]
		print(loc)
		val = [i["type"].lower(),i["area"].lower(),i["bedrooms"].lower(),i["bathrooms"].lower()]

		val.extend(loc)
		val.extend(name)
		arr.extend(val)
		arr = list(set(arr))
	val = dict(enumerate(arr))
	newVal = {}
	for key, value in val.items():
		newVal[value] = key
	pickle.dump(newVal,open('dictfile.p','wb'))	
	return newVal

def  createVec():
	bigarr = []
	arr = []
	global a
	arr = np.zeros(119)
	dicto = processTitle()
	for i in a["res"]:
		loc = i["location"].lower().split()
		name =  i["agent_name"].lower().split()
		print(loc)
		loc = [str(j).translate(None,'!, ') for j in loc]
		name = [str(k).translate(None,'!, ') for k in name]
		print(loc)
		val = [i["type"].lower(),i["area"].lower(),i["bedrooms"].lower(),i["bathrooms"].lower()]

		val.extend(loc)
		val.extend(name)
		arr = np.zeros(119)
		for k in val:
			print(dicto[k])
			arr[dicto[k]] = 1
		print("end")	
		bigarr.append(arr)
	pickle.dump(np.array(bigarr),open('titleVec.p','wb'))	
	return np.array(bigarr)

def priceDict():
	global a
	pDict ={}
	for i,j in enumerate(a["res"]): 
		pDict[i] = j["price"]
	return pDict	
	
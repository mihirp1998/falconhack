from fbchat import  Client, log
from fbchat.models import *
import apiai, codecs, json, pickle
import numpy as np

class Jarvis(Client):

    def apiaiCon(self):
        self.CLIENT_ACCESS_TOKEN = "f8dd1cd02dd549eda77d74e01d46106d"
        self.ai = apiai.ApiAI(self.CLIENT_ACCESS_TOKEN)
        self.request = self.ai.text_request()
        self.request.lang = 'de' #Default : English
        self.request.session_id = "<SESSION ID, UNIQUE FOR EACH USER>"
    def simmilarity(query, arr):
        arr = np.dot(query.T,arr)
        arg = np.argmax(arr)
        return arg
            
        

    def onMessage(self, author_id=None, message_object=None, thread_id=None, thread_type=ThreadType.USER, **kwargs):
        global dictVal
        global titleVal
        self.markAsRead(author_id)

        log.info("Message {} from {} in {}".format(message_object, thread_id, thread_type))
        msgText = message_object.text
        print(msgText)

        self.apiaiCon()

        msgText = message_object.text
 
        self.request.query = msgText

        response = self.request.getresponse().read()
        print(response)
        # try:
        #     if intention == "type description":

        # else if()
        arr = np.zeros(118)
        reader = codecs.getdecoder("utf-8")
        print(type(reader(response)))
        obj = json.loads(response)
        results = obj["result"]
        # try: 
        #     if results["metadata"]["intentName"] == "location":
        try:    
            if results["metadata"]["intentName"] == "location":
                ans = results["parameters"]["location"]["business-name"]
                arr[dictVal[ans]] = 1

            elif results["metadata"]["intentName"] == "area":
                ans = results["parameters"]["number"]
                arr[dictVal[ans]] = 1
            
            elif results["metadata"]["intentName"] == "bedbath":
                ans = results["parameters"]["num"]
                bed1 = ans[0]
                bathroom = ans[1]
                arr[dictVal[bed1]] = 1
                arr[dictVal[bathroom]] = 1

            elif results["metadata"]["intentName"] == "broker":
                ans = results["parameters"]["given_name"]
                arr[dictVal[ans]] = 1

            elif results["metadata"]["intentName"] == "type":
                ans = results["metadata"]["intentName"]
                arr[dictVal[ans]] = 1
        except Exception:
            print("no val")

        print(arr,"this is the array")    
        try:
            reply = obj["result"]["fulfillment"]["speech"]
        except Exception:
            reply = 'Are you looking for anythin else'
            print("not normal")
    
        if author_id!=self.uid:
            self.send(Message(text=reply), thread_id=thread_id, thread_type=thread_type)

        self.markAsDelivered(author_id, thread_id)


# Create an object of our class, enter your email and password for facebook.
client = Jarvis("sakshiraut15@yahoo.com", "helloworld")
dictVal = pickle.load(open('dictfile.p','rb'))
titleVal = pickle.load(open('titleVec.p','rb'))
# Listen for new message
client.listen()
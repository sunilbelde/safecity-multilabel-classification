import numpy as np
import flask
import pickle
from preprocessing import Preprocessing
from flask import request,render_template

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

model = pickle.load(open('lightgbm.pkl','rb'))

scalar=pickle.load(open('scalar.pkl','rb'))

preprocessor=Preprocessing()

def final_fun_1(queries):

  queries_vec=[]
  for query in queries:
    vector=preprocessor.get_vector_representation(query)
    if (vector[:300]==np.zeros(300)).sum()==300:
      return [[-1]]
    queries_vec.append(vector)
  queries_vec=np.asarray(queries_vec,)
  queries_vec[:,300:]=scalar.transform(queries_vec[:,300:])

  return model.predict(queries_vec)
  

@app.route('/',  methods =["GET", "POST"])
@app.route('/query',  methods =["GET", "POST"])
def predict():
  labels=['Commenting','Ogling/Facial Expressions/Staring','Touching /Groping']
  if request.method=='POST':
    query=request.form.get("query")
    predictions=final_fun_1([query])
    print(predictions[0])
    if predictions[0][0]!=-1:
      total=sum(predictions[0])
      return render_template('app.html', predictions=predictions[0],labels=labels,flag=True,query=query,total=total)
    else:
      return render_template("app.html",flag=False,labels=labels,text='Invalid story or could not categorize your story, are you sure your story is correct?')
  else:
    return render_template("app.html",flag=False,labels=labels,text='Write your story here : ')
 
if __name__ == "__main__": 
  app.run(host ='0.0.0.0', port = 5000, debug = True)
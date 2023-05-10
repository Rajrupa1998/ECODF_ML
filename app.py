from flask import Flask,render_template,request,jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import ml_classification_tasks
import ml_regression_tasks
from io import StringIO
import sys
import json
import seaborn as sb
import plotly
from io import StringIO
import base64
from sklearn import preprocessing
import plot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyJoules.energy_meter import measure_energy
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from sklearn.preprocessing import StandardScaler
from flask import Flask,render_template,request
import energy_measurement_main
import os
from werkzeug.utils import secure_filename
import measure_pandas
import measure_dask
import measure_vaex

app = Flask(__name__)




upload_file_location = "/home/rajrupa/Tool/Tool_demo/Dataset/"
app.config["upload_folder"] = upload_file_location

data_pandas=[]
data_dask=[]
data_vaex=[]


@app.route('/',methods=['GET','POST'])
def initialize():
    return render_template('landing_page.html')

@app.route('/dataframe',methods=['GET','POST'])
def initialize_df():
    row_list1=[]
    row_list2=[]
    if request.method == 'POST':
        library = request.form.get('library')
        library1 = request.form.get("library1")
        task = request.form.get('task')
        data_set = request.files["files1"]
        data_set.save(
            os.path.join(
                app.config["upload_folder"], secure_filename(data_set.filename)
            )
        )
        
        file_name=data_set.filename
        file_path=upload_file_location+str(file_name)

        #Adding conditions for load_csv operation
        if(task=="load_csv"):
            if(library=="pandas" or library1=="pandas"):
                df= measure_pandas.load_csv(path=file_path)
                data_pandas=energy_measurement_main.get_data1()
                
            if(library=="dask" or library1=="dask"):
                df= measure_dask.load_csv(path=file_path)
                data_dask=energy_measurement_main.get_data2()

            if(library=="vaex" or library1=="vaex"):
                df= measure_dask.load_csv(path=file_path)
                data_vaex=energy_measurement_main.get_data3()

        #Adding conditions for dropna operation   
        if(task=="dropna"):
            if(library=="pandas" or library1=="pandas"):
                df= measure_pandas.load_csv(path=file_path)
                measure_pandas.dropna(df)
                data_pandas=energy_measurement_main.get_data1()
                #data_pandas.insert(0,"Pandas")
            if(library=="dask" or library1=="dask"):
                df= measure_dask.load_csv(path=file_path)
                measure_dask.dropna(df)
                data_dask=energy_measurement_main.get_data2()
            if(library=="vaex" or library1=="vaex"):
                df= measure_vaex.load_csv(path=file_path)
                measure_vaex.dropna(df)
                data_vaex=energy_measurement_main.get_data3()

        #Adding conditions for fillna operation        
        if(task=="fillna"):
            if(library=="pandas" or library1=="pandas"):
                df= measure_pandas.load_csv(path=file_path)
                measure_pandas.fillna(df,val='0')
                data_pandas=energy_measurement_main.get_data1()
            if(library=="dask" or library1=="dask"):
                df= measure_dask.load_csv(path=file_path)
                measure_dask.fillna(df,val='0')
                data_dask=energy_measurement_main.get_data2()
            if(library=="vaex" or library1=="vaex"):
                df= measure_vaex.load_csv(path=file_path)
                measure_vaex.fillna(df, val='0')
                data_vaex=energy_measurement_main.get_data3()
                
        #Adding conditions for count operation
        if(task=="count"):
            if(library=="pandas" or library1=="pandas"):
                df= measure_pandas.load_csv(path=file_path)
                measure_pandas.count(df)
                data_pandas=energy_measurement_main.get_data1()
            if(library=="dask" or library1=="dask"):
                df= measure_dask.load_csv(path=file_path)
                measure_dask.count(df)
                data_dask=energy_measurement_main.get_data2()
            if(library=="vaex" or library1=="vaex"):
                df= measure_vaex.load_csv(path=file_path)
                measure_vaex.count(df)
                data_vaex=energy_measurement_main.get_data3()
        #Adding conditions for mean operation

        if(task=="mean"):
            if(library=="pandas" or library1=="pandas"):
                df= measure_pandas.load_csv(path=file_path)
                measure_pandas.mean(df)
                data_pandas=energy_measurement_main.get_data1()
            if(library=="dask" or library1=="dask"):
                df= measure_dask.load_csv(path=file_path)
                measure_dask.mean(df)
                data_dask=energy_measurement_main.get_data2()

        #Adding conditions for min operation
        if(task=="min"):
            if(library=="pandas" or library1=="pandas"):
                df= measure_pandas.load_csv(path=file_path)
                measure_pandas.min(df)
                data_pandas=energy_measurement_main.get_data1()
            if(library=="dask" or library1=="dask"):
                df= measure_dask.load_csv(path=file_path)
                measure_dask.min(df)
                data_dask=energy_measurement_main.get_data2()
            if(library=="vaex" or library1=="vaex"):
                df= measure_vaex.load_csv(path=file_path)
                measure_vaex.min(df)
                data_vaex=energy_measurement_main.get_data3()

        #Adding conditions for max operation
        if(task=="max"):
            if(library=="pandas" or library1=="pandas"):
                df= measure_pandas.load_csv(path=file_path)
                measure_pandas.max(df)
                data_pandas=energy_measurement_main.get_data1()
            if(library=="dask" or library1=="dask"):
                df= measure_dask.load_csv(path=file_path)
                measure_dask.max(df)
                data_dask=energy_measurement_main.get_data2()
            if(library=="vaex" or library1=="vaex"):
                df= measure_vaex.load_csv(path=file_path)
                measure_vaex.max(df)
                data_vaex=energy_measurement_main.get_data3()

        #Adding conditions for unique operation
        if(task=="unique"):
            if(library=="pandas" or library1=="pandas"):
                df= measure_pandas.load_csv(path=file_path)
                measure_pandas.unique(df)
                data_pandas=energy_measurement_main.get_data1()
            if(library=="dask" or library1=="dask"):
                df= measure_dask.load_csv(path=file_path)
                measure_dask.unique(df)
                data_dask=energy_measurement_main.get_data2()
            if(library=="vaex" or library1=="vaex"):
                df= measure_vaex.load_csv(path=file_path)
                measure_vaex.unique(df)
                data_vaex=energy_measurement_main.get_data3()

        #Getting the time and energy consumption values
        pandas_avg=0
        dask_avg=0
        vaex_avg=0
        if (library == "pandas" or library1 == "pandas"):
            x=data_pandas[::-1]
            pandas_energy_value_list=[]
            i =0
            while(i<10):
                pandas_energy_value_list.append(x[i])
                i+=1
            pandas_energy_value_list.reverse()
            pandas_energy_value_list=pandas_energy_value_list[:5]
            pandas_energy_value_list.insert(0,"Pandas")
            sum=0
            for i in range(2,6):
                sum=sum+pandas_energy_value_list[i]
            pandas_avg=sum/4
            
            #row_list1=pandas_energy_value_list

        if (library == "dask" or library1 == "dask"):
            y=data_dask[::-1]
            dask_energy_value_list=[]
            j=0
            if(library=="vaex" or library1=="vaex"):
                
                while(j<10):
                    dask_energy_value_list.append(y[j])
                    j+=1
                dask_energy_value_list.reverse()
                dask_energy_value_list=dask_energy_value_list[:5]
                dask_energy_value_list.insert(0,"Dask")
                
            else:
                while(j<5):
                    dask_energy_value_list.append(y[j])
                    j+=1
                dask_energy_value_list.reverse()
                dask_energy_value_list.insert(0,"Dask")
            sum=0
            for i in range(2,6):
                sum=sum+dask_energy_value_list[i]
            dask_avg=sum/4
            

        if (library == "vaex" or library1 == "vaex"):
            x=data_vaex[::-1]
            vaex_energy_value_list=[]
            i =0
            while(i<5):
                vaex_energy_value_list.append(x[i])
                i+=1
            vaex_energy_value_list.reverse()
            vaex_energy_value_list.insert(0,"Vaex")
            sum=0
            for i in range(2,6):
                sum=sum+vaex_energy_value_list[i]
            vaex_avg=sum/4
        
        #Setting default values for the results
        time_result="Pandas is faster for the chosen operation"
        energy_result="Pandas is more energy efficient for the chosen operation"

        #Rending values based on the chosen task and libraries
        if((library=="pandas" or library1=="pandas") and (library=="vaex" or library1=="vaex")):
            if(pandas_avg>vaex_avg):
                energy_result="Vaex is more energy efficient for the chosen operation"
            if(pandas_energy_value_list[1]>vaex_energy_value_list[1]):
                time_result="vaex is faster"

            return render_template("homepage.html", list=pandas_energy_value_list,list1=vaex_energy_value_list,time_result= time_result,energy_result=energy_result)
        
        if((library=="pandas" or library1=="pandas") and (library=="dask" or library1=="dask")):
            if(pandas_avg>dask_avg):
                energy_result="Dask is more energy efficient for the chosen operation"
            if(pandas_energy_value_list[1]>dask_energy_value_list[1]):
                time_result="Dask is faster for the chosen operation"
            return render_template("homepage.html", list=pandas_energy_value_list,list1=dask_energy_value_list,time_result= time_result,energy_result=energy_result)
        
        else:
            if(vaex_avg>dask_avg):
                energy_result="Dask is more energy efficient for the chosen operation"
            if(vaex_avg<dask_avg):
                energy_result="Vaex is more energy efficient for the chosen operation"
            if(vaex_energy_value_list[1]< dask_energy_value_list[1]):
                time_result="Vaex is faster for the chosen operation"
            else:
                time_result="Dask is faster for the chosen operation"

            return render_template("homepage.html", list=dask_energy_value_list,list1=vaex_energy_value_list,time_result= time_result, energy_result=energy_result)
        
            
    return render_template("homepage.html")

col_names=[]
file_path=""
var_list=[]
col_names_list=[]
@app.route('/upload', methods=['GET', 'POST'])
def form_example():
   
    

    
    if request.method == 'POST':   
        plot_results = {} 
        data_set = request.files["files1"]
        data_set.save(
            os.path.join(
                app.config["upload_folder"], secure_filename(data_set.filename)
            )
        )
        
        
        file_name=data_set.filename
        file_path=upload_file_location+str(file_name)
        print(file_path)
        df = pd.read_csv(file_path)
        col_names=list(df.columns)
        var_list.append(file_path)
        col_names_list.append(col_names)
        df.replace("?", np.nan, inplace = True)
        df_new=df.apply(LabelEncoder().fit_transform)
        plot_results['heatmap'] = plt.generate_heatmap(df_new)
        jsonObject = jsonify(plot_results)
        print(jsonObject)
        #return jsonObject
        return render_template('mainpage.html', col_names=col_names,file_path=file_path,jsonObject=jsonObject)
        
        
       
    else:
        #print(col_names)
        col_names=[]
        file_path=""
        return render_template('mainpage.html', col_names=col_names,file_path=file_path)

inputArray=[]
outputParameter=""   
outVariableList=[]  
@app.route('/getData', methods=['GET', 'POST'])
def getList():
    if request.method == 'POST':   
        # file_name=var_list[0]
        # df=pd.read_csv(file_name)
        # dataplot = sb.heatmap(df.corr(), annot=True)
        # plt.show()
        inputArrayElements=request.form.getlist('myCheck[]')
        inputArray.append(inputArrayElements)
        #print(inputArray)
        outputParameter=request.form.get('outCheck')
        outVariableList.append(outputParameter)
        #print(outputParameter)
        #file_name=var_list[0]
        return render_template('model.html')
    else:
        return render_template('mainpage.html',col_names=col_names,file_path=file_path)

  
@app.route('/selectModel',methods=['GET', 'POST'])
def selectModel():
    if request.method == 'POST':  
        file_name=var_list[0]
        outputParameter=outVariableList[0]
        col_names=col_names_list
        # print("Here are the details....")
        # print(file_name)
        # print(outputParameter)
        # print(col_names)
        training_features=inputArray[0]
        # print(training_features)
        target=[outputParameter]
        #print(target)
        selected_model1=request.form.get('model1')
        selected_model2=request.form.get('model2')
        predict_selection_model1=selected_model1+"_prediction"
        predict_selection_model2=selected_model2+"_prediction"
        predict_selection_model1=predict_selection_model1.replace('test_', '')
        predict_selection_model2=predict_selection_model2.replace('test_', '')
        print(predict_selection_model1)
        print(predict_selection_model2)
        test_data_split=request.form.get('size')
        
        df = pd.read_csv(file_name)
       
        df.replace("?", np.nan, inplace = True)
        df[df==np.inf]=np.nan
        df.fillna(df.mean(), inplace=True)
        df_new=df.apply(LabelEncoder().fit_transform)
        X_train, X_test, Y_train, Y_test = train_test_split(df_new[training_features],
                                                    df_new[target],
                                                     test_size=float(test_data_split))
        

        sc_X = StandardScaler()
        sc_Y = StandardScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        Y_train_scaled= sc_Y.fit_transform(Y_train)
        Y_train_scaled_int=Y_train_scaled.astype(int)
       

        def regression_values(selected_model,predict_selection_model):
            result_list=[]
            
            if "support" or "neural" in selected_model:
                getattr(ml_regression_tasks,selected_model)(X_train_scaled,Y_train_scaled)
                energy_result_df=ml_regression_tasks.store_values()
                energy_result_list=[]
                energy_result_list=energy_result_df.values.tolist()
                print("check")
                print(energy_result_list)
                prediction_list=[]
                prediction_list=getattr(ml_regression_tasks,predict_selection_model)(X_train_scaled,Y_train_scaled,X_test,Y_test)
                print(prediction_list)
                result_list.append(energy_result_list)
                result_list.append(prediction_list)
            else:
                getattr(ml_regression_tasks,selected_model)(X_train,Y_train)
                energy_result_df=ml_regression_tasks.store_values()
                energy_result_list=[]
                energy_result_list=energy_result_df.values.tolist()
                print("check")
                print(energy_result_list)
                prediction_list=[]
                prediction_list=getattr(ml_regression_tasks,predict_selection_model)(X_train,Y_train,X_test,Y_test)
                print(prediction_list)
                result_list.append(energy_result_list)
                result_list.append(prediction_list)
            return result_list

        def classification_values(selected_model,predict_selection_model): 
            result_list=[]
            energy_result_list=[]
            getattr(ml_classification_tasks,selected_model)(X_train_scaled,Y_train_scaled_int)
            energy_result_df=ml_classification_tasks.store_values()
            energy_result_list=energy_result_df.values.tolist()
            print(energy_result_list)
            prediction_list=[]
            prediction_list=getattr(ml_classification_tasks,predict_selection_model)(X_train_scaled,Y_train_scaled_int,X_test,Y_test)
            print(prediction_list)
            result_list.append(energy_result_list)
            result_list.append(prediction_list)
            return result_list
            
        
        if "regression" in selected_model1 and "regression" in selected_model2 :
             result_list1=regression_values(selected_model1,predict_selection_model1)
             result_list2=regression_values(selected_model2,predict_selection_model2)
             
             return render_template('result_regression.html',energy_result_list1=result_list1[0][0],energy_result_list2=result_list2[0][1],prediction_list1=result_list1[1],prediction_list2=result_list2[1]) 
        
        if "regression" in selected_model1 and "classification" in selected_model2 :
             result_list1=regression_values(selected_model1,predict_selection_model1)
             result_list2=classification_values(selected_model2,predict_selection_model2)
             
             return render_template('result_classify.html',energy_result_list1=result_list1[0][0],energy_result_list2=result_list2[0][0],prediction_list1=result_list1[1],prediction_list2=result_list2[1]) 
        
        if "classification" in selected_model1 and "regression" in selected_model2 :
             result_list1=classification_values(selected_model1,predict_selection_model1)
             result_list2=regression_values(selected_model2,predict_selection_model2)
             return render_template('result_classify.html',energy_result_list1=result_list1[0][0],energy_result_list2=result_list2[0][0],prediction_list2=result_list1[1],prediction_list1=result_list2[1]) 
        
        if "classification" in selected_model1 and "classification" in selected_model2 :
             result_list1=classification_values(selected_model1,predict_selection_model1)
             result_list2=classification_values(selected_model2,predict_selection_model2)
             print("check this...")
             print(result_list1[1])
             print(result_list2[1])
             return render_template('result_classification.html',energy_result_list1=result_list1[0][0],energy_result_list2=result_list2[0][1],prediction_list1=result_list1[1],prediction_list2=result_list2[1]) 
        
           
    
        
        else:
            return render_template('model.html',col_names=col_names,file_path=file_path) 

        
    

    
   

if __name__=="__main__":
    app.run(debug=True,port=int(os.getenv('PORT', 4444)))



#
# .\env\Scripts\activate.ps1
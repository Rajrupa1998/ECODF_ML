from flask import Flask,render_template,request

import energy_measurement_main
import os
from werkzeug.utils import secure_filename
import measure_pandas
import measure_dask
import measure_vaex
app = Flask(__name__)

upload_file_location = "/home/rajrupa/Tool/Tool_demo/"
app.config["upload_folder"] = upload_file_location


data_pandas=[]
data_dask=[]
data_vaex=[]
@app.route('/', methods=['GET', 'POST'])
def form_example():
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
                sum=sum+pandas_energy_value_list[i]
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
                sum=sum+pandas_energy_value_list[i]
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
   

if __name__=="__main__":
    app.run(debug=True)



#
# .\env\Scripts\activate.ps1
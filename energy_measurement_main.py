from flask import Blueprint, render_template
second = Blueprint("pra",__name__,static_folder="static",template_folder="templates")

import time
import pandas as pd

RAPL_API_DIR="/sys/class/powercap/intel-rapl"
package0_path=RAPL_API_DIR + '/intel-rapl:' + "0" + '/energy_uj'
dram0_path=RAPL_API_DIR + "/intel-rapl:"+ "0" + "/intel-rapl:"+ "0" +":"+"0"+"/energy_uj"
package1_path=RAPL_API_DIR + '/intel-rapl:' + "1" + '/energy_uj'
dram1_path=RAPL_API_DIR + "/intel-rapl:"+ "1" + "/intel-rapl:"+ "1" +":"+"0"+"/energy_uj"


value_list1=[]
value_list2=[]
value_list3=[]
def measure_energy(func):
    def wrapper(*args, **kwargs):
        package0_value=open(package0_path, 'r')
        package0_start=package0_value.readline()
        dram0_value=open(dram0_path, 'r')
        dram0_start=dram0_value.readline()
        dram1_value=open(dram1_path, 'r')
        dram1_start=dram1_value.readline()
        package1_value=open(package1_path, 'r')
        package1_start=package1_value.readline()
        start_time=time.time()

        func(*args, **kwargs)

        package0_value=open(package0_path, 'r')
        package0_end=package0_value.readline()
        dram0_value=open(dram0_path, 'r')
        dram0_end=dram0_value.readline()
        dram1_value=open(dram1_path, 'r')
        dram1_end=dram1_value.readline()
        package1_value=open(package1_path, 'r')
        package1_end=package1_value.readline()
        end_time=time.time()
        dram0=int(dram0_end)-int(dram0_start)
        dram1=int(dram1_end)-int(dram1_start)
        package0=int(package0_end)-int(package0_start)
        package1=int(package1_end)-int(package1_start)
        time_elapsed=end_time-start_time
        
    #    #print(dram0)
    #     #print(dram1)
    #     print(package0)
    #     print(package1)
        value_list1.append(time_elapsed)
        value_list1.append(package0)
        value_list1.append(package1)
        value_list1.append(dram0)
        value_list1.append(dram1)
        value_list2.append(time_elapsed)
        value_list2.append(package0)
        value_list2.append(package1)
        value_list2.append(dram0)
        value_list2.append(dram1)
        value_list3.append(time_elapsed)
        value_list3.append(package0)
        value_list3.append(package1)
        value_list3.append(dram0)
        value_list3.append(dram1)
        #print(value_list)
    return wrapper
        

@second.route("/csa")
def home():
    return(render_template("index.html"))




# @measure_energy
# def csv():
#    for i in range(20):
#     print("Hello")
#     print("Hi")
#     c=900+122
#     print(c)
    

def get_data1():
    return value_list1

def get_data2():
    return value_list2

def get_data3():
    return value_list3

def delete_data1():
    value_list1.clear()
    return value_list1

def delete_data2():
    value_list2.clear()
    return value_list2

def delete_data3():
    value_list2.clear()
    return value_list3







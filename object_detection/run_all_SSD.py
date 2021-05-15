import os

if __name__ == '__main__':
#    dataset_list=['Pathway1_1']
    dataset_list=['Pathway2_1','Pathway2_2','Pathway2_3','Doorway_1','Doorway_2','Doorway_3','Room_1','Room_2'
        ,'Staircase_1','Staircase_2','Staircase_3','Staircase_4','Bus_2','Bus_3']
    for dt in dataset_list:
        print(dt)
        os.system("python SSD_old.py -i /work/superorange5/MI3 -d "+dt+" -l -v")



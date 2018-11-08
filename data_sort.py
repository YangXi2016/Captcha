import os,time,cv2,re,shutil
from PIL import Image
import matplotlib.pyplot as plt
from os.path import join

DataSet_Path = "WEBLMT_dataset"
# label_filename = "result"+str(int(time.time()))+".txt"
label_filename = "result.txt"

def rename_pic_file_random():
    num = 0
    suffix = '.png'
    for _,_,files in os.walk(DataSet_Path):
        for filename in files: 
            # temp_uuid = uuid.uuid1().hex
            # new_filename = '_%s.png' % (temp_uuid)
            num +=1
            new_filename = str(num).zfill(4)+suffix
            os.rename(join(DataSet_Path,filename),join(DataSet_Path,new_filename))



def lable_pic_manul():
    pic_num = 803
    with open(join(DataSet_Path,label_filename),'r') as f:
        lines = f.readlines()
        data_errors = list(filter(lambda line:re.match(r'[A-Za-z0-9_]{4}\n',line)==None,lines))
        if len(data_errors) >0:
            raise RuntimeError('数据错误:\n'+"\n".join(data_errors))
        num = len(lines)
    for _,_,files in os.walk(DataSet_Path):
        i=0
        for filename in files:
            i+=1
            if i<=num:
                continue
            print("当前进度：%d / %d"%(i,pic_num)) 
            img_path = join(DataSet_Path,filename)
            img = cv2.imread(img_path)
            plt.imshow(img)
            plt.show()
            captcha = input("请输入验证码:\n")
            if captcha == '-1':
                break
            # lines.append(captcha+"\n")
            
            with open(join(DataSet_Path,label_filename),'a+') as f:
                f.write(captcha+"\n")

        # with open(join(DataSet_Path,label_filename),'w') as f:
        #     f.writelines(lines)
TestSet_Path = "WEBLMT_test/"
def rename_test_pic():
    for _,_,files in os.walk(TestSet_Path):
        for filename in files:
            old_path = join(TestSet_Path,filename)
            # filename = str(int(filename.split('.')[0])-600).zfill(4)+'.png'
            # new_path = join(TestSet_Path,filename)
            # os.rename(old_path,new_path)
            filename = filename.split('.')[0]+'_test.png'
            new_path = join(TestSet_Path,filename)
            shutil.copy(old_path,new_path)


# def add_break():
#     filename = join(DataSet_Path,label_filename)
#     with open(filename,'r') as f:
#         mystr = f.read()
#     with open(filename,'w') as f:
#         for i in range(int(len(mystr)/4)):
#             f.write(mystr[4*i:4+4*i]+'\n')

if __name__ == "__main__":
    # add_break()
    #rename_pic_file_random()
    # lable_pic_manul()
    # with open(join(DataSet_Path,label_filename),'r+') as f:
    #     str = f.read()
    #     str = str.lower()
    # with open(join(DataSet_Path,label_filename),'w+') as f:
    #     f.write(str)
    rename_test_pic()
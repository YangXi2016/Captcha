import  cv2,os
from PIL import Image
from os.path import join
import numpy as np

DataSet_Path = "WEBLMT_dataset/"
ClearLine_Path = "WEBLMT_clearline/"
ClearBackColor_Path = "WEBLMT_clearbackcolor/"
ClearNoise_Path = "WEBLMT_clearnoise/"
TrainSet_Path = "WEBLMT_train/"
TestSet_Path = "WEBLMT_test/"
Train_Divide_Path = "WEBLMT_train_divide/"
Test_Divide_Path = "WEBLMT_test_divide/"
Buffer_Path = "WEBLMT_buffer/"
label_filename = "result.txt"
suffix = '.png'
IMAGE_HEIGHT = 16
IMAGE_WEIGHT = 64

def clear_border(img,ratio):
  h, w = img.shape[:2]
  ROI_img = img[int(h*ratio[1]):int(h*(1-ratio[1])),int(w*ratio[0]):int(w*(1-ratio[0]))]

  return ROI_img

def get_dynamic_binary_image(img):
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰值化
  # 二值化
  th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  return th1



def get_line_position(line_img,backgroud_value = 255):
    lines = []
    h, w = line_img.shape[:2]
    for x in range(0, h):
        for y in range(0,w):
            if(line_img[x, y] != backgroud_value):
                lines.append(x)
                break
    return lines

def clear_line(src_img,lines):

    h,w = src_img.shape[:2]
    for x in lines:
        for y in range(w):
            src_img[x,y]=np.ceil((src_img[x-1,y]/2+src_img[x+1,y]/2))

    return src_img


def clear_line_auto(src):

    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_src = cv2.bitwise_not(gray_src)
    binary_src = cv2.adaptiveThreshold(gray_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # cv2.namedWindow("result image", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("result image", binary_src)
    # cv2.waitKey(0)

    # 提取水平线
    hline = cv2.getStructuringElement(cv2.MORPH_RECT, ((int(src.shape[1]*0.85)), 1), (-1, -1))
    temp = cv2.erode(binary_src, hline)
    dst = cv2.dilate(temp, hline)
    dst = cv2.bitwise_not(dst)
    # cv2.imshow("Final image", dst)
    # cv2.waitKey(0)

    lines = get_line_position(dst,backgroud_value=255)
    # print(lines)
    clear_line_img = clear_line(src,lines)

    return clear_line_img

# clear_line_auto()

def cutting_img(src_img):
    char_img_lists = []
    h,w = src_img.shape[:2]
    for i in range(4):
        ROI_img = src_img[:, int(w * i / 4):int(w * (i+1) / 4)]
        char_img_lists.append(ROI_img)
    return char_img_lists


def clear_noise(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 定义结构元素
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # 开运算
    # cv2.imshow('morph_close',closing)
    # cv2.waitKey(0)
    # filename = join(ProcedurePath,img_name.split('.')[0]+ 'morph_close.png')
    # cv2.imwrite(filename, closing)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 定义结构元素
    # dilation = cv2.dilate(closing, kernel)  # 腐蚀
    # cv2.imshow('dilation',dilation)
    # cv2.waitKey(0)
    # filename = join(ProcedurePath, img_name.split('.')[0]+'morph_dilation.png')
    # cv2.imwrite(filename, dilation)

    # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  # 开运算
    # cv2.imshow('morph_open',opening)
    # cv2.waitKey(0)
    # filename = join(ProcedurePath, img_name.split('.')[0]+'morph_open.png')
    # cv2.imwrite(filename, opening)
    
    return closing
    # return opening



def clear_backcolor_auto(img,frontcolor_num = 4):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    j =0
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = frontcolor_num
    ret,label,center=cv2.kmeans(Z,k,None,criteria,50,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image

    center = np.uint8(center)
    couts=[]
    for i in range(k):
        backcolor = [0,0,0]
        center = np.row_stack((center,backcolor))
        # position = np.where(label==i,i,k)
        couts.append(list(label.flatten()).count(i))
        # res = center[position.flatten()]
        # res2 = res.reshape((img.shape))
        # # temp_img = position.reshape(img.shape[:2])
        # # cv2.imshow("temp_img"+str(i),temp_img)
        # cv2.imwrite(str(i+1)+'.png',res2)
        # cv2.imshow("temp",res2)
        # cv2.waitKey(0)
    #print(couts)
    backcolor_index = couts.index(max(couts))
    #center = np.array([[0,0,0],[255,255,255]])
    position = np.where(label==backcolor_index,np.uint8(255),np.uint8(0))
    res = position.flatten()
    res2 = res.reshape((img.shape[:2]))

    # filename = join(ProcedurePath, img_name.split('.')[0] + '_clear_backcolor.png')
    # cv2.imwrite(filename,res2)
    # cv2.imshow('quondam image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return  res2

def divide_pic(Captcha_Path = TrainSet_Path,Divide_Path = Train_Divide_Path ,captcha_length=4,counts_file = "counts_file.txt"):
    # 在路径下新建文件夹，名字为a-z,0-9,用于存储分割后的验证码
    counts_recoder = {'total':0}
    for i in range(26):
        counts_recoder[chr(i+97)] = 0
        if not os.path.exists(Divide_Path + chr(i+97)):
            os.mkdir(Divide_Path + chr(i+97))
    for j in range(10):
        counts_recoder[str(j)] = 0
        if not os.path.exists(Divide_Path + str(j)):
            os.mkdir(Divide_Path + str(j))
    fp = open(Captcha_Path + "/result.txt") # 验证码训练集的答案路径
    divide_name = []
    for x in fp.readlines():
        divide_name.append(str(x).strip())
    fp.close()
    for i in range(1,len(os.listdir(Captcha_Path))):
        src_img_name = str(i).zfill(4) + suffix
        src_image = cv2.imread(join(Captcha_Path,src_img_name)) # 读取处理后的验证码
        for j in range(captcha_length):# 每张验证码有四个字符
            child_image = src_image[:,int(j*IMAGE_WEIGHT/4):int((j+1)*IMAGE_WEIGHT/4)] # 分割验证码图片（均分）
            save_path = join(Divide_Path,divide_name[i-1][j])
            save_img_name = str(i).zfill(4) + "-" + str(j) + suffix
            cv2.imwrite(join(save_path,save_img_name),child_image) # 存储分割后的图片

            counts_recoder[divide_name[i-1][j]]+=1
            counts_recoder['total'] +=1 
    with open(join(Divide_Path,counts_file),'w+') as f:
        for k,v in counts_recoder.items():
            f.write(k+':'+str(v)+'\n')


def temp_test():
    for _,_,files in os.walk(DataSet_Path):
        for filename in files:
            if filename.split('.')[1]!='png':
                continue
            
            # #获取降噪前图片
            # clearbackcolor_img_path = join(ClearBackColor_Path,filename)
            # clearbackcolor_img = cv2.imread(clearbackcolor_img_path)

            # #测试函数处理后保存
            # buffer_img_path =join(Buffer_Path,filename)
            # buffer_img = clear_noise(clearbackcolor_img)
            # cv2.imwrite(buffer_img_path,buffer_img)

            #转为二值化图像（PIL型）
            # need_img_path = join(Captcha_Path,filename)
            # need_img = Image.open(need_img_path)
            # need_img = need_img.convert('1')
            # need_img.save(need_img_path)

if __name__ == "__main__":
    temp_test()
    divide_pic(Captcha_Path=TrainSet_Path,Divide_Path=Train_Divide_Path)
    # divide_pic(Captcha_Path=TestSet_Path,Divide_Path=Test_Divide_Path)
    input("pause")
    for _,_,files in os.walk(DataSet_Path):
        for filename in files:
            if filename.split('.')[1]!='png':
                continue
            
            #获取原图像
            img_path = join(DataSet_Path,filename)
            img = cv2.imread(img_path)

            #祛除干扰线
            clearline_img_path = join(ClearLine_Path,filename)
            clearline_img = clear_line_auto(img)

            cv2.imwrite(clearline_img_path,clearline_img)

            #祛除背景色
            clearbackcolor_img_path = join(ClearBackColor_Path,filename)
            clearbackcolor_img = clear_backcolor_auto(clearline_img)
            cv2.imwrite(clearbackcolor_img_path,clearbackcolor_img)

            #降噪处理
            clearnoise_img_path = join(ClearNoise_Path,filename)
            clearnoise_img = clear_noise(clearbackcolor_img)
            cv2.imwrite(clearnoise_img_path,clearnoise_img)

            #转为二值化图像（PIL型）
            need_img_path = join(Captcha_Path,filename)
            need_img = Image.Open(need_img_path)
            need_img = need_img.convert('1')
            need_img.save()


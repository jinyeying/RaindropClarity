import os
import cv2
import numpy as np
from PIL import Image

def getimgarr(path_1):
    dir_1 = sorted(os.listdir(path_1))
    imgs = []
    for j in range(len(dir_1)):
        img = cv2.imread(path_1 + '/' +dir_1[j])
        imgs.append(img)
    imgs = np.array(imgs, dtype=float)

    return imgs

#inp_path = '/disk1/release/DayRainDrop/Clear'
#inp_path = '/disk1/release/NightRainDrop/Clear'
#inp_path = '/disk1/release/DayRainDrop_Test/Clear'
inp_path = '/disk1/release/NightRainDrop_Test/Clear'

countall = 0
countbf = 0
countrf = 0

sid = sorted(os.listdir(inp_path))
print(sid,len(sid))

for i in range(len(sid)):

    inp_path_clean = os.path.join(inp_path, sid[i])
    inp_path_blur = inp_path_clean.replace('/Clear/','/Blur/')
    #print(inp_path_clean,inp_path_blur)

    dropimgs = sorted(os.listdir(inp_path_clean))
    Droplist = []
    for didx in range(len(dropimgs)):
        if dropimgs[didx].endswith('.png'):
            Droplist.append(dropimgs[didx])
    # print(Droplist)
    # print('-------------------------------------------------')

    for j in range(len(Droplist)):
        cleaninp_frame_name = os.path.join(inp_path_clean, Droplist[j])
        blurinp_frame_name = cleaninp_frame_name.replace('/Clear/','/Blur/')
        # print(cleaninp_frame_name,blurinp_frame_name)
        countall = countall + 1
        cleanimage = Image.open(cleaninp_frame_name)
        blurimage = Image.open(blurinp_frame_name)
        if cleanimage == blurimage:
            countbf = countbf + 1
            parts = cleaninp_frame_name.split(os.sep)
            new_path = os.path.join(parts[-2], parts[-1])
            print(new_path)
        else:
            countrf = countrf + 1
print('-countall-',countall,'-background focus-',countbf,'-raindrop focus-',countrf)

#DayRainDrop         -countall- 5442,-background focus- 1836,-raindrop focus- 3606,
#NightRainDrop       -countall- 9744,-background focus- 4906,-raindrop focus- 4838,
#Total: 15186

#DayRainDrop_Train   -countall- 8655,-background focus- 4143,-raindrop focus- 4512
#DayRainDrop_Test    -countall- 729 -background focus- 261 -raindrop focus- 468
#NightRainDrop_Train -countall- 4713,-background focus- 1575,-raindrop focus- 3138
#NightRainDrop_Test  -countall- 1089 -background focus- 763 -raindrop focus- 326



